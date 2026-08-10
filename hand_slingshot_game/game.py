"""Pygame screens, visuals, and physics for the hand slingshot game."""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pygame

# Tunable game values.
WIDTH, HEIGHT, FPS = 1280, 720, 60
GROUND_Y = 650
SLING = pygame.Vector2(185, 520)
PROJECTILE_RADIUS = 27
GRAVITY = 900.0
MAX_LAUNCH_SPEED = 1320.0       # 20% stronger than the previous 1100 px/s.
MIN_LAUNCH_SPEED = 246.0        # Matching 20% increase for low-power shots.
MAX_PULL_PIXELS = 114.0
DEFAULT_TIME_LIMIT = 60
RELOAD_DELAY = 0.70
MAX_CAMERA_INDEX = 5
ENEMY_FIRE_INTERVAL = 1.8
BALL_RESPAWN_DELAY = 0.85
BLOCK_GRAVITY = 1050.0
LEVEL_CLEAR_DELAY = 1.15
LEVEL_SCREEN_SECONDS = 5.0
TUTORIAL_START_SECONDS = 3.0

# Audio mix levels (0.0 silent to 1.0 full volume).
MENU_HOVER_VOLUME = .25
MENU_SELECT_VOLUME = .34
COLLISION_VOLUME = .44
MUSIC_VOLUME = .16

# Three-stage campaign. Blocks are (x, y, width, height, hp, style).
LEVELS = (
    ("TRAINING YARD", (
        (740, 620, 500, 30, 0, "support"),
        (760, 545, 170, 75, 1, "red"),
        (970, 480, 62, 140, 1, "pink"),
        (1060, 545, 150, 75, 1, "red"),
    )),
    ("LONG RANGE", (
        (920, 620, 350, 30, 0, "support"),
        (940, 545, 150, 75, 1, "red"),
        (1120, 460, 58, 160, 2, "pink"),
        (1190, 550, 78, 70, 1, "red"),
    )),
    ("TOWER SIEGE", (
        (900, 610, 300, 40, 0, "support"),
        (930, 475, 58, 135, 1, "pink"),
        (1100, 475, 58, 135, 1, "pink"),
        (920, 420, 250, 55, 1, "red"),
        (1000, 330, 84, 90, 2, "shooter"),
    )),
)

SKY, GROUND, INK = (104, 201, 243), (82, 172, 88), (25, 43, 62)
PURPLE, BLUE, YELLOW, RED, WHITE = (35, 15, 162), (66, 190, 255), (255, 228, 120), (255, 82, 67), (250, 253, 255)
ASSETS = Path(__file__).with_name("assets")


@dataclass
class Block:
    rect: pygame.Rect
    hp: int
    max_hp: int
    style: str
    velocity: pygame.Vector2 = field(default_factory=pygame.Vector2)
    position: pygame.Vector2 = field(init=False)

    def __post_init__(self) -> None:
        self.position = pygame.Vector2(self.rect.topleft)

    @property
    def static(self) -> bool:
        return self.style == "support"

    @property
    def target(self) -> bool:
        return not self.static


@dataclass
class Particle:
    position: pygame.Vector2
    velocity: pygame.Vector2
    life: float
    color: tuple[int, int, int]


@dataclass
class EnemyShot:
    position: pygame.Vector2
    velocity: pygame.Vector2
    life: float = 6.0


@dataclass
class Explosion:
    position: pygame.Vector2
    variant: int
    angle: float
    size: int
    life: float = .48
    duration: float = .48


class SlingshotGame:
    def __init__(self, camera_index: int = 0) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hand Slingshot")
        self.audio_enabled = False
        self.sounds: dict[str, pygame.mixer.Sound] = {}
        self.music_playing = False
        self._hovered_name: Optional[str] = None
        self._init_audio()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 21)
        self.large_font = pygame.font.Font(None, 48)
        self.title_font = pygame.font.Font(None, 70)
        self.mode = "menu"
        self.time_limit = DEFAULT_TIME_LIMIT
        self.mouse_mode = False
        self.show_tracer = True
        self.camera_index = camera_index
        self._pressed_name: Optional[str] = None
        self._pressed_until = 0
        self.menu_art = self._remove_outer_white(self._load_image("menu-buttons.png"))
        self.tutorial_art = self._remove_outer_white(self._load_image("tutorial-panel.png"))
        self.bird = self._load_bird()
        self.block_sprites = self._load_block_sprites()
        self.explosion_sprites = self._load_explosion_sprites()
        # Match the opening-screen artwork immediately, including before the
        # first frame is drawn, so no false hover sound plays at startup.
        self.start_rect = pygame.Rect(515, 165, 250, 290)
        self.settings_rect = pygame.Rect(370, 390, 270, 300)
        self.quit_rect = pygame.Rect(645, 390, 265, 300)
        self.play_rect = pygame.Rect(390, 625, 190, 50)
        self.camera_rect = pygame.Rect(595, 625, 190, 50)
        self.mouse_rect = pygame.Rect(800, 625, 190, 50)
        self.back_rect = pygame.Rect(1005, 625, 120, 50)
        self.settings_minus = pygame.Rect(155, 346, 54, 48)
        self.settings_plus = pygame.Rect(425, 346, 54, 48)
        self.tracer_rect = pygame.Rect(155, 430, 324, 48)
        self.camera_prev_rect = pygame.Rect(700, 539, 56, 48)
        self.camera_next_rect = pygame.Rect(1114, 539, 56, 48)
        self.settings_done = pygame.Rect(505, 625, 270, 50)
        self.retry_rect = pygame.Rect(475, 370, 150, 46)
        self.end_menu_rect = pygame.Rect(655, 370, 150, 46)
        self.next_level_rect = pygame.Rect(475, 385, 160, 48)
        self.level_menu_rect = pygame.Rect(655, 385, 160, 48)
        self.reset_tutorial()
        self.reset_campaign()

    def _init_audio(self) -> None:
        """Load optional audio without making a missing sound device fatal."""
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            sound_files = {
                "selecting": ("selecting.mp3", MENU_HOVER_VOLUME),
                "selected": ("selected.mp3", MENU_SELECT_VOLUME),
                "collision": ("collision.mp3", COLLISION_VOLUME),
            }
            for name, (filename, volume) in sound_files.items():
                sound = pygame.mixer.Sound(str(ASSETS / filename))
                sound.set_volume(volume)
                self.sounds[name] = sound
            self.music_path = ASSETS / "battle-theme.mp3"
            self.audio_enabled = True
        except (pygame.error, OSError) as exc:
            print(f"Audio unavailable; continuing silently: {exc}")

    def _play_sound(self, name: str) -> None:
        if self.audio_enabled and name in self.sounds:
            self.sounds[name].play()

    def _sync_music(self) -> None:
        if not self.audio_enabled:
            return
        # Keep the battle theme continuous through the short level-clear screen,
        # but stop it in menus and on the game-over screen.
        should_play = self.mode in ("playing", "level_clear")
        if should_play and not self.music_playing:
            try:
                pygame.mixer.music.load(str(self.music_path))
                pygame.mixer.music.set_volume(MUSIC_VOLUME)
                pygame.mixer.music.play(-1, fade_ms=500)
                self.music_playing = True
            except (pygame.error, OSError) as exc:
                print(f"Background music unavailable: {exc}")
                self.audio_enabled = False
        elif not should_play and self.music_playing:
            pygame.mixer.music.fadeout(350)
            self.music_playing = False

    def _press(self, name: str) -> None:
        self._pressed_name = name
        self._pressed_until = pygame.time.get_ticks() + 170

    def _is_popping(self, name: str) -> bool:
        return self._pressed_name == name and pygame.time.get_ticks() < self._pressed_until

    @staticmethod
    def _load_image(name: str) -> Optional[pygame.Surface]:
        path = ASSETS / name
        try:
            return pygame.image.load(path).convert_alpha()
        except pygame.error:
            print(f"Could not load asset: {path}")
            return None

    @staticmethod
    def _remove_outer_white(surface: Optional[pygame.Surface]) -> Optional[pygame.Surface]:
        """Remove edge-connected white canvas while preserving enclosed white art."""
        if surface is None:
            return None
        width, height = surface.get_size()
        pending = deque()
        visited: set[tuple[int, int]] = set()
        for x in range(width):
            pending.extend(((x, 0), (x, height - 1)))
        for y in range(height):
            pending.extend(((0, y), (width - 1, y)))
        while pending:
            x, y = pending.popleft()
            if (x, y) in visited or not (0 <= x < width and 0 <= y < height):
                continue
            visited.add((x, y))
            color = surface.get_at((x, y))
            if min(color.r, color.g, color.b) < 235:
                continue
            surface.set_at((x, y), (color.r, color.g, color.b, 0))
            pending.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
        return surface

    def _load_bird(self) -> Optional[pygame.Surface]:
        return self._remove_outer_white(self._load_image("red-bird.png"))

    def _load_block_sprites(self) -> dict[str, pygame.Surface]:
        sheet = self._remove_outer_white(self._load_image("blocks.png"))
        if sheet is None:
            return {}
        crops = {
            "red": pygame.Rect(0, 312, 390, 156),
            "pink": pygame.Rect(403, 54, 160, 483),
            "shooter": pygame.Rect(599, 215, 159, 155),
        }
        return {name: sheet.subsurface(rect).copy() for name, rect in crops.items()}

    def _load_explosion_sprites(self) -> list[pygame.Surface]:
        sheet = self._remove_outer_white(self._load_image("explosion.png"))
        if sheet is None:
            return []
        # The supplied sheet contains one large and one compact impact shape.
        crops = (pygame.Rect(0, 45, 525, 530), pygame.Rect(615, 0, 392, 555))
        return [sheet.subsurface(rect).copy() for rect in crops]

    def reset_campaign(self) -> None:
        self.level_index = 0
        self.score = 0
        self.total_shots = 0
        self.load_level(0)

    def reset_tutorial(self) -> None:
        self.tutorial_checks = [False, False, False, False]
        self.tutorial_countdown = 0.0

    def update_tutorial(self, hand_found: bool, state_name: str, pull: float, launched: bool, dt: float) -> None:
        """Validate the four gesture phases in order and auto-start the game."""
        if self.mode != "tutorial" or self.mouse_mode:
            return
        current = next((index for index, done in enumerate(self.tutorial_checks) if not done), None)
        conditions = (
            hand_found and state_name == "AIMING",
            state_name == "DRAWING",
            state_name == "DRAWING" and pull >= .35,
            launched or state_name == "RELEASED",
        )
        if current is not None and conditions[current]:
            self.tutorial_checks[current] = True
        if all(self.tutorial_checks):
            self.tutorial_countdown = self.tutorial_countdown or TUTORIAL_START_SECONDS
            self.tutorial_countdown = max(0.0, self.tutorial_countdown - dt)
            if self.tutorial_countdown <= 0:
                self.start_game()

    def load_level(self, index: int) -> None:
        self.level_index = index
        self.level_start_score = self.score
        self.level_start_total_shots = self.total_shots
        self.projectile = SLING.copy()
        self.velocity = pygame.Vector2()
        self.flying = False
        self.shots = 0
        self.timer = float(self.time_limit)
        self.particles: list[Particle] = []
        self.explosion: Optional[Explosion] = None
        self.trail: list[tuple[pygame.Vector2, float]] = []
        self.enemy_shots: list[EnemyShot] = []
        self.enemy_fire_timer = ENEMY_FIRE_INTERVAL
        self.level_clear_timer = 0.0
        self.level_transition_timer = LEVEL_SCREEN_SECONDS
        self.ball_destroyed = False
        self.respawn_message = "RELOADING"
        self.respawn_timer = 0.0
        self.flight_time = 0.0
        self.hit_blocks: set[int] = set()
        self.rest_time = 0.0
        self.blocks = [Block(pygame.Rect(x, y, w, h), hp, hp, style)
                       for x, y, w, h, hp, style in LEVELS[index][1]]

    def start_game(self) -> None:
        self.reset_campaign()
        self.mode = "playing"

    def retry_level(self) -> None:
        self.score = self.level_start_score
        self.total_shots = self.level_start_total_shots
        self.load_level(self.level_index)
        self.mode = "playing"

    def next_level(self) -> None:
        self.load_level(self.level_index + 1)
        self.mode = "playing"

    @staticmethod
    def aim_direction(aim) -> pygame.Vector2:
        horizontal = (float(aim[0]) - 0.5) * 46.0
        vertical = (0.5 - float(aim[1])) * 34.0
        degrees = max(-84.0, min(-18.0, -55.0 + horizontal + vertical))
        return pygame.Vector2(math.cos(math.radians(degrees)), math.sin(math.radians(degrees)))

    def set_aiming_projectile(self, aim, pull: float, drawing: bool) -> pygame.Vector2:
        direction = self.aim_direction(aim)
        if not self.flying:
            self.projectile = SLING - direction * (MAX_PULL_PIXELS * pull if drawing else 0)
        return direction

    def launch(self, direction: pygame.Vector2, power: float) -> None:
        if self.mode == "playing" and not self.flying and not self.ball_destroyed and power > 0.04:
            self.velocity = direction * max(MIN_LAUNCH_SPEED, MAX_LAUNCH_SPEED * power)
            self.flying = True
            self.shots += 1
            self.total_shots += 1
            self.flight_time = 0.0
            self.trail.clear()
            self.hit_blocks.clear()
            if self.level_index == 2:
                self.enemy_fire_timer = min(self.enemy_fire_timer, .85)

    def _burst(self, color: tuple[int, int, int]) -> None:
        for _ in range(15):
            angle, speed = random.uniform(0, math.tau), random.uniform(70, 250)
            self.particles.append(Particle(self.projectile.copy(), pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                                           random.uniform(.35, .70), color))

    def _trigger_explosion(self, position: pygame.Vector2) -> None:
        """Start one randomized impact animation; never stack explosions."""
        if not self.explosion_sprites or (self.explosion is not None and self.explosion.life > 0):
            return
        jitter = pygame.Vector2(random.uniform(-8, 8), random.uniform(-8, 8))
        self.explosion = Explosion(
            position.copy() + jitter,
            random.randrange(len(self.explosion_sprites)),
            random.uniform(-28, 28),
            random.randint(105, 150),
        )
        self._play_sound("collision")

    def _reload_sling(self) -> None:
        self.projectile = SLING.copy()
        self.velocity = pygame.Vector2()
        self.flying = False
        self.ball_destroyed = False
        self.hit_blocks.clear()
        self.enemy_shots.clear()
        self.enemy_fire_timer = ENEMY_FIRE_INTERVAL
        self.rest_time = 0.0

    def _destroy_ball(self, message: str, color: tuple[int, int, int]) -> None:
        if self.ball_destroyed:
            return
        self.ball_destroyed = True
        self.flying = False
        self.velocity = pygame.Vector2()
        self.trail.clear()
        self.respawn_timer = BALL_RESPAWN_DELAY
        self.respawn_message = message
        self._burst(color)

    def _update_enemy_fire(self, dt: float) -> None:
        shooters = [block for block in self.blocks if block.style == "shooter" and block.hp > 0]
        if not shooters or self.ball_destroyed or not self.flying:
            return
        self.enemy_fire_timer -= dt
        if self.enemy_fire_timer > 0:
            return
        shooter = shooters[0]
        origin = pygame.Vector2(shooter.rect.center)
        target = self.projectile.copy()
        direction = target - origin
        if direction.length_squared() > 0:
            direction = direction.normalize()
            self.enemy_shots.append(EnemyShot(origin, direction * 330.0))
        self.enemy_fire_timer = ENEMY_FIRE_INTERVAL

    def _update_enemy_shots(self, dt: float) -> None:
        for shot in self.enemy_shots[:]:
            shot.position += shot.velocity * dt
            shot.life -= dt
            if shot.life <= 0 or not (-40 <= shot.position.x <= WIDTH + 40 and -40 <= shot.position.y <= HEIGHT + 40):
                self.enemy_shots.remove(shot)
                continue
            if not self.ball_destroyed and shot.position.distance_to(self.projectile) <= PROJECTILE_RADIUS + 12:
                self.enemy_shots.remove(shot)
                self._destroy_ball("BIRD HIT - RELOADING", (173, 94, 255))
                break

    def _update_block_physics(self, dt: float) -> None:
        """Let damaged structures collapse onto other live blocks and supports."""
        # Destroyed blocks leave the structure completely; remaining blocks can
        # no longer land on them and therefore collapse into the empty space.
        dynamic = [block for block in self.blocks if not block.static and block.hp > 0]
        # Lower blocks settle first, giving blocks above a stable surface this frame.
        for block in sorted(dynamic, key=lambda item: item.rect.bottom, reverse=True):
            old_bottom = block.rect.bottom
            block.velocity.y += BLOCK_GRAVITY * dt
            block.position += block.velocity * dt
            block.rect.topleft = (round(block.position.x), round(block.position.y))
            landing_y = GROUND_Y
            for support in self.blocks:
                if support is block or (not support.static and support.hp <= 0):
                    continue
                horizontal_overlap = min(block.rect.right, support.rect.right) - max(block.rect.left, support.rect.left)
                if horizontal_overlap < min(18, block.rect.width // 3):
                    continue
                if old_bottom <= support.rect.top + 5 and block.rect.bottom >= support.rect.top:
                    landing_y = min(landing_y, support.rect.top)
            if block.rect.bottom >= landing_y and block.velocity.y >= 0:
                block.rect.bottom = landing_y
                block.position.update(block.rect.topleft)
                block.velocity.y = 0
                block.velocity.x *= .72
            elif block.hp <= 0:
                block.velocity.x *= .985

    def update(self, dt: float) -> None:
        self._sync_music()
        self._update_hover_sound()
        if self.explosion is not None:
            self.explosion.life -= dt
            if self.explosion.life <= 0:
                self.explosion = None
        for particle in self.particles[:]:
            particle.position += particle.velocity * dt
            particle.velocity.y += GRAVITY * .35 * dt
            particle.life -= dt
            if particle.life <= 0:
                self.particles.remove(particle)
        self.trail = [(position, life - dt * 2.6) for position, life in self.trail if life - dt * 2.6 > 0]
        if self.mode == "level_clear":
            self.level_transition_timer = max(0.0, self.level_transition_timer - dt)
            if self.level_transition_timer <= 0:
                if self.level_index + 1 < len(LEVELS):
                    self.next_level()
                else:
                    self.mode = "menu"
                    self.reset_tutorial()
            return
        if self.mode != "playing":
            return
        self.timer = max(0.0, self.timer - dt)
        if self.timer <= 0:
            self.mode = "lose"
            return
        self._update_block_physics(dt)
        if self.flying:
            self._update_enemy_fire(dt)
            self._update_enemy_shots(dt)
        elif self.enemy_shots:
            self.enemy_shots.clear()
        if self.ball_destroyed:
            self.respawn_timer -= dt
            if self.respawn_timer <= 0:
                self._reload_sling()
            return
        if not self.flying:
            if all(block.hp <= 0 for block in self.blocks if block.target):
                self.level_clear_timer += dt
                if self.level_clear_timer >= LEVEL_CLEAR_DELAY:
                    self.mode = "level_clear"
                    self.level_transition_timer = LEVEL_SCREEN_SECONDS
            return
        self.flight_time += dt
        self.velocity.y += GRAVITY * dt
        self.projectile += self.velocity * dt
        if self.show_tracer:
            self.trail.insert(0, (self.projectile.copy(), 1.0))
            self.trail = self.trail[:28]
        if self.projectile.y + PROJECTILE_RADIUS >= GROUND_Y:
            self.projectile.y = GROUND_Y - PROJECTILE_RADIUS
            self.velocity.y *= -.38
            self.velocity.x *= .72
            if abs(self.velocity.y) < 70:
                self.velocity = pygame.Vector2()
        ball = pygame.Rect(0, 0, PROJECTILE_RADIUS * 2, PROJECTILE_RADIUS * 2)
        ball.center = self.projectile
        # Colored targets get collision priority over the foundation beneath them.
        for index, block in enumerate(self.blocks):
            if block.target and block.hp > 0 and index not in self.hit_blocks and ball.colliderect(block.rect):
                self.hit_blocks.add(index)
                block.hp -= 1
                self.score += 100
                block.velocity.x += self.velocity.x * .16
                block.velocity.y = min(block.velocity.y, self.velocity.y * .08 - 55)
                self._trigger_explosion(self.projectile)
                self._destroy_ball("HIT CONFIRMED - RELOADING",
                                   YELLOW if block.hp == 0 else (238, 165, 74))
                break
        if not self.ball_destroyed:
            for index, block in enumerate(self.blocks):
                if block.static and index not in self.hit_blocks and ball.colliderect(block.rect):
                    self.hit_blocks.add(index)
                    self._trigger_explosion(self.projectile)
                    self._destroy_ball("SUPPORT HIT - RELOADING", (151, 105, 65))
                    break
        if all(block.hp <= 0 for block in self.blocks if block.target):
            self.level_clear_timer += dt
        elif (self.projectile.x > WIDTH + PROJECTILE_RADIUS
              or self.projectile.x < -PROJECTILE_RADIUS
              or self.projectile.y < -220
              or self.flight_time >= 8.0):
            self._destroy_ball("SHOT COMPLETE - RELOADING", WHITE)
        elif self.velocity.length_squared() < 1:
            self.rest_time += dt
            if self.rest_time >= RELOAD_DELAY:
                self._destroy_ball("SHOT COMPLETE - RELOADING", WHITE)
        else:
            self.rest_time = 0.0

    def _button_map(self):
        maps = {
            "menu": ((self.start_rect, "start"), (self.settings_rect, "settings"), (self.quit_rect, "quit")),
            "tutorial": ((self.play_rect, "play"), (self.camera_rect, "camera"),
                         (self.mouse_rect, "mouse"), (self.back_rect, "back")),
            "settings": ((self.settings_minus, "timer_minus"), (self.settings_plus, "timer_plus"),
                         (self.tracer_rect, "tracer"), (self.camera_prev_rect, "camera_prev"),
                         (self.camera_next_rect, "camera_next"), (self.settings_done, "done")),
            "win": ((self.retry_rect, "retry"), (self.end_menu_rect, "end_menu")),
            "lose": ((self.retry_rect, "retry"), (self.end_menu_rect, "end_menu")),
        }
        return maps.get(self.mode, ())

    def _update_hover_sound(self) -> None:
        hovered = next((name for rect, name in self._button_map()
                        if rect.collidepoint(pygame.mouse.get_pos())), None)
        if hovered != self._hovered_name:
            if hovered is not None:
                self._play_sound("selecting")
            self._hovered_name = hovered

    def handle_event(self, event: pygame.event.Event) -> Optional[str]:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            point = event.pos
            for rect, name in self._button_map():
                if rect.collidepoint(point):
                    self._press(name)
                    break
            return None
        if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
            return None
        point = event.pos
        if any(rect.collidepoint(point) for rect, _ in self._button_map()):
            self._play_sound("selected")
        if self.mode == "menu":
            if self.start_rect.collidepoint(point):
                self.mode = "tutorial"
                self.mouse_mode = False
                self.reset_tutorial()
                return "open_tutorial"
            elif self.settings_rect.collidepoint(point):
                self.mode = "settings"
                return "open_settings"
            elif self.quit_rect.collidepoint(point):
                return "quit"
        elif self.mode == "tutorial":
            if self.play_rect.collidepoint(point):
                self.start_game()
            elif self.camera_rect.collidepoint(point):
                self.mouse_mode = False
                return "enable_camera"
            elif self.mouse_rect.collidepoint(point):
                self.mouse_mode = True
            elif self.back_rect.collidepoint(point):
                self.mode = "menu"
        elif self.mode == "settings":
            if self.settings_minus.collidepoint(point):
                self.time_limit = max(30, self.time_limit - 15)
            elif self.settings_plus.collidepoint(point):
                self.time_limit = min(120, self.time_limit + 15)
            elif self.tracer_rect.collidepoint(point):
                self.show_tracer = not self.show_tracer
            elif self.camera_prev_rect.collidepoint(point):
                self.camera_index = (self.camera_index - 1) % (MAX_CAMERA_INDEX + 1)
                self.mouse_mode = False
                return "switch_camera"
            elif self.camera_next_rect.collidepoint(point):
                self.camera_index = (self.camera_index + 1) % (MAX_CAMERA_INDEX + 1)
                self.mouse_mode = False
                return "switch_camera"
            elif self.settings_done.collidepoint(point):
                self.mode = "menu"
        elif self.mode in ("win", "lose"):
            if self.retry_rect.collidepoint(point):
                self.start_game() if self.mode == "win" else self.retry_level()
            elif self.end_menu_rect.collidepoint(point):
                self.mode = "menu"
        return None

    def _text(self, value: str, pos, font=None, color=WHITE, center=False) -> None:
        image = (font or self.font).render(value, True, color)
        rect = image.get_rect(center=pos) if center else image.get_rect(topleft=pos)
        self.screen.blit(image, rect)

    def _button(self, rect: pygame.Rect, text: str, color=PURPLE, name: str = "") -> None:
        hovered = rect.collidepoint(pygame.mouse.get_pos())
        amount = 12 if self._is_popping(name) else (6 if hovered else 0)
        drawn = rect.inflate(amount, amount)
        shadow = drawn.move(0, 6)
        pygame.draw.rect(self.screen, (12, 8, 73), shadow, border_radius=11)
        pygame.draw.rect(self.screen, BLUE, drawn.inflate(4, 4), border_radius=11)
        pygame.draw.rect(self.screen, color, drawn, border_radius=9)
        font = self.small_font if len(text) > 13 else self.font
        self._text(text, drawn.center, font, WHITE, True)

    def _panel(self, rect: pygame.Rect, color=(25, 57, 85)) -> None:
        pygame.draw.rect(self.screen, color, rect, border_radius=14)
        pygame.draw.rect(self.screen, WHITE, rect, 2, border_radius=14)

    def _draw_ui_background(self) -> None:
        self.screen.fill((24, 14, 116))
        glow = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.circle(glow, (78, 59, 197, 110), (WIDTH // 2, 80), 470)
        pygame.draw.circle(glow, (51, 183, 255, 38), (1050, 570), 290)
        self.screen.blit(glow, (0, 0))

    def _draw_brand(self, section: str) -> None:
        self._text("HAND SLINGSHOT", (45, 30), self.large_font, WHITE)
        section_image = self.font.render(section, True, BLUE)
        self.screen.blit(section_image, section_image.get_rect(topright=(WIDTH - 45, 42)))
        pygame.draw.line(self.screen, (88, 77, 195), (45, 92), (WIDTH - 45, 92), 2)

    def _draw_menu(self) -> None:
        self._draw_ui_background()
        self._text("WEBCAM ARCADE", (WIDTH // 2, 32), self.small_font, BLUE, True)
        self._text("HAND SLINGSHOT", (WIDTH // 2, 86), self.title_font, WHITE, True)
        self._text("Aim with your hand. Clear the range.", (WIDTH // 2, 126), self.small_font, (210, 211, 255), True)
        art_rect = pygame.Rect(370, 145, 540, 559)
        if self.menu_art:
            scaled_art = pygame.transform.smoothscale(self.menu_art, art_rect.size)
            self.screen.blit(scaled_art, art_rect)
        # Transparent click areas line up with the three user-designed hexagons.
        self.start_rect = pygame.Rect(art_rect.x + 145, art_rect.y + 20, 250, 290)
        self.settings_rect = pygame.Rect(art_rect.x, art_rect.y + 245, 270, 300)
        self.quit_rect = pygame.Rect(art_rect.x + 275, art_rect.y + 245, 265, 300)
        mouse = pygame.mouse.get_pos()
        for rect, name in ((self.start_rect, "start"), (self.settings_rect, "settings"), (self.quit_rect, "quit")):
            if not self.menu_art or (not rect.collidepoint(mouse) and not self._is_popping(name)):
                continue
            relative = rect.move(-art_rect.x, -art_rect.y).clip(pygame.Rect((0, 0), art_rect.size))
            crop = scaled_art.subsurface(relative).copy()
            amount = 18 if self._is_popping(name) else 10
            raised = pygame.transform.smoothscale(crop, (relative.w + amount, relative.h + amount))
            raised_rect = raised.get_rect(center=rect.center)
            pygame.draw.polygon(self.screen, YELLOW, (
                (rect.centerx, rect.top - 5), (rect.right + 4, rect.y + rect.h // 4),
                (rect.right + 4, rect.y + rect.h * 3 // 4), (rect.centerx, rect.bottom + 5),
                (rect.left - 4, rect.y + rect.h * 3 // 4), (rect.left - 4, rect.y + rect.h // 4)), 5)
            self.screen.blit(raised, raised_rect)
        self._text("Click a button to begin", (WIDTH // 2, 700), self.small_font, (210, 211, 255), True)

    def _draw_camera(self, frame, rect: pygame.Rect) -> None:
        pygame.draw.rect(self.screen, BLUE, rect.inflate(8, 8), border_radius=14)
        if frame is None:
            pygame.draw.rect(self.screen, (18, 12, 94), rect, border_radius=12)
            self._text("NO CAMERA PREVIEW", rect.center, self.font, (190, 194, 235), True)
            return
        height, width = frame.shape[:2]
        surface = pygame.image.frombuffer(np.ascontiguousarray(frame[:, :, ::-1]).tobytes(), (width, height), "RGB")
        self.screen.blit(pygame.transform.smoothscale(surface, rect.size), rect)

    def _draw_tutorial(self, frame, camera_ok: bool) -> None:
        self._draw_ui_background()
        self._draw_brand("HOW TO PLAY")
        self._panel(pygame.Rect(55, 125, 560, 460), (49, 38, 151))
        self._panel(pygame.Rect(650, 125, 575, 460), (31, 27, 122))
        self._text("LEARN THE GESTURES", (90, 158), self.font, YELLOW)
        instructions = (
            ("1", "SHOW YOUR HAND", "Hold your palm clearly inside the frame."),
            ("2", "MOVE HAND BACK", "Move away from the camera to start drawing."),
            ("3", "CHARGE THE SHOT", "Keep moving back until the pull is strong."),
            ("4", "MOVE FORWARD", "Move toward the camera to launch."),
        )
        for index, (number, heading, detail) in enumerate(instructions):
            y = 205 + index * 84
            complete = self.tutorial_checks[index]
            active = index == next((i for i, done in enumerate(self.tutorial_checks) if not done), -1)
            circle_color = (70, 211, 126) if complete else (YELLOW if active else BLUE)
            pygame.draw.circle(self.screen, circle_color, (105, y + 18), 22)
            if complete:
                pygame.draw.line(self.screen, WHITE, (95, y + 18), (102, y + 26), 4)
                pygame.draw.line(self.screen, WHITE, (102, y + 26), (116, y + 10), 4)
            else:
                self._text(number, (105, y + 18), self.font, PURPLE, True)
            self._text(heading, (145, y), self.font, YELLOW if active else WHITE)
            self._text(detail, (145, y + 30), self.small_font, (211, 214, 247))
        self._text("CAMERA CHECK", (685, 158), self.font, YELLOW)
        camera_preview = pygame.Rect(685, 195, 505, 330)
        self._draw_camera(frame, camera_preview)
        state = "CAMERA READY" if camera_ok else ("MOUSE MODE" if self.mouse_mode else "CAMERA OFF")
        badge = pygame.Rect(850, 538, 175, 30)
        pygame.draw.rect(self.screen, (27, 145, 100) if camera_ok else (107, 83, 164), badge, border_radius=15)
        self._text(state, badge.center, self.small_font, WHITE, True)
        if all(self.tutorial_checks):
            seconds = max(1, math.ceil(self.tutorial_countdown))
            self._text(f"ALL SET - STARTING IN {seconds}", (335, 574), self.font, (112, 255, 174), True)
        self._button(self.play_rect, "START GAME", PURPLE, "play")
        self._button(self.camera_rect, "ENABLE CAMERA", (79, 81, 185), "camera")
        self._button(self.mouse_rect, "USE MOUSE", (79, 81, 185), "mouse")
        self._button(self.back_rect, "BACK", (79, 81, 185), "back")

    def _draw_settings(self, frame, camera_ok: bool) -> None:
        self._draw_ui_background()
        self._draw_brand("SETTINGS")
        self._panel(pygame.Rect(70, 125, 500, 470), (49, 38, 151))
        self._panel(pygame.Rect(610, 125, 600, 470), (31, 27, 122))
        self._text("GAMEPLAY", (105, 160), self.font, YELLOW)
        self._text("ROUND TIMER", (155, 280), self.small_font, (196, 232, 255))
        self._button(self.settings_minus, "-", PURPLE, "timer_minus")
        self._button(self.settings_plus, "+", PURPLE, "timer_plus")
        self._text(f"{self.time_limit} SECONDS", (317, 370), self.large_font, YELLOW, True)
        self._button(self.tracer_rect, f"SHOT TRACER: {'ON' if self.show_tracer else 'OFF'}", (79, 81, 185), "tracer")
        self._text("CAMERA SOURCE", (645, 160), self.font, YELLOW)
        camera_preview = pygame.Rect(645, 200, 530, 315)
        self._draw_camera(frame, camera_preview)
        self._button(self.camera_prev_rect, "<", PURPLE, "camera_prev")
        self._button(self.camera_next_rect, ">", PURPLE, "camera_next")
        source = f"CAMERA {self.camera_index}"
        self._text(source, (935, 556), self.large_font, WHITE, True)
        camera_state = "LIVE PREVIEW" if camera_ok else "SOURCE UNAVAILABLE"
        self._text(camera_state, (935, 585), self.small_font, (126, 255, 185) if camera_ok else (255, 157, 157), True)
        self._button(self.settings_done, "SAVE & RETURN", PURPLE, "done")

    def _draw_background(self) -> None:
        self.screen.fill(SKY)
        for x, y, scale in ((130, 95, 1), (410, 180, .75), (780, 100, .6), (1100, 175, .8)):
            for dx, dy, radius in ((0, 10, 22), (28, 0, 30), (62, 10, 20)):
                pygame.draw.circle(self.screen, (236, 249, 255), (int(x + dx * scale), int(y + dy * scale)), int(radius * scale))
        pygame.draw.rect(self.screen, GROUND, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
        pygame.draw.rect(self.screen, (56, 145, 70), (0, GROUND_Y, WIDTH, 7))

    def _draw_block(self, block: Block) -> None:
        if block.static:
            pygame.draw.rect(self.screen, (76, 55, 35), block.rect.move(7, 7), border_radius=7)
            pygame.draw.rect(self.screen, (139, 94, 55), block.rect, border_radius=7)
            pygame.draw.rect(self.screen, (78, 50, 29), block.rect, 3, border_radius=7)
            pygame.draw.line(self.screen, (190, 137, 80), (block.rect.x + 12, block.rect.y + 9),
                             (block.rect.right - 12, block.rect.y + 9), 3)
            for x in range(block.rect.x + 45, block.rect.right, 62):
                pygame.draw.line(self.screen, (93, 61, 35), (x, block.rect.y + 4), (x, block.rect.bottom - 4), 2)
            return
        if block.hp <= 0:
            return
        shadow = block.rect.move(7, 8)
        pygame.draw.rect(self.screen, (49, 98, 62), shadow, border_radius=max(8, block.rect.height // 5))
        sprite = self.block_sprites.get(block.style)
        if sprite:
            image = pygame.transform.smoothscale(sprite, block.rect.size)
            self.screen.blit(image, block.rect)
        else:
            colors = {"red": (255, 87, 91), "pink": (215, 152, 232), "shooter": (133, 75, 255)}
            pygame.draw.rect(self.screen, colors.get(block.style, RED), block.rect,
                             border_radius=max(8, block.rect.height // 5))
            eye = (block.rect.x + block.rect.width // 4, block.rect.y + block.rect.height // 3)
            pygame.draw.circle(self.screen, WHITE, eye, min(block.rect.width, block.rect.height) // 5 + 4)
            pygame.draw.circle(self.screen, (0, 0, 0), eye, min(block.rect.width, block.rect.height) // 6)
        if block.hp < block.max_hp:
            pygame.draw.line(self.screen, (62, 32, 92), (block.rect.centerx, block.rect.y + 5),
                             (block.rect.centerx + 10, block.rect.centery), 4)
            pygame.draw.line(self.screen, (62, 32, 92), (block.rect.centerx + 10, block.rect.centery),
                             (block.rect.centerx - 7, block.rect.bottom - 5), 4)
        if block.max_hp > 1:
            bar_width = max(42, min(110, block.rect.width))
            bar = pygame.Rect(0, 0, bar_width, 9)
            bar.midbottom = (block.rect.centerx, block.rect.y - 5)
            pygame.draw.rect(self.screen, (31, 38, 54), bar, border_radius=5)
            ratio = max(0.0, block.hp / block.max_hp)
            health_color = (104, 229, 126) if ratio > .55 else ((255, 205, 79) if ratio > .25 else RED)
            fill = bar.copy()
            fill.width = max(3, round(bar.width * ratio))
            pygame.draw.rect(self.screen, health_color, fill, border_radius=5)
            pygame.draw.rect(self.screen, WHITE, bar, 1, border_radius=5)
        if block.style == "shooter" and block.hp > 0:
            glow = pygame.Surface(block.rect.inflate(18, 18).size, pygame.SRCALPHA)
            pygame.draw.rect(glow, (173, 94, 255, 60), glow.get_rect(), border_radius=18)
            self.screen.blit(glow, block.rect.inflate(18, 18))

    def _draw_bird(self) -> None:
        if self.ball_destroyed:
            return
        if self.bird:
            image = pygame.transform.rotozoom(self.bird, -self.velocity.angle_to(pygame.Vector2(1, 0)) * .18 if self.velocity else 0, .29)
            self.screen.blit(image, image.get_rect(center=self.projectile))
        else:
            pygame.draw.circle(self.screen, RED, self.projectile, PROJECTILE_RADIUS)

    def _draw_explosion(self) -> None:
        if self.explosion is None or not self.explosion_sprites:
            return
        effect = self.explosion
        source = self.explosion_sprites[effect.variant]
        progress = 1.0 - effect.life / effect.duration
        animated_size = effect.size * (.62 + .48 * min(1.0, progress * 2.2))
        ratio = animated_size / max(source.get_size())
        size = (max(1, round(source.get_width() * ratio)), max(1, round(source.get_height() * ratio)))
        image = pygame.transform.smoothscale(source, size)
        image = pygame.transform.rotozoom(image, effect.angle, 1.0)
        alpha = round(255 * (effect.life / effect.duration) ** .55)
        image.set_alpha(max(0, min(255, alpha)))
        self.screen.blit(image, image.get_rect(center=effect.position))

    def _draw_game(self, state_name: str, aim, pull: float, power: float, drawing: bool) -> None:
        self._draw_background()
        direction = self.set_aiming_projectile(aim, pull, drawing) if not self.flying else self.aim_direction(aim)
        if drawing and not self.flying:
            position, velocity = self.projectile.copy(), direction * max(MIN_LAUNCH_SPEED, MAX_LAUNCH_SPEED * power)
            for index in range(28):
                position += velocity * .08
                velocity.y += GRAVITY * .08
                pygame.draw.circle(self.screen, WHITE, position, 4 if index < 10 else 3)
            # Bands are visible only while the ball is held, never during flight.
            pygame.draw.line(self.screen, (80, 46, 29), (168, 502), self.projectile, 5)
            pygame.draw.line(self.screen, (80, 46, 29), (202, 502), self.projectile, 5)
        pygame.draw.line(self.screen, (106, 57, 30), (160, GROUND_Y), (182, 520), 20)
        pygame.draw.line(self.screen, (106, 57, 30), (182, 520), (208, GROUND_Y), 20)
        for position, life in reversed(self.trail):
            pygame.draw.circle(self.screen, WHITE, position, max(2, int(7 * life)))
        for block in self.blocks:
            self._draw_block(block)
        for shot in self.enemy_shots:
            direction = shot.velocity.normalize() if shot.velocity.length_squared() else pygame.Vector2()
            for index in range(4, 0, -1):
                point = shot.position - direction * index * 9
                pygame.draw.circle(self.screen, (191, 122, 255), point, index + 2)
            pygame.draw.circle(self.screen, (66, 14, 126), shot.position, 14)
            pygame.draw.circle(self.screen, (218, 151, 255), shot.position, 9)
            pygame.draw.circle(self.screen, WHITE, shot.position + pygame.Vector2(-3, -3), 3)
        for particle in self.particles:
            pygame.draw.circle(self.screen, particle.color, particle.position, max(2, int(5 * particle.life)))
        self._draw_explosion()
        self._draw_bird()
        self._panel(pygame.Rect(25, 22, 250, 72))
        self._text("SCORE", (45, 36), self.small_font, (190, 225, 245))
        self._text(str(self.score).zfill(4), (45, 55), self.large_font)
        self._panel(pygame.Rect(1002, 22, 255, 72))
        self._text("TIME", (1022, 36), self.small_font, (190, 225, 245))
        self._text(f"{math.ceil(self.timer)}s", (1022, 55), self.large_font, RED if self.timer < 15 else WHITE)
        remaining = sum(block.hp > 0 for block in self.blocks if block.target)
        self._text(f"BLOCKS {remaining}", (1135, 63), self.small_font, YELLOW)
        level_name = LEVELS[self.level_index][0]
        self._panel(pygame.Rect(466, 20, 348, 58))
        self._text(f"LEVEL {self.level_index + 1}/3", (490, 31), self.small_font, (190, 225, 245))
        self._text(level_name, (640, 55), self.font, YELLOW, True)
        self._panel(pygame.Rect(25, 115, 265, 56))
        self._text("PULL POWER", (45, 130), self.small_font, (190, 225, 245))
        pygame.draw.rect(self.screen, (21, 42, 62), (45, 148, 210, 12), border_radius=6)
        pygame.draw.rect(self.screen, RED, (45, 148, int(210 * power), 12), border_radius=6)
        if self.ball_destroyed:
            message = self.respawn_message
        elif self.level_index == 2 and any(block.style == "shooter" and block.hp > 0 for block in self.blocks):
            message = "PURPLE TOWER FIRES BACK"
        else:
            message = "MOVE FORWARD TO FIRE" if drawing else "MOVE HAND BACK TO CHARGE"
        self._text(message, (WIDTH // 2, 104), self.font, RED if self.ball_destroyed else INK, True)
        self._text("Automatic reload and level flow enabled", (25, 683), self.small_font, INK)

    def draw(self, state_name="IDLE", aim=(.5, .5), pull=0.0, power=0.0, drawing=False, frame=None, camera_ok=False) -> None:
        if self.mode == "menu":
            self._draw_menu()
        elif self.mode == "tutorial":
            self._draw_tutorial(frame, camera_ok)
        elif self.mode == "settings":
            self._draw_settings(frame, camera_ok)
        elif self.mode == "playing":
            self._draw_game(state_name, aim, pull, power, drawing)
        elif self.mode == "level_clear":
            self._draw_game(state_name, aim, pull, power, False)
            panel = pygame.Rect(0, 0, 580, 235)
            panel.center = (WIDTH // 2, 320)
            self._panel(panel, PURPLE)
            final_level = self.level_index == len(LEVELS) - 1
            title = "YOU WIN!" if final_level else f"LEVEL {self.level_index + 1} CLEAR!"
            self._text(title, (WIDTH // 2, 267), self.title_font, YELLOW, True)
            if final_level:
                detail = "CAMPAIGN COMPLETE"
                countdown = f"RETURNING TO MENU IN {max(1, math.ceil(self.level_transition_timer))}"
            else:
                detail = f"NEXT: {LEVELS[self.level_index + 1][0]}"
                countdown = f"LOADING IN {max(1, math.ceil(self.level_transition_timer))}"
            self._text(detail, (WIDTH // 2, 320), self.font, WHITE, True)
            self._text(countdown, (WIDTH // 2, 370), self.font, (147, 222, 255), True)
        elif self.mode in ("win", "lose"):
            self._draw_game(state_name, aim, pull, power, False)
            title = "YOU WIN!" if self.mode == "win" else "TIME'S UP"
            subtitle = f"SCORE {self.score}  |  {self.total_shots} SHOTS"
            panel = pygame.Rect(0, 0, 580, 220); panel.center = (WIDTH // 2, 310)
            self._panel(panel, PURPLE)
            self._text(title, (WIDTH // 2, 260), self.title_font, YELLOW if self.mode == "win" else WHITE, True)
            self._text(subtitle, (WIDTH // 2, 315), self.font, WHITE, True)
            self._button(self.retry_rect, "PLAY AGAIN", PURPLE, "retry")
            self._button(self.end_menu_rect, "MAIN MENU", (79, 81, 185), "end_menu")
        else:
            self.screen.fill(PURPLE)
            self._text("THANKS FOR PLAYING", (WIDTH // 2, HEIGHT // 2), self.large_font, WHITE, True)
        pygame.display.flip()

    def close(self) -> None:
        if pygame.mixer.get_init() is not None:
            pygame.mixer.music.stop()
            pygame.mixer.stop()
        pygame.quit()
