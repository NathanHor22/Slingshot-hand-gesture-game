"""Pygame presentation and small, self-contained 2D slingshot physics."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

# Game values are pixels / seconds and intentionally easy to tune.
WIDTH, HEIGHT, FPS = 1280, 720, 60
GROUND_Y = 650
SLING = pygame.Vector2(180, 530)
PROJECTILE_RADIUS = 18
GRAVITY = 900.0
MAX_LAUNCH_SPEED = 850.0
MAX_PULL_PIXELS = 110.0
MAX_TRAJECTORY_POINTS = 28

# Visual palette: no external art assets are used.
SKY = (126, 211, 255)
GROUND = (79, 171, 87)
INK = (32, 47, 63)
PANEL = (21, 42, 62)
ACCENT = (255, 183, 64)
POWER = (255, 91, 70)
WHITE = (249, 253, 255)


@dataclass
class Target:
    rect: pygame.Rect
    label: str
    alive: bool = True


@dataclass
class Particle:
    position: pygame.Vector2
    velocity: pygame.Vector2
    life: float
    color: tuple[int, int, int]


class SlingshotGame:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Hand Slingshot Game")
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.large_font = pygame.font.Font(None, 42)
        self.title_font = pygame.font.Font(None, 54)
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self) -> None:
        self.projectile = SLING.copy()
        self.velocity = pygame.Vector2()
        self.flying = False
        self.score = 0
        self.shots = 0
        self.particles: list[Particle] = []
        self.targets = [
            Target(pygame.Rect(850, 570, 46, 80), "100"),
            Target(pygame.Rect(970, 530, 54, 120), "150"),
            Target(pygame.Rect(1090, 590, 62, 60), "100"),
        ]

    @staticmethod
    def aim_direction(aim) -> pygame.Vector2:
        """Map normalized palm coordinates to an upward, right-facing shot vector."""
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
        if not self.flying and power > 0.04:
            self.velocity = direction * max(180.0, MAX_LAUNCH_SPEED * power)
            self.flying = True
            self.shots += 1

    def _burst(self, color: tuple[int, int, int]) -> None:
        for _ in range(14):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(75, 250)
            self.particles.append(Particle(
                self.projectile.copy(), pygame.Vector2(math.cos(angle), math.sin(angle)) * speed,
                random.uniform(0.35, 0.70), color,
            ))

    def update(self, dt: float) -> None:
        for particle in self.particles[:]:
            particle.position += particle.velocity * dt
            particle.velocity.y += GRAVITY * 0.35 * dt
            particle.life -= dt
            if particle.life <= 0:
                self.particles.remove(particle)
        if not self.flying:
            return
        self.velocity.y += GRAVITY * dt
        self.projectile += self.velocity * dt
        if self.projectile.y + PROJECTILE_RADIUS >= GROUND_Y:
            self.projectile.y = GROUND_Y - PROJECTILE_RADIUS
            self.velocity.y *= -0.38
            self.velocity.x *= 0.72
            if abs(self.velocity.y) < 70:
                self.velocity = pygame.Vector2()
        projectile_box = pygame.Rect(0, 0, PROJECTILE_RADIUS * 2, PROJECTILE_RADIUS * 2)
        projectile_box.center = self.projectile
        for target in self.targets:
            if target.alive and projectile_box.colliderect(target.rect):
                target.alive = False
                self.score += int(target.label)
                self.velocity *= 0.65
                self._burst(ACCENT)

    def _text(self, text: str, x: int, y: int, color=WHITE, font=None) -> None:
        active_font = font or self.font
        self.screen.blit(active_font.render(text, True, color), (x, y))

    def _rounded_panel(self, rect: pygame.Rect, color=PANEL) -> None:
        pygame.draw.rect(self.screen, color, rect, border_radius=14)
        pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=14)

    def _draw_background(self) -> None:
        self.screen.fill(SKY)
        # Small, drifting clouds make the static playfield easier to read.
        drift = (pygame.time.get_ticks() / 75) % (WIDTH + 220)
        for x, y, scale in ((drift - 220, 120, 1.0), (drift - 650, 205, 0.72), (drift + 360, 85, 0.58)):
            for dx, dy, radius in ((0, 10, 23), (28, 0, 31), (62, 12, 20)):
                pygame.draw.circle(self.screen, (235, 249, 255), (int(x + dx * scale), int(y + dy * scale)), int(radius * scale))
        pygame.draw.rect(self.screen, GROUND, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
        pygame.draw.rect(self.screen, (56, 142, 70), (0, GROUND_Y, WIDTH, 7))

    def _draw_slingshot(self, drawing: bool) -> None:
        # Rear band is drawn first; front band appears over the projectile below.
        if drawing and not self.flying:
            pygame.draw.line(self.screen, (76, 43, 28), (SLING.x - 12, SLING.y - 18), self.projectile, 5)
        pygame.draw.line(self.screen, (100, 57, 31), (SLING.x - 20, GROUND_Y), (SLING.x - 2, SLING.y), 18)
        pygame.draw.line(self.screen, (130, 77, 39), (SLING.x + 20, GROUND_Y), (SLING.x + 2, SLING.y), 18)
        pygame.draw.line(self.screen, (76, 43, 28), (SLING.x + 12, SLING.y - 18), self.projectile, 5)

    def _draw_projectile(self) -> None:
        center = (int(self.projectile.x), int(self.projectile.y))
        pygame.draw.circle(self.screen, (148, 48, 45), center, PROJECTILE_RADIUS + 3)
        pygame.draw.circle(self.screen, POWER, center, PROJECTILE_RADIUS)
        pygame.draw.circle(self.screen, (255, 229, 132), (center[0] + 5, center[1] - 5), 5)
        pygame.draw.circle(self.screen, INK, (center[0] - 5, center[1] - 1), 3)
        pygame.draw.circle(self.screen, INK, (center[0] + 6, center[1] - 1), 3)

    def _draw_targets(self) -> None:
        for target in self.targets:
            if not target.alive:
                continue
            shadow = target.rect.move(5, 6)
            pygame.draw.rect(self.screen, (47, 104, 59), shadow, border_radius=6)
            pygame.draw.rect(self.screen, ACCENT, target.rect, border_radius=6)
            pygame.draw.rect(self.screen, (125, 78, 30), target.rect, 3, border_radius=6)
            self._text(target.label, target.rect.centerx - 13, target.rect.centery - 11, INK, self.small_font)

    def _draw_trajectory(self, direction: pygame.Vector2, pull: float) -> None:
        speed = max(180, MAX_LAUNCH_SPEED * pull)
        position, velocity = self.projectile.copy(), direction * speed
        for point_index in range(MAX_TRAJECTORY_POINTS):
            position += velocity * 0.08
            velocity.y += GRAVITY * 0.08
            radius = 4 if point_index < 10 else 3
            pygame.draw.circle(self.screen, WHITE, position, radius)

    def _draw_hud(self, state_name: str, pull: float, drawing: bool, camera_ok: bool) -> None:
        self._rounded_panel(pygame.Rect(24, 22, 308, 82))
        self._text("PULL POWER", 43, 35, (181, 214, 235), self.small_font)
        pygame.draw.rect(self.screen, (52, 73, 91), (43, 63, 252, 18), border_radius=9)
        pygame.draw.rect(self.screen, POWER, (43, 63, int(252 * pull), 18), border_radius=9)
        self._text(f"{pull * 100:3.0f}%", 303, 56, WHITE, self.font)

        self._rounded_panel(pygame.Rect(1007, 22, 249, 82))
        self._text("SCORE", 1027, 34, (181, 214, 235), self.small_font)
        self._text(str(self.score).zfill(4), 1026, 53, WHITE, self.large_font)
        self._text(f"SHOTS {self.shots}", 1160, 63, ACCENT, self.small_font)

        steps = [
            ("1", "Show open palm", state_name in ("AIMING", "DRAWING", "RELEASED", "COOLDOWN")),
            ("2", "Close fist to grab", drawing),
            ("3", "Move hand back", drawing),
            ("4", "Open fist to launch", drawing),
        ]
        panel = pygame.Rect(24, 128, 245, 180)
        self._rounded_panel(panel, (25, 63, 84))
        self._text("HOW TO PLAY", 43, 145, (181, 214, 235), self.small_font)
        for index, (number, label, active) in enumerate(steps):
            y = 176 + index * 30
            color = ACCENT if active else (177, 201, 214)
            pygame.draw.circle(self.screen, color, (53, y + 8), 10)
            self._text(number, 49, y, INK, self.small_font)
            self._text(label, 72, y, WHITE if active else (202, 220, 230), self.small_font)

        status = "WEBCAM OFF — MOUSE FALLBACK" if not camera_ok else state_name.replace("_", " ")
        status_color = POWER if state_name == "DRAWING" else ACCENT
        self._rounded_panel(pygame.Rect(24, 604, 330, 34), (25, 63, 84))
        self._text(f"GESTURE: {status}", 40, 611, status_color, self.small_font)
        if not self.flying and self.shots > 0:
            self._text("Press R for a fresh round", 960, 665, INK, self.small_font)
        self._text("R restart   Esc quit", 25, 675, INK, self.small_font)

    def draw(self, state_name: str, aim, pull: float, drawing: bool, camera_ok: bool) -> None:
        self._draw_background()
        direction = self.set_aiming_projectile(aim, pull, drawing) if not self.flying else self.aim_direction(aim)
        self._draw_slingshot(drawing)
        if drawing and not self.flying:
            self._draw_trajectory(direction, pull)
        self._draw_targets()
        for particle in self.particles:
            pygame.draw.circle(self.screen, particle.color, particle.position, max(2, int(5 * particle.life)))
        self._draw_projectile()
        self._draw_hud(state_name, pull, drawing, camera_ok)
        pygame.display.flip()

    def close(self) -> None:
        pygame.quit()
