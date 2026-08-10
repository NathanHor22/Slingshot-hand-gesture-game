"""Run the desktop Pygame hand slingshot game."""
from __future__ import annotations

import pygame

from game import FPS, HEIGHT, SLING, WIDTH, SlingshotGame
from gesture_controller import GestureController, GestureState
from hand_tracker import HandTracker, list_cameras, parse_args


def main() -> None:
    args = parse_args("Play the hand-controlled slingshot game.")
    if args.list:
        list_cameras()
        return
    tracker = HandTracker(args.camera)
    controller = GestureController()
    game = SlingshotGame(args.camera)
    camera_ok = False
    running, previous_mouse_down, previous_mouse_pull = True, False, 0.0
    try:
        while running:
            dt = game.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    running = False
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    if game.mode == "menu":
                        running = False
                    else:
                        game.mode = "menu"
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r and game.mode in ("playing", "lose"):
                    game.retry_level()
                    continue
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r and game.mode == "win":
                    game.start_game()
                    continue
                action = game.handle_event(event)
                if action == "quit":
                    running = False
                if action in ("open_tutorial", "open_settings", "enable_camera", "switch_camera") and (
                        action == "switch_camera" or not camera_ok):
                    tracker.close()
                    tracker = HandTracker(game.camera_index)
                    camera_ok = tracker.start()

            hand = frame = None
            if camera_ok and game.mode in ("tutorial", "settings", "playing"):
                camera_ok, hand, frame = tracker.read()
                if hand is not None and game.mode == "tutorial":
                    tracker.draw_debug(frame, hand, "TUTORIAL", with_text=False)

            # Default values make the menu and tutorial safe when no webcam is open.
            aim, pull, power, drawing, state_name = (.5, .5), 0.0, 0.0, False, "IDLE"
            if game.mode == "tutorial" and camera_ok and not game.mouse_mode:
                output = controller.update(hand)
                aim, pull, power = output.aim, output.pull_distance, output.power
                drawing, state_name = output.state is GestureState.DRAWING, output.state.name
                game.update_tutorial(hand is not None, state_name, pull, output.launch, dt)
            elif game.mode == "playing":
                if game.mouse_mode or not camera_ok:
                    mouse = pygame.Vector2(pygame.mouse.get_pos())
                    mouse_down = pygame.mouse.get_pressed()[0]
                    aim = (max(0.0, min(1.0, mouse.x / WIDTH)), max(0.0, min(1.0, mouse.y / HEIGHT)))
                    drawing = mouse_down and not game.flying
                    pull = min(mouse.distance_to(SLING) / 350.0, 1.0) if drawing else 0.0
                    power = pull
                    direction = game.set_aiming_projectile(aim, pull, drawing)
                    if previous_mouse_down and not mouse_down:
                        game.launch(direction, max(previous_mouse_pull, 0.25))
                    previous_mouse_down = mouse_down
                    previous_mouse_pull = pull if mouse_down else 0.0
                    state_name = "MOUSE MODE"
                else:
                    output = controller.update(hand)
                    aim, pull, power = output.aim, output.pull_distance, output.power
                    drawing, state_name = output.state is GestureState.DRAWING, output.state.name
                    direction = game.set_aiming_projectile(aim, pull, drawing)
                    if output.launch:
                        game.launch(direction, output.power)
            game.update(dt)
            # Camera preview is limited to setup screens; gameplay intentionally has no inset.
            preview = frame if game.mode in ("tutorial", "settings") else None
            game.draw(state_name, aim, pull, power, drawing, preview, camera_ok)
    finally:
        tracker.close()
        game.close()


if __name__ == "__main__":
    main()
