"""Run the hand-controlled slingshot game."""
from __future__ import annotations

import pygame

from game import FPS, SLING, SlingshotGame
from gesture_controller import GestureController, GestureState
from hand_tracker import HandTracker, list_cameras, parse_args

# Set True to test Pygame physics with mouse: hold left button, drag, then release.
MOUSE_DEBUG_MODE = False


def main() -> None:
    args = parse_args("Play the hand-controlled slingshot game.")
    if args.list:
        list_cameras()
        return
    tracker = HandTracker(args.camera)
    camera_ok = tracker.start()
    controller, game = GestureController(), SlingshotGame()
    running, previous_mouse_down, mouse_drawing, previous_mouse_pull = True, False, False, 0.0
    try:
        while running:
            dt = game.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    game.reset()

            hand = frame = None
            if camera_ok:
                camera_ok, hand, frame = tracker.read()
            output = controller.update(hand)
            # Kept as a physics-verification fallback if no webcam is available.
            if MOUSE_DEBUG_MODE or not camera_ok:
                mouse = pygame.Vector2(pygame.mouse.get_pos())
                mouse_down = pygame.mouse.get_pressed()[0]
                aim = (max(0.0, min(1.0, mouse.x / 1280)), max(0.0, min(1.0, mouse.y / 720)))
                mouse_drawing = mouse_down and not game.flying
                # Measured from the fixed sling, not the projectile: the projectile moves
                # toward the mouse as pull grows, which fed back and suppressed the pull.
                pull = min(mouse.distance_to(SLING) / 350.0, 1.0) if mouse_drawing else 0.0
                power = pull
                direction = game.set_aiming_projectile(aim, pull, mouse_drawing)
                if previous_mouse_down and not mouse_down and mouse_drawing is False:
                    game.launch(direction, max(previous_mouse_pull, 0.2))
                previous_mouse_down = mouse_down
                previous_mouse_pull = pull if mouse_down else 0.0
            else:
                aim, pull, power = output.aim, output.pull_distance, output.power
                direction = game.set_aiming_projectile(aim, pull, output.state is GestureState.DRAWING)
                if output.launch:
                    game.launch(direction, output.power)
                    # The debug view reports the actual velocity used by Pygame.
                    output.launch_velocity = game.velocity
                # Skeleton only: the game HUD already reports state, so text would just be noise.
                if frame is not None and hand is not None:
                    tracker.draw_debug(frame, hand, output.state.name, output.launch_velocity, with_text=False)
            game.update(dt)
            game.draw(output.state.name, aim, pull, power, output.state is GestureState.DRAWING, camera_ok, frame)
    finally:
        tracker.close()
        game.close()


if __name__ == "__main__":
    main()
