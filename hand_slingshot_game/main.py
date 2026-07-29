"""Run the hand-controlled slingshot game."""
from __future__ import annotations

import cv2
import pygame

from game import FPS, SlingshotGame
from gesture_controller import GestureController, GestureState
from hand_tracker import HandTracker

# Set True to test Pygame physics with mouse: hold left button, drag, then release.
MOUSE_DEBUG_MODE = False


def main() -> None:
    tracker = HandTracker()
    camera_ok = tracker.start()
    controller, game = GestureController(), SlingshotGame()
    running, previous_mouse_down, mouse_drawing, previous_mouse_pull = True, False, False, 0.0
    try:
        while running:
            dt = game.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
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
                pull = min(mouse.distance_to(game.projectile) / 350.0, 1.0) if mouse_drawing else 0.0
                direction = game.set_aiming_projectile(aim, pull, mouse_drawing)
                if previous_mouse_down and not mouse_down and mouse_drawing is False:
                    game.launch(direction, max(previous_mouse_pull, 0.2))
                previous_mouse_down = mouse_down
                previous_mouse_pull = pull if mouse_down else 0.0
            else:
                aim, pull = output.aim, output.pull_distance
                direction = game.set_aiming_projectile(aim, pull, output.state is GestureState.DRAWING)
                if output.launch:
                    game.launch(direction, output.power)
                    # The debug view reports the actual velocity used by Pygame.
                    output.launch_velocity = game.velocity
                if frame is not None:
                    tracker.draw_debug(frame, hand, output.state.name, output.launch_velocity) if hand else None
                    cv2.imshow("Hand tracker debug", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        running = False
            game.update(dt)
            game.draw(output.state.name, aim, pull, output.state is GestureState.DRAWING, camera_ok)
    finally:
        tracker.close()
        cv2.destroyAllWindows()
        game.close()


if __name__ == "__main__":
    main()
