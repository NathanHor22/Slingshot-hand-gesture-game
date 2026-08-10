# Hand Slingshot Game

A Windows Python 3 webcam game with a Pygame menu, guided tutorial, settings,
timed rounds, automatic level flow, and hand-controlled slingshot physics.

## Setup

Use Python 3.10 or 3.11:

    cd hand_slingshot_game
    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python main.py

On the first run, the MediaPipe hand model is downloaded beside
hand_tracker.py.

## Event flow

1. Select Start on the opening screen. The game automatically opens the
   tutorial and attempts to enable the selected camera.
2. Perform the four tutorial movements. Each correct movement receives a green
   checkmark.
3. Completing the tutorial starts the game automatically after three seconds.
4. Clear every colored block before the timer reaches zero.
5. Each completed level displays a five-second result screen before the next
   level loads. Finishing level three returns to the main menu.

The camera appears in Tutorial and Settings but remains hidden during gameplay.
Every menu, retry, and settings action has an on-screen button, so the event
does not require a keyboard.

## Controls

- Show your hand to track and aim.
- Move your hand away from the camera to draw and charge the shot.
- Move your hand toward the camera to launch.
- Use mouse mode from the tutorial if a camera is unavailable.

## Levels

- Level 1 - Training Yard: close-range character blocks.
- Level 2 - Long Range: farther targets and a two-hit pink block.
- Level 3 - Tower Siege: a collapsing tower with a two-hit purple shooter.

Colored blocks use gravity and fall when their support is destroyed. Brown,
eye-free foundations are permanent and cannot be destroyed. The purple tower
only fires while the player's bird is airborne, and enemy shots clear when the
sling reloads.

Only targets requiring two or more hits display health bars. A block disappears
as soon as its health reaches zero, immediately removing its collision and
structural support so blocks above it fall into the changed structure. Every
target or support impact consumes the active bird and reliably reloads the
sling. Launch strength is 20 percent higher than the original campaign values.
Block impacts also play one randomized explosion at a time. The supplied
explosion shapes vary in size, rotation, position, and style, then grow and
fade during the reload.

Menu options play a short sound once when the pointer moves onto them and a
confirmation sound when selected. Gameplay uses a quiet, looping battle theme;
block collisions play their impact sound together with the explosion. If an
audio device is unavailable, the game continues silently instead of crashing.

Open Settings to adjust the timer, toggle the tracer, or change camera source.
The live camera preview confirms the selected source immediately.

Run python hand_tracker.py to inspect all 21 landmarks and tune the named
tracking constants. Supplied artwork is stored in assets/.
