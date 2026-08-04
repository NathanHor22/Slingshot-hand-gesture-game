# Hand Slingshot Game

A Windows Python 3 webcam game: aim with your hand, pull it back away from the camera to charge, then throw forward to launch. No fist or pinch needed.

## Setup

Use Python 3.10 or 3.11 (MediaPipe is most reliable on these Windows versions):

```powershell
cd hand_slingshot_game
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

On the first run, the MediaPipe Hand Landmarker model is downloaded automatically beside `hand_tracker.py`. Internet access is needed only for that first download.

## Choosing a webcam

If you have more than one camera, list them and pick the one you want:

```powershell
python main.py --list          # shows the usable webcam indices
python main.py --camera 1      # play using that camera
```

`--camera` also works on `hand_tracker.py`, which is the quickest way to confirm
which index is which before starting a round. The default is index `0`; change
`CAMERA_INDEX` in `hand_tracker.py` to make a different camera permanent.

## Controls

No fist is required at any point. The draw is driven entirely by how far your hand
is from the camera, because finger closedness is the noisiest signal MediaPipe gives
us and nothing essential should depend on it.

- Show your hand: it is tracked and the palm centre aims the shot.
- Pull your hand back, away from the camera, to charge. The further back you draw,
  the more power. Power is charged from the furthest point you reach, so moving
  forward never spends it.
- Throw forward to fire. Speed does not matter: a slow forward move launches just as
  reliably as a fast one, which only adds a power bonus. Opening a hand that happened
  to be closed also fires, but is never required.
- Move back to neutral without ever charging and the draw quietly cancels, so idle
  hand movement cannot misfire.
- The ball returns to the sling automatically about 0.7 s after it comes to rest.

## Levels

Two levels, played in order. Clearing every target advances after a short pause, and
the score carries over.

| Level | Targets | Notes |
| --- | --- | --- |
| 1 — Warm Up | 3 | Close and generous; nothing needs more than ~0.56 power. |
| 2 — Long Shot | 4 | Further and smaller, with a floating 250-point target that needs a near-full draw (~0.70). |

Press `R` at any time to restart from level 1.
- `R`: restart targets and score. `Esc` or `Q`: quit.

The game runs in a single window: the mirrored webcam feed with the hand skeleton is drawn into the bottom right of the game screen, and the HUD reports the gesture state.

Run `python hand_tracker.py` by itself to inspect all 21 landmarks and tune the named constants at the top of `hand_tracker.py` and `gesture_controller.py`. That standalone view adds the numeric readouts on top of the video: closedness, palm size, relative depth, state, and the most recent launch components.

If a webcam cannot open, the game remains usable with the mouse fallback: set `MOUSE_DEBUG_MODE = True` in `main.py`, hold the left mouse button to draw, then release.
