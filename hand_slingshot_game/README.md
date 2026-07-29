# Hand Slingshot Game

A Windows Python 3 webcam game: aim with an open palm, close a fist to grab the bird, move your hand away from the camera to pull, then open your fist to launch.

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

## Controls

- Open palm: track and aim.
- Close fist: grab after a short stable hold.
- While holding: move the hand farther from the camera (smaller palm) to increase pull.
- Open fist after at least 150 ms: launch. A 500 ms cooldown prevents repeat shots.
- `R`: restart targets and score. `Esc` or `Q` in the camera window: quit.

Run `python hand_tracker.py` by itself to inspect all 21 landmarks and tune the named constants at the top of `hand_tracker.py` and `gesture_controller.py`. The OpenCV debug window shows mirrored video, skeleton, palm center, closedness, palm size, relative depth, state, and the most recent launch components.

If a webcam cannot open, the game remains usable with the mouse fallback: set `MOUSE_DEBUG_MODE = True` in `main.py`, hold the left mouse button to draw, then release.
