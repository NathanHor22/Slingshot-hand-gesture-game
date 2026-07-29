"""Webcam hand tracking built on MediaPipe Tasks Hand Landmarker.

Run this module directly to calibrate landmark and closed-fist thresholds before
running the game: ``python hand_tracker.py``.
"""
from __future__ import annotations

import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Camera and smoothing values.  Adjust these first for a particular webcam.
CAMERA_INDEX = 0                 # Default Windows webcam number.
EMA_ALPHA = 0.30                 # Higher means more responsive, less smooth.
MIN_DETECTION_CONFIDENCE = 0.55  # Reject uncertain initial detections.
MIN_TRACKING_CONFIDENCE = 0.50   # Reject uncertain continuing tracks.
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")

# Fist metric: mean fingertip-to-palm distance, divided by palm scale.
# These are deliberately named because hand shapes/cameras benefit from tuning.
OPEN_TIP_DISTANCE = 1.45         # Typical normalized average for a spread palm.
CLOSED_TIP_DISTANCE = 0.62       # Typical normalized average for a closed fist.
PALM_DEPTH_REFERENCE = 0.18      # Palm size used only for a display depth estimate.

PALM_IDS = (0, 5, 9, 13, 17)     # Wrist and four MCP joints.
TIP_IDS = (4, 8, 12, 16, 20)
FINGER_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


@dataclass
class HandData:
    """Smoothed values used by the gesture controller; coordinates are 0..1."""
    landmarks: np.ndarray
    palm_center: np.ndarray
    palm_size: float
    closedness: float
    depth_estimate: float
    velocity: np.ndarray
    timestamp: float


class HandTracker:
    def __init__(self, camera_index: int = CAMERA_INDEX) -> None:
        self.camera_index = camera_index
        self.cap: Optional[cv2.VideoCapture] = None
        self.landmarker = None
        self._last_palm: Optional[np.ndarray] = None
        self._last_size: Optional[float] = None
        self._last_closedness: Optional[float] = None
        self._last_time: Optional[float] = None
        self._velocity = np.zeros(2, dtype=np.float32)

    @staticmethod
    def _ema(previous, value):
        return value if previous is None else EMA_ALPHA * value + (1.0 - EMA_ALPHA) * previous

    def start(self) -> bool:
        """Open camera and model. Returns False instead of crashing if unavailable."""
        try:
            if not MODEL_PATH.exists():
                print("Downloading MediaPipe Hand Landmarker model (first run only)...")
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                print(f"Could not open webcam {self.camera_index}.")
                return False
            return True
        except Exception as exc:
            print(f"Could not initialise hand tracking: {exc}")
            self.close()
            return False

    def read(self) -> tuple[bool, Optional[HandData], Optional[np.ndarray]]:
        """Return (ok, first-hand data or None, mirrored debug BGR frame)."""
        if self.cap is None or self.landmarker is None:
            return False, None, None
        ok, frame = self.cap.read()
        if not ok:
            return False, None, None
        frame = cv2.flip(frame, 1)  # A mirror makes left/right interaction natural.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int(time.perf_counter() * 1000)
        result = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
        if not result.hand_landmarks:
            self._draw_text(frame, "No hand", (12, 30), (40, 40, 255))
            return True, None, frame

        # The task is configured for one hand; explicitly use index zero as requested.
        points = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
        now = time.perf_counter()
        raw_palm = points[list(PALM_IDS), :2].mean(axis=0)
        # Wrist-to-middle MCP is a stable scale that changes naturally with depth.
        raw_size = float(np.linalg.norm(points[0, :2] - points[9, :2]))
        scale = max(raw_size, 1e-4)
        mean_tip_distance = float(np.mean(np.linalg.norm(points[list(TIP_IDS), :2] - raw_palm, axis=1)) / scale)
        # 0=open and 1=closed, clamped so unusual poses cannot exceed the range.
        raw_closedness = float(np.clip((OPEN_TIP_DISTANCE - mean_tip_distance) /
                                       (OPEN_TIP_DISTANCE - CLOSED_TIP_DISTANCE), 0.0, 1.0))

        palm = self._ema(self._last_palm, raw_palm)
        palm_size = float(self._ema(self._last_size, raw_size))
        closedness = float(self._ema(self._last_closedness, raw_closedness))
        if self._last_palm is not None and self._last_time is not None:
            dt = max(now - self._last_time, 1e-3)
            raw_velocity = (palm - self._last_palm) / dt
            self._velocity = self._ema(self._velocity, raw_velocity)
        self._last_palm, self._last_size = palm, palm_size
        self._last_closedness, self._last_time = closedness, now
        depth = palm_size / PALM_DEPTH_REFERENCE  # Relative only: larger = nearer.
        data = HandData(points, palm, palm_size, closedness, depth, self._velocity.copy(), now)
        self.draw_debug(frame, data)
        return True, data, frame

    @staticmethod
    def _draw_text(frame, text, pos, color=(255, 255, 255)):
        cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 2, cv2.LINE_AA)

    def draw_debug(self, frame: np.ndarray, data: HandData, state: str = "AIMING", launch_velocity=None) -> None:
        h, w = frame.shape[:2]
        pts = (data.landmarks[:, :2] * [w, h]).astype(int)
        for a, b in FINGER_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (70, 220, 70), 2)
        for point in pts:
            cv2.circle(frame, tuple(point), 3, (255, 255, 255), -1)
        center = tuple((data.palm_center * [w, h]).astype(int))
        cv2.circle(frame, center, 7, (0, 210, 255), -1)
        lines = [f"closedness: {data.closedness:.2f}", f"palm size: {data.palm_size:.3f}",
                 f"depth (relative): {data.depth_estimate:.2f}", f"velocity: {data.velocity[0]:.2f}, {data.velocity[1]:.2f}",
                 f"state: {state}"]
        if launch_velocity is not None:
            lines.append(f"launch velocity: {launch_velocity[0]:.0f}, {launch_velocity[1]:.0f}")
        for i, line in enumerate(lines):
            self._draw_text(frame, line, (12, 30 + i * 25))

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None


def main() -> None:
    tracker = HandTracker()
    if not tracker.start():
        return
    print("Tracking live. Press Q or Esc to quit.")
    try:
        while True:
            ok, _, frame = tracker.read()
            if not ok:
                break
            cv2.imshow("Hand tracker debug", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
