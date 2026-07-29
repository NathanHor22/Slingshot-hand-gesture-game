"""Debounced gesture state machine for a hand-controlled slingshot."""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from hand_tracker import HandData

# Gesture thresholds / timing.  Hysteresis and stability times prevent one-frame shots.
FIST_CLOSE_THRESHOLD = 0.68       # Enter DRAWING only above this closedness.
FIST_OPEN_THRESHOLD = 0.38        # Release only below this, giving hysteresis.
CLOSE_STABLE_MS = 80              # Fist must be steadily closed before grab.
MIN_DRAW_MS = 150                 # User-requested minimum grab duration before launch.
COOLDOWN_MS = 500                 # Ignore gestures after every launch.
MAX_DEPTH_PULL = 0.38             # Size reduction relative to grab that equals full depth pull.
MAX_FORWARD_SPEED = 1.40          # Palm-size/sec considered a full forward release speed.


class GestureState(Enum):
    IDLE = auto()
    AIMING = auto()
    DRAWING = auto()
    RELEASED = auto()
    COOLDOWN = auto()


@dataclass
class GestureOutput:
    state: GestureState
    aim: np.ndarray
    pull_strength: float = 0.0
    pull_distance: float = 0.0
    power: float = 0.0
    launch: bool = False
    launch_velocity: Optional[np.ndarray] = None


class GestureController:
    def __init__(self) -> None:
        self.state = GestureState.IDLE
        self._close_started: Optional[float] = None
        self._draw_started: Optional[float] = None
        self._grab_size: Optional[float] = None
        self._held_pull_strength = 0.0
        self._last_size: Optional[float] = None
        self._last_time: Optional[float] = None
        self._cooldown_until = 0.0

    @staticmethod
    def _empty(state: GestureState) -> GestureOutput:
        return GestureOutput(state, np.array([0.5, 0.5], dtype=np.float32))

    def update(self, hand: Optional[HandData], now: Optional[float] = None) -> GestureOutput:
        now = time.perf_counter() if now is None else now
        if hand is None:
            if self.state is GestureState.COOLDOWN and now >= self._cooldown_until:
                self.state = GestureState.IDLE
            elif self.state is not GestureState.COOLDOWN:
                self.state, self._close_started, self._draw_started = GestureState.IDLE, None, None
            return self._empty(self.state)

        aim = hand.palm_center.copy()
        if self._last_size is not None and self._last_time is not None:
            size_speed = (hand.palm_size - self._last_size) / max(now - self._last_time, 1e-3)
        else:
            size_speed = 0.0
        self._last_size, self._last_time = hand.palm_size, now

        if self.state is GestureState.COOLDOWN:
            if now >= self._cooldown_until:
                self.state = GestureState.AIMING
            else:
                return GestureOutput(self.state, aim)
        if self.state is GestureState.RELEASED:
            self.state = GestureState.AIMING
        if self.state is GestureState.IDLE:
            self.state = GestureState.AIMING

        if self.state is GestureState.AIMING:
            if hand.closedness >= FIST_CLOSE_THRESHOLD:
                self._close_started = self._close_started or now
                if (now - self._close_started) * 1000 >= CLOSE_STABLE_MS:
                    self.state = GestureState.DRAWING
                    self._draw_started, self._grab_size = now, hand.palm_size
                    self._held_pull_strength = hand.closedness
            else:
                self._close_started = None
            return GestureOutput(self.state, aim, pull_strength=hand.closedness)

        # Depth increases away from camera: a smaller palm than at grab means pull back.
        depth_change = max(0.0, (self._grab_size - hand.palm_size) / max(self._grab_size, 1e-4))
        depth_pull = min(depth_change / MAX_DEPTH_PULL, 1.0)
        pull_strength = float(np.clip(hand.closedness, 0.0, 1.0))
        # Finger closure and backwards depth movement both contribute to the elastic pull.
        pull_distance = float(np.clip(0.60 * pull_strength + 0.40 * depth_pull, 0.0, 1.0))
        # Preserve the closed-fist value: by the release frame fingers are open.
        if hand.closedness > FIST_OPEN_THRESHOLD:
            self._held_pull_strength = pull_strength
        draw_ms = (now - self._draw_started) * 1000
        if hand.closedness <= FIST_OPEN_THRESHOLD and draw_ms >= MIN_DRAW_MS:
            forward_release_speed = float(np.clip(size_speed / MAX_FORWARD_SPEED, 0.0, 1.0))
            pull_strength = self._held_pull_strength
            power = float(np.clip(0.8 * pull_strength + 0.2 * forward_release_speed, 0.0, 1.0))
            self.state, self._cooldown_until = GestureState.RELEASED, now + COOLDOWN_MS / 1000
            return GestureOutput(self.state, aim, pull_strength, pull_distance, power, True,
                                 np.array([forward_release_speed, pull_distance], dtype=np.float32))
        return GestureOutput(self.state, aim, pull_strength, pull_distance)
