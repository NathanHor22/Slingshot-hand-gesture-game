"""Debounced gesture state machine for a hand-controlled slingshot."""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from hand_tracker import HandData

# Timing.  Hysteresis and stability windows prevent one-frame shots.
MIN_DRAW_MS = 120                 # Minimum charge time before a throw can fire.
COOLDOWN_MS = 450                 # Ignore gestures after every launch.

# The draw is driven purely by hand depth, so no fist is needed to hold or throw the
# ball.  Finger closedness is the noisiest signal MediaPipe gives us, so nothing
# required depends on it any more.
BASELINE_ALPHA = 0.03             # How fast the neutral hand distance re-centres.
DRAW_ENTER = 0.07                 # Shrink below neutral that starts charging.
DRAW_CANCEL = 0.02                # Drifting back inside this cancels without firing.
MAX_DEPTH_PULL = 0.28             # Shrink below neutral that equals a full-power draw.
MAX_FORWARD_SPEED = 1.60          # Palm-size/sec considered a full-speed throw.
THRUST_TRIGGER = 0.34             # Fraction of MAX_FORWARD_SPEED that fires immediately.
PULL_WEIGHT = 0.70                # Draw distance dominates power; the throw adds the rest.
MIN_LAUNCH_POWER = 0.18           # An uncharged flick still lobs the ball a little.

# A slow forward move fires too: releasing this much of the draw counts as a throw, so
# the shot never depends on being fast enough, only on having charged something first.
MIN_CHARGED_PULL = 0.18           # Draw that must exist before a forward move can fire.
RELEASE_FRACTION = 0.45           # Fraction of the peak draw given back that launches.

# Optional convenience trigger: opening a hand that happened to be closed also fires.
# It is never required, and never contributes to power.
FIST_HELD_THRESHOLD = 0.70
FIST_OPEN_THRESHOLD = 0.48


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
        self._draw_started: Optional[float] = None
        self._grab_size: Optional[float] = None
        self._baseline: Optional[float] = None
        self._was_closed = False
        self._peak_pull = 0.0
        self._last_size: Optional[float] = None
        self._last_time: Optional[float] = None
        self._cooldown_until = 0.0

    @staticmethod
    def _empty(state: GestureState) -> GestureOutput:
        return GestureOutput(state, np.array([0.5, 0.5], dtype=np.float32))

    def _power(self, thrust: float) -> float:
        """Power comes from how far the ball was drawn, plus how hard it was thrown.

        Fist closedness is deliberately absent: it gates whether the ball is held,
        and letting it set power made every shot at least half strength.
        """
        charged = PULL_WEIGHT * self._peak_pull + (1.0 - PULL_WEIGHT) * thrust
        return float(np.clip(MIN_LAUNCH_POWER + (1.0 - MIN_LAUNCH_POWER) * charged, 0.0, 1.0))

    def update(self, hand: Optional[HandData], now: Optional[float] = None) -> GestureOutput:
        now = time.perf_counter() if now is None else now
        if hand is None:
            if self.state is GestureState.COOLDOWN and now >= self._cooldown_until:
                self.state = GestureState.IDLE
            elif self.state is not GestureState.COOLDOWN:
                # Losing the hand drops the draw and forgets where neutral was.
                self.state, self._draw_started, self._baseline = GestureState.IDLE, None, None
                self._peak_pull, self._was_closed = 0.0, False
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
            # Neutral hand distance re-centres slowly, so any seating position works
            # and the draw never depends on an absolute palm size.
            self._baseline = hand.palm_size if self._baseline is None else \
                self._baseline + (hand.palm_size - self._baseline) * BASELINE_ALPHA
            shrink = (self._baseline - hand.palm_size) / max(self._baseline, 1e-4)
            if shrink >= DRAW_ENTER:
                self.state = GestureState.DRAWING
                self._draw_started, self._grab_size = now, self._baseline
                self._peak_pull, self._was_closed = 0.0, False
            return GestureOutput(self.state, aim, pull_strength=hand.closedness)

        # Depth increases away from camera: a smaller palm than neutral means pulled back.
        # The entry threshold is subtracted so charging starts from zero, not a step.
        depth_change = max(0.0, (self._grab_size - hand.palm_size) / max(self._grab_size, 1e-4))
        depth_pull = float(np.clip((depth_change - DRAW_ENTER) / max(MAX_DEPTH_PULL - DRAW_ENTER, 1e-4), 0.0, 1.0))
        # Throwing forward shrinks the live draw, so power is charged from the peak reached.
        self._peak_pull = max(self._peak_pull, depth_pull)
        thrust = float(np.clip(size_speed / MAX_FORWARD_SPEED, 0.0, 1.0))
        power = self._power(thrust)
        draw_ms = (now - self._draw_started) * 1000
        # Opening a hand that was closed still fires, but an open hand throws just as well.
        if hand.closedness >= FIST_HELD_THRESHOLD:
            self._was_closed = True
        opened_fist = self._was_closed and hand.closedness <= FIST_OPEN_THRESHOLD
        # Giving the draw back counts as a throw at any speed; a fast one just fires sooner.
        released = (self._peak_pull >= MIN_CHARGED_PULL
                    and (self._peak_pull - depth_pull) >= RELEASE_FRACTION * self._peak_pull)
        if (thrust >= THRUST_TRIGGER or released or opened_fist) and draw_ms >= MIN_DRAW_MS:
            self.state, self._cooldown_until = GestureState.RELEASED, now + COOLDOWN_MS / 1000
            return GestureOutput(self.state, aim, self._peak_pull, depth_pull, power, True,
                                 np.array([thrust, self._peak_pull], dtype=np.float32))
        # Only bail out when nothing meaningful was ever charged, so noise cannot misfire
        # and a real draw is never silently thrown away.
        if depth_change <= DRAW_CANCEL and self._peak_pull < MIN_CHARGED_PULL:
            self.state, self._peak_pull, self._was_closed = GestureState.AIMING, 0.0, False
            return GestureOutput(self.state, aim, pull_strength=hand.closedness)
        return GestureOutput(self.state, aim, self._peak_pull, depth_pull, power)
