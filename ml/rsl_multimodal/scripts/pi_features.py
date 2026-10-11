"""Rebuild the engineered sensor vector on-device, matching training.

Every definition here was recovered from `sensor_features.csv` and is checked against it
by `scripts/verify_pi_features.py`. On outlier-free rows the continuous features agree to
about 1e-6 in raw units (under 1e-5 in scaler-z units - the CSV's own rounding), and the
two `*_Increasing` flags disagree on 0.6% / 0.1% of rows: training computed `Change > 0`
with no tolerance, so a float-noise change of +3.6e-15 was recorded as "increasing". The
device treats that as zero, which is the physically correct reading. Getting a definition
wrong is silent - the model still returns a number, fed from a different distribution -
which is why the verification script exists.

Cleaning. Training first replaced IQR outliers with NaN and LINEARLY INTERPOLATED them -
a whole-dataset, non-causal operation that a live device cannot perform, because it needs
the *next* reading. The deployable equivalent, used here, freezes the IQR bounds from
training and substitutes the last in-range reading (forward fill only). When there is no
earlier in-range reading - an out-of-range FIRST reading of a session - the value is
clipped to the nearest bound instead. (The first version passed it through unchanged, so
a 45 C cold start reached the network while the API reported it as clamped.) The two
approaches agree on every sample whose window contains no outlier, and
`verify_pi_features.py` reports the residual on the rest rather than hiding it.

Definitions (X is one of Temperature / Humidity / Gas, all on the SMOOTHED series):

    X_Clean         = the reading, or the last in-range reading if out of IQR bounds
    X_Smoothed      = mean of the last 5 CLEAN readings
    X_Rolling_Mean  = mean of the last 5 SMOOTHED values          (double-smoothed)
    X_Rolling_STD   = sample std (ddof=1) of the last 5 SMOOTHED values, 0 for one sample
    X_Change        = X_Smoothed - previous X_Smoothed
    X_Rate          = X_Change / hours since the previous reading
    X_Increasing    = 1 if X_Change > 0 else 0
    X_From_Initial  = X_Smoothed - the first X_Smoothed of the session
    A_B_Interaction  = A_Smoothed * B_Smoothed
"""

from __future__ import annotations

from collections import deque

import numpy as np

CHANNELS = ("Temperature", "Humidity", "Gas")
WINDOW = 5
# A running mean accumulates float noise that pandas' vectorised diff does not, so a
# genuinely unchanged reading can come out as +3e-15. Without this epsilon the
# X_Increasing flag flips on 64 rows where training recorded "not increasing".
ZERO_EPS = 1e-9


class SensorFeatureBuilder:
    """Streaming feature builder. One instance per monitoring session."""

    def __init__(self, window: int = WINDOW, bounds: dict | None = None):
        self.window = window
        # {"Temperature": {"lower": .., "upper": ..}, ...} - frozen at training time.
        self.bounds = bounds or {}
        self.last_clean: dict[str, float] = {}
        self.raw = {c: deque(maxlen=window) for c in CHANNELS}
        self.smoothed = {c: deque(maxlen=window) for c in CHANNELS}
        self.prev_smoothed: dict[str, float] = {}
        self.initial_smoothed: dict[str, float] = {}
        # What cleaning did to the most recent reading, per channel:
        # {"Temperature": {"raw": 45.0, "used": 29.33, "how": "clipped"}}
        # Lets callers report accurately instead of guessing.
        self.last_adjustments: dict[str, dict] = {}

    # -- (de)serialisation so a Pi can survive a restart mid-session -------------------
    def state_dict(self) -> dict:
        return {
            "window": self.window,
            "bounds": self.bounds,
            "last_clean": self.last_clean,
            "raw": {c: list(self.raw[c]) for c in CHANNELS},
            "smoothed": {c: list(self.smoothed[c]) for c in CHANNELS},
            "prev_smoothed": self.prev_smoothed,
            "initial_smoothed": self.initial_smoothed,
        }

    @classmethod
    def from_state(cls, state: dict | None, bounds: dict | None = None) -> "SensorFeatureBuilder":
        state = state or {}
        b = cls(int(state.get("window", WINDOW)), bounds or state.get("bounds"))
        if not state:
            return b
        b.last_clean = dict(state.get("last_clean", {}))
        for c in CHANNELS:
            b.raw[c].extend(state.get("raw", {}).get(c, []))
            b.smoothed[c].extend(state.get("smoothed", {}).get(c, []))
        b.prev_smoothed = dict(state.get("prev_smoothed", {}))
        b.initial_smoothed = dict(state.get("initial_smoothed", {}))
        return b

    def clean(self, channel: str, x: float) -> float:
        """Causal stand-in for training's interpolate-over-outliers step."""
        b = self.bounds.get(channel)
        if b is not None and not (b["lower"] <= x <= b["upper"]):
            raw = x
            if channel in self.last_clean:
                x, how = self.last_clean[channel], "forward_filled"
            else:
                x, how = min(max(x, b["lower"]), b["upper"]), "clipped"
            self.last_adjustments[channel] = {"raw": raw, "used": x, "how": how}
        self.last_clean[channel] = x
        return x

    # -- the feature vector ------------------------------------------------------------
    def update(self, readings: dict[str, float], hours_since_previous: float) -> dict[str, float]:
        v: dict[str, float] = {}
        smoothed_now: dict[str, float] = {}
        self.last_adjustments = {}
        missing = [c for c in CHANNELS if readings.get(c) is None]
        if missing:
            raise KeyError(f"reading is missing channel(s): {missing}")

        for c in CHANNELS:
            x = float(readings[c])
            cleaned = self.clean(c, x)
            self.raw[c].append(cleaned)
            sm = float(np.mean(self.raw[c]))
            self.smoothed[c].append(sm)
            smoothed_now[c] = sm
            self.initial_smoothed.setdefault(c, sm)

            win = np.asarray(self.smoothed[c], dtype="float64")
            prev = self.prev_smoothed.get(c)
            change = 0.0 if prev is None else sm - prev

            v[c] = x
            v[f"{c}_Smoothed"] = sm
            v[f"{c}_Rolling_Mean"] = float(win.mean())
            v[f"{c}_Rolling_STD"] = float(win.std(ddof=1)) if win.size > 1 else 0.0
            v[f"{c}_Change"] = change
            v[f"{c}_Rate"] = change / hours_since_previous if hours_since_previous else 0.0
            v[f"{c}_Increasing"] = float(change > ZERO_EPS)
            v[f"{c}_From_Initial"] = sm - self.initial_smoothed[c]

        v["Temperature_Humidity_Interaction"] = smoothed_now["Temperature"] * smoothed_now["Humidity"]
        v["Temperature_Gas_Interaction"] = smoothed_now["Temperature"] * smoothed_now["Gas"]
        v["Humidity_Gas_Interaction"] = smoothed_now["Humidity"] * smoothed_now["Gas"]

        self.prev_smoothed = dict(smoothed_now)
        return v

    def vector(self, readings: dict[str, float], hours_since_previous: float,
               feature_names: list[str]) -> np.ndarray:
        v = self.update(readings, hours_since_previous)
        missing = [c for c in feature_names if c not in v]
        if missing:
            raise KeyError(f"cannot reconstruct these features on-device: {missing}")
        return np.array([[v[c] for c in feature_names]], dtype="float32")
