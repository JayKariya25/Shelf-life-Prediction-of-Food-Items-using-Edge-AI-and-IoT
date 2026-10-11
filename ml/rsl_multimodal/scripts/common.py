"""Shared configuration and helpers for the multimodal Remaining-Shelf-Life (RSL) pipeline.

Pipeline shape
--------------
    sensor CSV  -> sensor branch  -> 64-d embedding \
                                                     >-- concat -> fusion head -> RSL (days)
    224x224 JPG -> image  branch  -> 128-d embedding /

Each branch is trained on its own first. Both are then frozen and only the fusion
head is trained, so the two modalities remain independently retrainable/swappable
and the fused graph converts cleanly to int8 TFLite for a Raspberry Pi.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "data_images"
SENSOR_FEATURES_CSV = ROOT / "sensor_features.csv"
SENSOR_PREPROCESSED_CSV = ROOT / "sensor_preprocessed.csv"

ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
for _d in (ARTIFACTS, MODELS, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)

MASTER_CSV = ARTIFACTS / "master_dataset.csv"
SENSOR_SCALER_PKL = ARTIFACTS / "sensor_scaler.pkl"
PREP_CONFIG_JSON = ARTIFACTS / "preprocessing_config.json"

# --------------------------------------------------------------------------------------
# Task definition
# --------------------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
MAX_SHELF_LIFE_DAY = 11          # RemainingShelfLife = MAX_SHELF_LIFE_DAY - Day
TARGET = "RemainingShelfLife"
SEED = 42

# Embedding widths of the two frozen branches.
SENSOR_EMBED_DIM = 64
IMAGE_EMBED_DIM = 128

# --------------------------------------------------------------------------------------
# Sensor feature sets
# --------------------------------------------------------------------------------------
# Columns that encode WALL-CLOCK POSITION rather than a physical measurement. Every one of
# them is ~0.97-1.00 correlated with Day, i.e. with the target itself. Feeding any of them
# to the model reproduces `RSL = 11 - Day` arithmetically and reports a fake R^2; none of
# them is knowable on a Raspberry Pi that is handed an arbitrary sample. Never use.
LEAKAGE_COLUMNS = [
    "Sl_No", "Day", "Date", "Timestamp",
    "Year", "Month", "Day_Of_Month", "Day_Of_Week",
    "Elapsed_Time_Hours", "Elapsed_Time_Days",
    "Cumulative_Temperature", "Cumulative_Humidity", "Cumulative_Gas",
]

# Physically-measurable, deployable features (the default sensor branch input).
SENSOR_FEATURES_FULL = [
    # instantaneous
    "Temperature", "Humidity", "Gas",
    # de-noised
    "Temperature_Smoothed", "Humidity_Smoothed", "Gas_Smoothed",
    # short-term dynamics
    "Temperature_Change", "Humidity_Change", "Gas_Change",
    "Temperature_Rate", "Humidity_Rate", "Gas_Rate",
    # local statistics
    "Temperature_Rolling_Mean", "Humidity_Rolling_Mean", "Gas_Rolling_Mean",
    "Temperature_Rolling_STD", "Humidity_Rolling_STD", "Gas_Rolling_STD",
    # cross terms
    "Temperature_Humidity_Interaction", "Temperature_Gas_Interaction",
    "Humidity_Gas_Interaction",
    # drift from the start of the monitoring session (a Pi knows its own first reading)
    "Temperature_From_Initial", "Humidity_From_Initial", "Gas_From_Initial",
    # direction
    "Gas_Increasing", "Temperature_Increasing", "Humidity_Increasing",
]

# Deliberately NOT model inputs. The *_Level bands in sensor_features.csv are not a
# function of the reading they describe - Gas_Level "Low" is exactly 179 while
# "Very_High" spans 152-195, and each of the four Gas bands holds exactly 3316/4 rows, so
# they were cut on rank rather than on value. Consequences:
#   * they cannot be recomputed on a Pi from a live reading, so a deployed model that
#     depends on them cannot be fed correctly;
#   * Temperature_Level is the single value "Moderate" for all 3316 rows (no information);
#   * Gas_Level correlates -0.31 with Day, i.e. it partly encodes elapsed time.
# Including them lowers sensor-only test MAE from 0.720 to 0.620 days, but that gain is
# not reproducible at inference time, so the deployable model does without them.
UNREPRODUCIBLE_AT_INFERENCE = [
    "Temperature_Level_Ord", "Humidity_Level_Ord", "Gas_Level_Ord",
    # *_Trend is excluded for the same practical reason: its definition could not be
    # recovered from the data. Every rolling least-squares slope over the raw, cleaned or
    # smoothed series (windows 3-30, all min_periods) fails to reproduce the column, so a
    # Pi cannot compute it from live readings. See scripts/verify_pi_features.py, which
    # asserts that every feature the model does use IS reproducible.
    "Temperature_Trend", "Humidity_Trend", "Gas_Trend",
]

# Exact definitions recovered from sensor_features.csv and verified to ~1e-14 by
# scripts/verify_pi_features.py. predict_pi.py must implement these, not approximations:
#   X_Smoothed      = X_Clean.rolling(5, min_periods=1).mean()
#   X_Rolling_Mean  = X_Smoothed.rolling(5, min_periods=1).mean()      (double-smoothed)
#   X_Rolling_STD   = X_Smoothed.rolling(5, min_periods=1).std(ddof=1)
#   X_Change        = X_Smoothed.diff()
#   X_Rate          = X_Change / Time_Difference_Hours
#   X_Increasing    = X_Change > 0
#   X_From_Initial  = X_Smoothed - X_Smoothed[0]     (anchor = start of the session)
#   A_B_Interaction = A_Smoothed * B_Smoothed
SMOOTHING_WINDOW = 5

# The 3 raw channels only - matches the earlier DEMO_MODEL and is the strictest
# "what a bare Pi reads right now" baseline.
SENSOR_FEATURES_MINIMAL = ["Temperature", "Humidity", "Gas"]

LEVEL_ORDER = {"Low": 0, "Medium": 1, "Moderate": 2, "High": 3, "Very_High": 4}
LEVEL_COLUMNS = ["Temperature_Level", "Humidity_Level", "Gas_Level"]

# --------------------------------------------------------------------------------------
# Split strategies
# --------------------------------------------------------------------------------------
# "blocked"  - group temporally adjacent frames into blocks and split whole blocks.
#              Frames captured ~2 minutes apart are near-duplicates; a plain random split
#              puts a frame in train and its twin in test, which inflates every metric.
#              This is the split the headline numbers are reported on.
# "random"   - plain stratified random split, reproduces the earlier DEMO_MODEL protocol
#              and is kept only so the two can be compared honestly.
SPLIT_STRATEGIES = ("blocked", "random")
# A 30-minute block was not enough: with ~2-minute sampling, 93% of test rows still sat
# within 30 minutes of a train row, and 1-nearest-neighbour on *untrained* ImageNet
# features scored MAE 0.07 days on that split - i.e. the score was duplicate lookup, not
# learning. Blocks are now 3 hours AND every val/test row within EMBARGO_MINUTES of any
# train row is dropped from evaluation entirely. scripts/ablations.py re-runs the
# nearest-neighbour probe on each split so the leakage claim stays measured, not asserted.
BLOCK_MINUTES = 180
EMBARGO_MINUTES = 60
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15


def add_level_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Map the categorical *_Level bands onto an ordinal scale."""
    df = df.copy()
    for col in LEVEL_COLUMNS:
        if col in df.columns:
            df[f"{col}_Ord"] = df[col].map(LEVEL_ORDER).astype("float32")
    return df


def feature_columns(feature_set: str = "full") -> list[str]:
    if feature_set == "full":
        return list(SENSOR_FEATURES_FULL)
    if feature_set == "minimal":
        return list(SENSOR_FEATURES_MINIMAL)
    raise ValueError(f"unknown feature_set: {feature_set!r}")


def load_master() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"{MASTER_CSV} not found - run scripts/prepare_data.py first."
        )
    df = pd.read_csv(MASTER_CSV)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def split_masks(df: pd.DataFrame, strategy: str = "blocked"):
    col = f"split_{strategy}"
    if col not in df.columns:
        raise KeyError(f"{col} missing from master dataset")
    return (df[col] == "train").to_numpy(), (df[col] == "val").to_numpy(), (df[col] == "test").to_numpy()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype="float64").ravel()
    y_pred = np.asarray(y_pred, dtype="float64").ravel()
    err = y_pred - y_true
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "MedAE": float(np.median(np.abs(err))),
        "WithinHalfDay": float((np.abs(err) <= 0.5).mean()),
        "WithinOneDay": float((np.abs(err) <= 1.0).mean()),
        "n": int(y_true.size),
    }


def _json_safe(obj):
    """NaN/Infinity are not valid JSON - emit null so strict parsers can read the file."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False))


def set_seeds(seed: int = SEED) -> None:
    import random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass
