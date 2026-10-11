"""Loading helpers shared by the three training scripts."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import common as C


def load_all(split: str = "blocked", feature_set: str = "full"):
    """Return (df, X_sensor_scaled, backbone_features, y, masks)."""
    df = C.load_master()

    feats = C.feature_columns(feature_set)
    with open(C.SENSOR_SCALER_PKL, "rb") as fh:
        bundle = pickle.load(fh)
    scaler = bundle["scalers"][f"{split}__{feature_set}"]
    X = scaler.transform(df[feats].to_numpy("float64")).astype("float32")

    backbone_path = C.ARTIFACTS / "backbone_features.npy"
    if not backbone_path.exists():
        raise FileNotFoundError(
            f"{backbone_path} not found - run scripts/extract_image_features.py first."
        )
    F = np.load(backbone_path)
    if len(F) != len(df):
        raise ValueError(f"backbone cache ({len(F)}) is out of sync with master ({len(df)})")

    y = df[C.TARGET].to_numpy("float32")
    masks = C.split_masks(df, split)
    return df, X, F, y, masks


def slice_masks(arrays: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    return [a[mask] for a in arrays]


def save_predictions(df: pd.DataFrame, mask: np.ndarray, y_true, y_pred, path) -> None:
    out = df.loc[mask, ["image_file", "Timestamp", "Day"]].copy()
    out["actual_RSL"] = np.asarray(y_true).ravel()
    out["predicted_RSL"] = np.asarray(y_pred).ravel()
    out["abs_error"] = (out["predicted_RSL"] - out["actual_RSL"]).abs()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
