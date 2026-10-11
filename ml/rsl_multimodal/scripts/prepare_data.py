"""Build the aligned image+sensor master dataset, the RSL target and reusable splits.

Outputs
-------
artifacts/master_dataset.csv        one row per usable sample, with both split columns
artifacts/sensor_scaler.pkl         StandardScaler fitted on TRAIN rows only
artifacts/preprocessing_config.json exact configuration used, for the Pi/inference side
artifacts/data_audit.json           what was dropped and why
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import common as C


def outlier_bounds(df: pd.DataFrame) -> dict:
    """Recover the IQR bounds that produced the *_Clean columns in preprocessing.

    Verified: replacing readings outside [Q1-1.5*IQR, Q3+1.5*IQR] with NaN and linearly
    interpolating reproduces every *_Clean column to ~7e-15. These bounds are frozen into
    the preprocessing config so the Raspberry Pi can apply the same rule causally.
    """
    out = {}
    for ch in ("Temperature", "Humidity", "Gas"):
        q1, q3 = df[ch].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        flagged = (df[ch] < lower) | (df[ch] > upper)
        rebuilt = df[ch].mask(flagged).interpolate(method="linear").ffill().bfill()
        out[ch] = {
            "lower": lower,
            "upper": upper,
            "n_outliers": int(flagged.sum()),
            "reproduces_Clean_max_abs_error": float((rebuilt - df[f"{ch}_Clean"]).abs().max()),
        }
    return out


def build_master() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(C.SENSOR_FEATURES_CSV)
    audit = {"rows_in_sensor_features": int(len(df))}
    audit["outlier_bounds"] = outlier_bounds(df)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["image_file"] = df["Image_Path"].astype(str).map(lambda p: Path(str(p).replace("\\", "/")).name)

    # Target -------------------------------------------------------------------------
    df[C.TARGET] = (C.MAX_SHELF_LIFE_DAY - df["Day"]).astype("float32")

    # Keep only rows whose 224x224 image actually exists on disk ------------------------
    available = {p.name for p in C.IMAGE_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    has_image = df["image_file"].isin(available)
    audit["rows_without_image"] = int((~has_image).sum())
    audit["images_on_disk"] = int(len(available))
    df = df.loc[has_image].reset_index(drop=True)

    # Ordinal encodings, then assert the model inputs are clean -------------------------
    df = C.add_level_ordinals(df)
    feats = C.feature_columns("full")
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise KeyError(f"missing engineered feature columns: {missing}")

    n_before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats + [C.TARGET]).reset_index(drop=True)
    audit["rows_dropped_nan_or_inf"] = int(n_before - len(df))

    # Data-driven leakage check: rather than trusting the hand-written exclusion list,
    # measure every model input against the target and refuse anything that is effectively
    # a clock. |corr| >= 0.90 with Day means the column reconstructs RSL = 11 - Day.
    corr = df[feats].apply(lambda c: c.corr(df["Day"])).abs()
    leaked = sorted(corr[corr >= 0.90].index)
    if leaked:
        raise AssertionError(
            f"model inputs are near-perfectly correlated with the target: "
            f"{ {c: round(float(corr[c]), 4) for c in leaked} }"
        )
    audit["max_abs_corr_with_Day_among_inputs"] = {
        "column": str(corr.idxmax()), "value": float(corr.max())
    }

    audit["rows_final"] = int(len(df))
    audit["day_counts"] = {int(k): int(v) for k, v in df["Day"].value_counts().sort_index().items()}
    audit["target_range"] = [float(df[C.TARGET].min()), float(df[C.TARGET].max())]
    return df, audit


def assign_blocked_split(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Split whole time-blocks, never individual frames.

    Frames are captured roughly every 2 minutes, so neighbouring frames are near
    duplicates. Blocks of BLOCK_MINUTES are assigned as units, stratified within each Day
    so that every RSL value is still represented in train/val/test.
    """
    t0 = df["Timestamp"].min()
    # Compound (Day, block) key: a wall-clock block can straddle midnight, and assignment
    # runs per Day, so a bare block id could land in two different splits.
    block = ((df["Timestamp"] - t0).dt.total_seconds() // (C.BLOCK_MINUTES * 60)).astype(int)
    df = df.assign(_block=list(zip(df["Day"], block)))

    out = pd.Series("train", index=df.index, dtype=object)
    for _day, day_df in df.groupby("Day"):
        blocks = list(day_df["_block"].unique())
        idx = day_df.index
        if len(blocks) < 3:
            # A day too short to block-split (Day 10 has 1 row, Day 11 has 8). Splitting it
            # at row level is mildly optimistic, but forcing it entirely into train would
            # remove RSL 0 and 1 - the spoiled end of the range - from evaluation entirely,
            # which is the worse failure for a shelf-life model.
            rows = idx.to_numpy().copy()
            rng.shuffle(rows)
            n_train = max(1, int(round(C.TRAIN_FRAC * len(rows))))
            n_val = max(0, int(round(C.VAL_FRAC * len(rows))))
            out.loc[rows[n_train:n_train + n_val]] = "val"
            out.loc[rows[n_train + n_val:]] = "test"
            continue
        n = len(blocks)
        blocks = [blocks[i] for i in rng.permutation(n)]
        n_train = min(max(1, int(round(C.TRAIN_FRAC * n))), n - 2)
        n_val = min(max(1, int(round(C.VAL_FRAC * n))), n - n_train - 1)
        val_b = set(blocks[n_train:n_train + n_val])
        test_b = set(blocks[n_train + n_val:])
        out.loc[idx[day_df["_block"].isin(val_b)]] = "val"
        out.loc[idx[day_df["_block"].isin(test_b)]] = "test"
    return out


def apply_embargo(df: pd.DataFrame, split: pd.Series) -> tuple[pd.Series, dict]:
    """Drop val/test rows that sit within EMBARGO_MINUTES of any train row.

    Block boundaries alone are not enough: a test frame one block-edge away from a train
    frame is still a near-duplicate of it. Embargoed rows are marked "dropped" and take no
    part in training or scoring.
    """
    ts = df["Timestamp"].to_numpy("datetime64[s]").astype("int64")
    train_ts = np.sort(ts[(split == "train").to_numpy()])
    embargo_s = C.EMBARGO_MINUTES * 60

    out = split.copy()
    dropped = {"val": 0, "test": 0}
    for part in ("val", "test"):
        mask = (split == part).to_numpy()
        if not mask.any():
            continue
        pos = np.searchsorted(train_ts, ts[mask])
        left = np.where(pos > 0, ts[mask] - train_ts[np.clip(pos - 1, 0, None)], np.inf)
        right = np.where(pos < len(train_ts), train_ts[np.clip(pos, None, len(train_ts) - 1)] - ts[mask], np.inf)
        too_close = np.minimum(left, right) < embargo_s
        idx = df.index[mask][too_close]
        out.loc[idx] = "dropped"
        dropped[part] = int(too_close.sum())
    return out, dropped


def assign_random_split(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Plain per-Day stratified random split (the earlier DEMO_MODEL protocol)."""
    out = pd.Series("train", index=df.index, dtype=object)
    for _, day_df in df.groupby("Day"):
        idx = day_df.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(C.TRAIN_FRAC * n))
        n_val = int(round(C.VAL_FRAC * n))
        out.loc[idx[n_train:n_train + n_val]] = "val"
        out.loc[idx[n_train + n_val:]] = "test"
    return out


def main() -> None:
    C.set_seeds()
    df, audit = build_master()

    df["split_blocked"] = assign_blocked_split(df, np.random.default_rng(C.SEED))
    df["split_blocked"], embargoed = apply_embargo(df, df["split_blocked"])
    df["split_random"] = assign_random_split(df, np.random.default_rng(C.SEED))
    audit["embargoed_rows"] = embargoed
    audit["embargo_minutes"] = C.EMBARGO_MINUTES
    audit["block_minutes"] = C.BLOCK_MINUTES
    audit["split_counts"] = {
        s: df[f"split_{s}"].value_counts().to_dict() for s in C.SPLIT_STRATEGIES
    }

    # Scalers fitted on TRAIN ONLY, one per split strategy x feature set ----------------
    feats_full = C.feature_columns("full")
    feats_min = C.feature_columns("minimal")
    scalers: dict[str, StandardScaler] = {}
    for strategy in C.SPLIT_STRATEGIES:
        train_mask = df[f"split_{strategy}"] == "train"
        for name, feats in (("full", feats_full), ("minimal", feats_min)):
            sc = StandardScaler().fit(df.loc[train_mask, feats].to_numpy("float64"))
            scalers[f"{strategy}__{name}"] = sc

    keep = [
        "image_file", "Timestamp", "Day", C.TARGET,
        "split_blocked", "split_random", *feats_full,
    ]
    df[keep].to_csv(C.MASTER_CSV, index=False)

    with open(C.SENSOR_SCALER_PKL, "wb") as fh:
        pickle.dump({"scalers": scalers, "features": {"full": feats_full, "minimal": feats_min}}, fh)

    # The range of raw readings the model was actually fitted on. Serving code uses it to
    # tell interpolation from extrapolation: the instantaneous X feature is the RAW
    # reading (as in training), so an out-of-envelope value reaches the network even
    # though its smoothed features are cleaned.
    train_rows = df["split_blocked"] == "train"
    input_envelope = {
        c: {"min": float(df.loc[train_rows, c].min()), "max": float(df.loc[train_rows, c].max()),
            "source": "raw readings, blocked train split"}
        for c in ("Temperature", "Humidity", "Gas")
    }

    C.save_json(C.PREP_CONFIG_JSON, {
        "image_size": list(C.IMAGE_SIZE),
        "image_preprocessing": "none - raw 0-255 pixels; MobileNetV3 rescales inside the graph "
                               "(mobilenet_v3.preprocess_input is the identity)",
        "image_source": "224x224 RGB JPEG q95, LANCZOS (matches Image_Preprocessing config)",
        "image_decoding": "tf.io.decode_jpeg defaults (fast integer IDCT, fancy upsampling); "
                          "reproduced by scripts/image_io.py",
        "input_envelope": input_envelope,
        "target": {"column": C.TARGET, "definition": f"{C.MAX_SHELF_LIFE_DAY} - Day", "unit": "days"},
        "sensor_features": {"full": feats_full, "minimal": feats_min},
        "excluded_leakage_columns": C.LEAKAGE_COLUMNS,
        "excluded_unreproducible_columns": C.UNREPRODUCIBLE_AT_INFERENCE,
        "outlier_bounds": audit["outlier_bounds"],
        "smoothing_window": C.SMOOTHING_WINDOW,
        "split": {
            "strategies": list(C.SPLIT_STRATEGIES),
            "block_minutes": C.BLOCK_MINUTES,
            "embargo_minutes": C.EMBARGO_MINUTES,
            "fractions": {"train": C.TRAIN_FRAC, "val": C.VAL_FRAC,
                          "test": round(1 - C.TRAIN_FRAC - C.VAL_FRAC, 4)},
            "seed": C.SEED,
        },
        "embedding_dims": {"sensor": C.SENSOR_EMBED_DIM, "image": C.IMAGE_EMBED_DIM},
    })
    C.save_json(C.ARTIFACTS / "data_audit.json", audit)

    print(json.dumps(audit, indent=2))
    print(f"\nwrote {C.MASTER_CSV}")


if __name__ == "__main__":
    main()
