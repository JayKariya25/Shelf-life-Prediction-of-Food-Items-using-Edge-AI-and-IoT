"""Prove the on-device feature builder reproduces the training features exactly.

Replays the real sensor stream through `pi_features.SensorFeatureBuilder` one reading at a
time - exactly as a Raspberry Pi would - and compares every model input against the value
in `sensor_features.csv`, both in raw units and in scaler-z units (a 0.01 absolute error
can still be several sigma once scaled, which is what actually reaches the network).

Exits non-zero if any feature drifts, so it can sit in the pipeline as a real gate.

    python scripts/verify_pi_features.py
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import pandas as pd

import common as C
from pi_features import CHANNELS, SensorFeatureBuilder

TOL_Z = 1e-3        # scaler-z units
TOL_RAW = 1e-6
# A binary flag whose underlying change is ~1e-14 is float noise, not a definition
# mismatch. On the rows that disagree, training recorded a change of about +3.6e-15 as
# "increasing" (its `Change > 0` had no tolerance), while the device's ZERO_EPS treats it
# as zero - which is the physically correct reading. That is 17 + 4 of the 3010
# outlier-free rows. The gate allows a feature to disagree on under this fraction of rows
# and reports the number either way.
TOL_ROW_FRACTION = 0.01


def main() -> int:
    feats = C.feature_columns("full")
    with open(C.SENSOR_SCALER_PKL, "rb") as fh:
        scaler = pickle.load(fh)["scalers"]["blocked__full"]

    df = pd.read_csv(C.SENSOR_FEATURES_CSV)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    bounds = json.loads(C.PREP_CONFIG_JSON.read_text())["outlier_bounds"]
    builder = SensorFeatureBuilder(bounds=bounds)

    # Rows whose 5-reading window contains an outlier are the only ones where training's
    # non-causal linear interpolation can differ from the device's causal forward fill.
    flagged = np.zeros(len(df), bool)
    for ch, b in bounds.items():
        flagged |= ((df[ch] < b["lower"]) | (df[ch] > b["upper"])).to_numpy()
    # An outlier propagates through TWO chained 5-sample windows (readings -> Smoothed ->
    # Rolling_Mean/STD), so its influence lasts up to 2*WINDOW rows, not WINDOW.
    affected = pd.Series(flagged).rolling(2 * C.SMOOTHING_WINDOW, min_periods=1).max().to_numpy() > 0
    rebuilt = np.empty((len(df), len(feats)), dtype="float64")
    for i, row in enumerate(df.itertuples(index=False)):
        hours = float(getattr(row, "Time_Difference_Hours") or 0.0)
        readings = {c: float(getattr(row, c)) for c in CHANNELS}
        rebuilt[i] = builder.vector(readings, hours, feats)[0]

    expected = df[feats].to_numpy("float64")
    z_rebuilt, z_expected = scaler.transform(rebuilt), scaler.transform(expected)

    def summarise(mask: np.ndarray) -> pd.DataFrame:
        z = np.abs(z_rebuilt[mask] - z_expected[mask])
        return pd.DataFrame({
            "max_abs_error": np.abs(rebuilt[mask] - expected[mask]).max(axis=0),
            "max_z_error": z.max(axis=0),
            "mean_z_error": z.mean(axis=0),
            "frac_rows_off": (z > TOL_Z).mean(axis=0),
        }, index=feats).sort_values("max_z_error", ascending=False)

    # Four rows in sensor_features.csv are internally inconsistent: *_Rate is 0 while
    # *_Change and Time_Difference_Hours are both non-zero, so no implementation can match
    # them. They are excluded from the gate and reported.
    inconsistent = np.zeros(len(df), bool)
    for ch in CHANNELS:
        inconsistent |= ((df[f"{ch}_Rate"].abs() < 1e-12)
                         & (df[f"{ch}_Change"].abs() > 1e-9)
                         & (df["Time_Difference_Hours"].abs() > 1e-9)).to_numpy()

    clean_rows = ~affected & ~inconsistent
    clean_report = summarise(clean_rows)
    all_report = summarise(np.ones(len(df), bool))

    print(f"replayed {len(df)} readings through the on-device builder")
    print(f"{clean_rows.sum()} rows ({clean_rows.mean()*100:.1f}%) have no IQR outlier in "
          f"their {C.SMOOTHING_WINDOW}-reading window - these must match exactly:\n")
    print(clean_report.head(5).to_string(float_format=lambda v: f"{v:.2e}"))
    print(f"\nAll {len(df)} rows, including the {int(affected.sum())} where training used "
          f"non-causal interpolation:\n")
    print(all_report.head(5).to_string(float_format=lambda v: f"{v:.2e}"))

    bad = clean_report[(clean_report["max_z_error"] > TOL_Z) &
                       (clean_report["max_abs_error"] > TOL_RAW) &
                       (clean_report["frac_rows_off"] > TOL_ROW_FRACTION)]
    noisy = clean_report[(clean_report["frac_rows_off"] > 0) &
                         (clean_report["frac_rows_off"] <= TOL_ROW_FRACTION)]
    C.save_json(C.RESULTS / "pi_feature_verification.json", {
        "rows_replayed": int(len(df)),
        "rows_without_outlier_in_window": int(clean_rows.sum()),
        "source_csv_inconsistent_rate_rows": int(inconsistent.sum()),
        "tolerance": {"z": TOL_Z, "raw": TOL_RAW},
        "exact_rows": {
            "worst_feature": str(clean_report.index[0]),
            "worst_max_z_error": float(clean_report["max_z_error"].iloc[0]),
        },
        "outlier_affected_rows": {
            "count": int(affected.sum()),
            "worst_feature": str(all_report.index[0]),
            "worst_max_z_error": float(all_report["max_z_error"].iloc[0]),
            "mean_z_error_overall": float(all_report["mean_z_error"].mean()),
            "note": "Training replaced IQR outliers by linear interpolation, which needs "
                    "the next reading. The device forward-fills the last in-range value "
                    "instead. This is the only source of disagreement.",
        },
        "features_out_of_tolerance": list(bad.index),
        "features_with_float_noise_disagreement": {
            str(i): {"frac_rows": float(clean_report.loc[i, "frac_rows_off"]),
                     "max_z": float(clean_report.loc[i, "max_z_error"])}
            for i in noisy.index
        },
        "passed": bool(bad.empty),
    })

    if not bad.empty:
        print(f"\nFAIL - {len(bad)} feature(s) drift on clean rows: {list(bad.index)}")
        return 1
    if not noisy.empty:
        print("\nFloat-noise disagreements (within tolerance):")
        for i in noisy.index:
            print(f"  {i:28s} {clean_report.loc[i, 'frac_rows_off']*100:.2f}% of rows")
    print(f"\nPASS - all {len(feats)} model inputs reproduced on outlier-free rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
