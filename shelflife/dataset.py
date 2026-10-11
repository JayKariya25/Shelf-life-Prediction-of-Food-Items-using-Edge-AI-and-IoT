"""Read-only access to the model's held-out test split, for the admin console.

The split lives in ``ml/rsl_multimodal/artifacts/master_dataset.csv``. Each row
carries the 27 *precomputed* sensor features as they were at that moment of the
monitoring run - they depend on the preceding readings, so they cannot be rebuilt
from a single row - plus the image file and the true remaining shelf life.

Nothing here writes to the dataset. Standard library ``csv`` keeps the console usable
without pandas.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

SPLIT_COLUMN = "split_blocked"
TARGET = "RemainingShelfLife"
REQUIRED_COLUMNS = {"image_file", "Timestamp", "Day", TARGET, SPLIT_COLUMN,
                    "Temperature", "Humidity", "Gas"}
METADATA_COLUMNS = {"image_file", "Timestamp", "Day", TARGET, "split_blocked", "split_random"}


class HeldOutSplit:
    """Rows of the master dataset whose blocked split is ``test``."""

    def __init__(self, bundle_dir: str | Path) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.master_path = self.bundle_dir / "artifacts" / "master_dataset.csv"
        self.image_dir = self.bundle_dir / "data_images"
        self.error: str | None = None
        self.rows: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self.feature_columns: list[str] = []
        self._load()

    @property
    def available(self) -> bool:
        return self.error is None and bool(self.rows)

    def _load(self) -> None:
        if not self.master_path.is_file():
            self.error = f"No held-out split metadata at {self.master_path}"
            return
        try:
            with self.master_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                header = list(reader.fieldnames or [])
                missing = REQUIRED_COLUMNS - set(header)
                if missing:
                    self.error = f"master_dataset.csv is missing columns: {sorted(missing)}"
                    return
                self.feature_columns = [c for c in header if c not in METADATA_COLUMNS]
                for raw in reader:
                    split = (raw.get(SPLIT_COLUMN) or "").strip().lower()
                    self.counts[split] = self.counts.get(split, 0) + 1
                    if split != "test":
                        continue
                    parsed = self._parse(raw)
                    if parsed is not None:
                        self.rows.append(parsed)
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            self.error = f"Could not read master_dataset.csv: {exc}"
            return
        if not self.rows:
            self.error = f"No rows with {SPLIT_COLUMN} == 'test' were found."

    def _parse(self, raw: dict[str, str]) -> dict[str, Any] | None:
        try:
            features = {c: float(raw[c]) for c in self.feature_columns}
            actual = float(raw[TARGET])
            day = int(float(raw["Day"]))
        except (TypeError, ValueError, KeyError):
            return None  # skip an unparseable row rather than break the console
        if math.isnan(actual) or any(math.isnan(v) for v in features.values()):
            return None
        name = Path((raw.get("image_file") or "").strip()).name
        image = self.image_dir / name if name else None
        return {
            "image_file": name,
            "image_path": str(image) if image is not None and image.is_file() else None,
            "timestamp": raw.get("Timestamp"),
            "day": day,
            "actual": actual,
            "temperature": features.get("Temperature"),
            "humidity": features.get("Humidity"),
            "gas": features.get("Gas"),
            "features": features,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, index: int) -> dict[str, Any] | None:
        if 0 <= index < len(self.rows):
            return self.rows[index]
        return None

    def vector(self, row: dict[str, Any], feature_names: list[str]) -> list[list[float]]:
        """The row's features in the model's order. Raises on a missing column."""
        missing = [f for f in feature_names if f not in row["features"]]
        if missing:
            raise KeyError(f"master_dataset.csv lacks model features: {missing}")
        return [[row["features"][f] for f in feature_names]]

    def summary(self) -> dict[str, Any]:
        with_images = sum(1 for row in self.rows if row["image_path"])
        return {
            "available": self.available,
            "error": self.error,
            "master_path": str(self.master_path),
            "image_dir": str(self.image_dir),
            "split_column": SPLIT_COLUMN,
            "test_rows": len(self.rows),
            "rows_with_images": with_images,
            "rows_missing_images": len(self.rows) - with_images,
            "split_counts": self.counts,
            "feature_columns": len(self.feature_columns),
        }


@lru_cache(maxsize=4)
def load_split(bundle_dir: str) -> HeldOutSplit:
    return HeldOutSplit(bundle_dir)


def clear_cache() -> None:
    load_split.cache_clear()


# --- regression metrics ------------------------------------------------------
def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    """MAE, RMSE, R^2 and friends over paired values.

    R^2 is undefined when every actual value is identical (zero variance); it is
    reported as ``None`` rather than as a misleading 0 or 1.
    """
    n = min(len(actual), len(predicted))
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "bias": None,
                "max_error": None, "within_half_day": None, "within_one_day": None}

    errors = [predicted[i] - actual[i] for i in range(n)]
    absolute = [abs(e) for e in errors]
    mae = sum(absolute) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)
    bias = sum(errors) / n

    mean_actual = sum(actual[:n]) / n
    ss_total = sum((actual[i] - mean_actual) ** 2 for i in range(n))
    ss_residual = sum(e * e for e in errors)
    r2 = None if ss_total == 0 else 1.0 - (ss_residual / ss_total)

    return {
        "n": n,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4) if r2 is not None else None,
        "bias": round(bias, 4),
        "max_error": round(max(absolute), 4),
        "within_half_day": round(sum(a <= 0.5 for a in absolute) / n, 4),
        "within_one_day": round(sum(a <= 1.0 for a in absolute) / n, 4),
    }
