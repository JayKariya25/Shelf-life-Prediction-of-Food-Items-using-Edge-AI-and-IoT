"""Read-only access to the held-out split metadata, for the admin console.

This module never writes to the dataset, the splits file, the scaler or the
model. It exists so the developer interface can reproduce the Streamlit demo's
held-out evaluation inside the web application.

Standard library only (``csv``, not pandas) so the admin console still works on
an install that has the model runtime but not the full data-science stack.
"""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

SENSOR_COLUMNS = ("Temperature", "Humidity", "Gas")
REQUIRED_COLUMNS = {
    "split",
    "Imagepath",
    "resolved_image_path",
    "RemainingShelfLife",
    *SENSOR_COLUMNS,
}


class DatasetError(RuntimeError):
    """The splits metadata is missing or unusable."""


class HeldOutSplit:
    """The rows of ``dataset_splits.csv`` whose split is ``test``."""

    def __init__(self, splits_path: str | Path, image_dir: str | Path) -> None:
        self.splits_path = Path(splits_path)
        self.image_dir = Path(image_dir)
        self.error: str | None = None
        self.rows: list[dict[str, Any]] = []
        self.counts: dict[str, int] = {}
        self._load()

    @property
    def available(self) -> bool:
        return self.error is None and bool(self.rows)

    def _load(self) -> None:
        if not self.splits_path.is_file():
            self.error = f"No split metadata at {self.splits_path}"
            return
        try:
            with self.splits_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                header = set(reader.fieldnames or [])
                missing = REQUIRED_COLUMNS - header
                if missing:
                    self.error = f"Split metadata is missing columns: {sorted(missing)}"
                    return
                for raw in reader:
                    split = (raw.get("split") or "").strip().lower()
                    self.counts[split] = self.counts.get(split, 0) + 1
                    if split != "test":
                        continue
                    parsed = self._parse(raw)
                    if parsed is not None:
                        self.rows.append(parsed)
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            self.error = f"Could not read the split metadata: {exc}"
            return

        if not self.rows:
            self.error = "No rows with split == 'test' were found."

    def _parse(self, raw: dict[str, str]) -> dict[str, Any] | None:
        try:
            values = {name: float(raw[name]) for name in SENSOR_COLUMNS}
            actual = float(raw["RemainingShelfLife"])
        except (TypeError, ValueError):
            return None  # skip an unparseable row rather than crash the console
        if any(math.isnan(v) for v in values.values()) or math.isnan(actual):
            return None
        return {
            "image_label": raw["Imagepath"],
            "image_path": self.resolve_image(raw),
            "temperature": values["Temperature"],
            "humidity": values["Humidity"],
            "gas": values["Gas"],
            "actual": actual,
        }

    def resolve_image(self, raw: dict[str, str]) -> str | None:
        """Prefer the recorded absolute path, tolerating a moved project folder."""
        recorded = (raw.get("resolved_image_path") or "").strip()
        if recorded:
            candidate = Path(recorded)
            if candidate.is_file():
                return str(candidate)
        name = Path((raw.get("Imagepath") or "").strip()).name
        if name:
            fallback = self.image_dir / name
            if fallback.is_file():
                return str(fallback)
        return None

    def __len__(self) -> int:
        return len(self.rows)

    def get(self, index: int) -> dict[str, Any] | None:
        if 0 <= index < len(self.rows):
            return self.rows[index]
        return None

    def summary(self) -> dict[str, Any]:
        with_images = sum(1 for row in self.rows if row["image_path"])
        return {
            "available": self.available,
            "error": self.error,
            "splits_path": str(self.splits_path),
            "image_dir": str(self.image_dir),
            "test_rows": len(self.rows),
            "rows_with_images": with_images,
            "rows_missing_images": len(self.rows) - with_images,
            "split_counts": self.counts,
        }


@lru_cache(maxsize=4)
def load_split(splits_path: str, image_dir: str) -> HeldOutSplit:
    """Cached loader keyed on the configured paths."""
    return HeldOutSplit(splits_path, image_dir)


def clear_cache() -> None:
    load_split.cache_clear()


# --- regression metrics ------------------------------------------------------
def regression_metrics(actual: list[float], predicted: list[float]) -> dict[str, Any]:
    """MAE, RMSE and R^2 over paired values.

    R^2 is undefined when every actual value is identical (zero variance); it is
    reported as ``None`` rather than as a misleading 0 or 1.
    """
    n = min(len(actual), len(predicted))
    if n == 0:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "bias": None}

    errors = [predicted[i] - actual[i] for i in range(n)]
    mae = sum(abs(e) for e in errors) / n
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
        "max_error": round(max(abs(e) for e in errors), 4),
    }
