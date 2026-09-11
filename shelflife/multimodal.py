"""Adapter for the trained multimodal Keras model.

Mirrors the inference contract of the project's Streamlit demo exactly:

* saved with ``tf.keras.models.load_model(path, compile=False)``
* two named inputs: ``image`` (None, 224, 224, 3) and ``sensor`` (None, 3)
* sensor order is ``["Temperature", "Humidity", "Gas"]``, scaled by the fitted
  ``sensor_scaler.pkl``
* images are RGB, resized to 224x224 with BILINEAR, then passed through
  ``mobilenet_v3.preprocess_input``
* the single output is remaining shelf life in **days**

The heavy imports (TensorFlow, joblib, PIL) happen inside :meth:`load` so the
web application still starts, and still serves the kinetic baseline, on a
machine where they are not installed.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SENSOR_COLUMNS = ("Temperature", "Humidity", "Gas")
EXPECTED_IMAGE_SHAPE = (None, 224, 224, 3)
EXPECTED_SENSOR_SHAPE = (None, 3)
EXPECTED_INPUTS = {"image", "sensor"}


@dataclass(slots=True)
class ModelOutput:
    remaining_days: float
    inference_ms: float


class ModelUnavailable(RuntimeError):
    """The trained model cannot serve this request."""


class MultimodalShelfLifeModel:
    """Loads and runs the trained Keras model. Never mutates the artifacts."""

    name = "Multimodal MobileNetV3 + sensor MLP"
    kind = "trained-model"
    uses_image = True
    requires_gas = True

    def __init__(
        self,
        model_path: str | Path,
        scaler_path: str | Path,
        config_path: str | Path = "",
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.config_path = Path(config_path) if config_path else None

        self.available = False
        self.load_error: str | None = None
        self.version = "unknown"
        self.config: dict[str, Any] = {}
        self.metrics: dict[str, float] = {}
        self.image_size = (224, 224)

        self._model = None
        self._scaler = None
        self._tf = None
        self._np = None
        self._pil = None
        # Keras predict() is not guaranteed thread-safe; the dev server and most
        # WSGI servers handle requests concurrently, so calls are serialised.
        self._lock = threading.Lock()

        self.load()

    # --- loading ---------------------------------------------------------
    def load(self) -> None:
        missing = [
            str(path)
            for path in (self.model_path, self.scaler_path)
            if not path.exists()
        ]
        if self.config_path is not None and not self.config_path.exists():
            missing.append(str(self.config_path))
        if missing:
            self.load_error = "Missing artifact(s): " + ", ".join(missing)
            return

        try:
            import numpy as np
            import tensorflow as tf
            from PIL import Image
        except ImportError as exc:
            self.load_error = (
                f"{exc}. The trained model needs tensorflow, numpy, pillow and joblib. "
                "Note that TensorFlow publishes no wheel for Python 3.14 - run the app "
                "on Python 3.12."
            )
            return

        try:
            import joblib
        except ImportError as exc:
            self.load_error = f"joblib is required to load the fitted scaler: {exc}"
            return

        try:
            if self.config_path is not None:
                self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
                order = self.config.get("sensor_columns")
                if order is not None and tuple(order) != SENSOR_COLUMNS:
                    raise ValueError(
                        f"config.json sensor_columns is {order}; expected {list(SENSOR_COLUMNS)}. "
                        "Refusing to run with a mismatched feature order."
                    )
                size = self.config.get("image_size")
                if size:
                    self.image_size = (int(size[0]), int(size[1]))
                self.version = str(self.config.get("version", self.config.get("model_version", "unknown")))
                self.name = str(self.config.get("name", self.name))
                self.metrics = {
                    key: float(value)
                    for key, value in (self.config.get("metrics") or {}).items()
                    if isinstance(value, (int, float))
                }
                # Tolerate metrics stored at the top level of the config too.
                for key in ("test_mae", "test_rmse", "test_r2", "mae", "rmse", "r2"):
                    if isinstance(self.config.get(key), (int, float)):
                        self.metrics.setdefault(key, float(self.config[key]))

            model = tf.keras.models.load_model(self.model_path, compile=False)
            self._validate_signature(model)

            self._model = model
            self._scaler = joblib.load(self.scaler_path)
            self._validate_scaler()
            self._tf, self._np, self._pil = tf, np, Image
            self.available = True
            self.load_error = None
            log.info("Trained multimodal model loaded from %s", self.model_path)
        except Exception as exc:
            self.load_error = f"{type(exc).__name__}: {exc}"
            self._model = None
            self._scaler = None
            self.available = False
            log.warning("Trained model failed to load: %s", self.load_error)

    def _validate_signature(self, model) -> None:
        """Fail loudly on a model that does not match the expected contract."""
        names = [tensor.name.split(":")[0] for tensor in model.inputs]
        if set(names) != EXPECTED_INPUTS:
            raise ValueError(
                f"Saved model inputs are {names}; expected exactly {sorted(EXPECTED_INPUTS)}."
            )
        shapes = {name: tuple(tensor.shape) for name, tensor in zip(names, model.inputs)}
        if shapes["image"] != EXPECTED_IMAGE_SHAPE or shapes["sensor"] != EXPECTED_SENSOR_SHAPE:
            raise ValueError(f"Unexpected saved model input shapes: {shapes}")
        outputs = model.outputs
        if len(outputs) != 1 or tuple(outputs[0].shape)[-1] != 1:
            raise ValueError(
                f"Expected one scalar regression output, got shapes "
                f"{[tuple(o.shape) for o in outputs]}."
            )
        self.input_names = names
        self.input_shapes = shapes

    def _validate_scaler(self) -> None:
        expected = getattr(self._scaler, "n_features_in_", None)
        if expected is not None and int(expected) != len(SENSOR_COLUMNS):
            raise ValueError(
                f"Fitted scaler expects {expected} features; the model takes "
                f"{len(SENSOR_COLUMNS)} ({', '.join(SENSOR_COLUMNS)})."
            )

    # --- inference --------------------------------------------------------
    def open_image(self, source) -> Any:
        """Open any path or file-like object as RGB, raising a clear error."""
        if not self.available:
            raise ModelUnavailable(self.load_error or "Model is not loaded")
        Image = self._pil
        try:
            with Image.open(source) as handle:
                return handle.convert("RGB").copy()
        except Exception as exc:
            raise ValueError(f"The image could not be opened: {exc}") from exc

    def preprocess_image(self, image) -> Any:
        """RGB -> 224x224 bilinear -> MobileNetV3 preprocessing -> batch of 1."""
        np, tf, Image = self._np, self._tf, self._pil
        resized = image.resize(self.image_size, Image.Resampling.BILINEAR)
        pixels = np.asarray(resized, dtype=np.float32)
        return tf.keras.applications.mobilenet_v3.preprocess_input(pixels)[None, ...]

    def preprocess_sensor(self, temperature: float, humidity: float, gas: float) -> Any:
        np = self._np
        raw = np.array([[temperature, humidity, gas]], dtype=np.float32)
        return self._scaler.transform(raw).astype(np.float32)

    def predict(self, image, temperature: float, humidity: float, gas: float) -> ModelOutput:
        """Run one forward pass. Returns remaining shelf life in days."""
        if not self.available:
            raise ModelUnavailable(self.load_error or "Model is not loaded")
        for label, value in (("temperature", temperature), ("humidity", humidity), ("gas", gas)):
            if value is None:
                raise ModelUnavailable(
                    f"The trained model needs a {label} reading and none was supplied."
                )

        started = time.perf_counter()
        payload = {
            "image": self.preprocess_image(image),
            "sensor": self.preprocess_sensor(float(temperature), float(humidity), float(gas)),
        }
        with self._lock:
            output = self._model.predict(payload, verbose=0)
        elapsed = (time.perf_counter() - started) * 1000.0
        return ModelOutput(remaining_days=float(output[0, 0]), inference_ms=elapsed)

    # --- reporting --------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "available": self.available,
            "load_error": self.load_error,
            "model_path": str(self.model_path),
            "scaler_path": str(self.scaler_path),
            "config_path": str(self.config_path) if self.config_path else None,
            "model_exists": self.model_path.exists(),
            "scaler_exists": self.scaler_path.exists(),
            "config_exists": bool(self.config_path and self.config_path.exists()),
            "image_size": list(self.image_size),
            "sensor_columns": list(SENSOR_COLUMNS),
            "metrics": self.metrics,
            "input_names": getattr(self, "input_names", []),
            "input_shapes": {k: list(v) for k, v in getattr(self, "input_shapes", {}).items()},
        }
