"""The trained multimodal Remaining-Shelf-Life (RSL) model, served in-process.

The model and the code that must agree with it live together in ``ml/rsl_multimodal``:

    models/image_ensemble_blocked_full_float16.tflite   image (1,224,224,3) -> days
    artifacts/preprocessing_config.json         outlier bounds, input envelope, target
    scripts/image_io.py                         training-identical image decoding
    results/...                                 the measured error figures

    models/fusion_blocked_full_float16.tflite   image + sensor (1,27) -> days
    artifacts/sensor_scaler.pkl                 {"scalers": {...}, "features": {...}}
    scripts/pi_features.py                      the 27 engineered sensor features

Two export shapes are supported and the loaded graph decides which: the image-only
ensemble that is served by default, and the older image+sensor fusion graph. The sensor
half of the bundle is only read when the graph actually has a sensor input. Everything
below branches on ``image_only``, which is set from the TFLite input contract itself
rather than from configuration, so a mismatch cannot go unnoticed.

``pi_features.py`` and ``image_io.py`` are loaded *from the bundle* rather than copied
into this package, so the feature definitions and image transform are always the ones
that ship with - and are verified against - the model being served.

Nothing here writes to the bundle.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import pickle
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger(__name__)

SPLIT = "blocked"
FEATURE_SET = "full"
CHANNELS = ("Temperature", "Humidity", "Gas")


class ModelUnavailable(RuntimeError):
    """The trained model cannot serve this request."""


@dataclass(slots=True)
class Reading:
    """One sensor sample in a monitoring session."""

    temperature: float
    humidity: float
    gas: float
    hours_since_previous: float = 0.0


@dataclass(slots=True)
class RSLResult:
    raw_days: float                    # unclamped network output
    days: float                        # clamped to [0, max_days]
    reading_number: int                # 1 = cold start
    inference_ms: float
    decoder: str
    warnings: list[str] = field(default_factory=list)
    adjustments: dict[str, dict] = field(default_factory=dict)
    out_of_envelope: dict[str, float] = field(default_factory=dict)


def _load_module(name: str, path: Path):
    """Import a single file under a private name, without touching sys.path.

    The module is registered in ``sys.modules`` before it runs: ``@dataclass`` looks its
    defining module up there and fails on an unregistered one.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _interpreter_class():
    try:
        from tflite_runtime.interpreter import Interpreter  # Raspberry Pi
        return Interpreter, "tflite-runtime"
    except ImportError:
        pass
    try:
        from ai_edge_litert.interpreter import Interpreter  # LiteRT
        return Interpreter, "ai-edge-litert"
    except ImportError:
        pass
    import tensorflow as tf  # development machine
    return tf.lite.Interpreter, f"tensorflow {tf.__version__}"


class RSLModel:
    """Loads the bundle once and serves thread-safe predictions."""

    kind = "trained-model"
    uses_image = True

    def __init__(self, bundle_dir: str | Path,
                 model_file: str = "models/image_ensemble_blocked_full_float16.tflite"):
        self.bundle_dir = Path(bundle_dir)
        self.model_path = self.bundle_dir / model_file
        self.scaler_path = self.bundle_dir / "artifacts" / "sensor_scaler.pkl"
        self.config_path = self.bundle_dir / "artifacts" / "preprocessing_config.json"
        self.features_path = self.bundle_dir / "scripts" / "pi_features.py"
        self.image_io_path = self.bundle_dir / "scripts" / "image_io.py"

        self.name = "Trained RSL model"
        self.version = self.model_path.stem
        self.available = False
        # Decided by the graph in _bind_contract, not by configuration.
        self.image_only = False
        self.load_error: str | None = None
        self.runtime = None

        self.feature_names: list[str] = []
        self.bounds: dict[str, dict] = {}
        self.envelope: dict[str, dict] = {}
        self.max_days = 11.0
        self.image_size = (224, 224)
        self.metrics: dict[str, Any] = {}
        self.verification: dict[str, Any] = {}

        self._np = None
        self._features = None
        self._image_io = None
        self._scaler = None
        self._interpreter = None
        self._image_input = None
        self._sensor_input = None
        self._output = None
        self._lock = threading.Lock()

        self.load()

    # ------------------------------------------------------------------ load --
    @property
    def core_files(self) -> dict[str, Path]:
        """Needed by every export, whatever its inputs are."""
        return {
            "model": self.model_path,
            "config": self.config_path,
            "image loader": self.image_io_path,
        }

    @property
    def sensor_files(self) -> dict[str, Path]:
        """Needed only by an export that takes the 27-feature sensor vector."""
        return {
            "scaler": self.scaler_path,
            "feature builder": self.features_path,
        }

    @property
    def required_files(self) -> dict[str, Path]:
        files = dict(self.core_files)
        if not self.image_only:
            files.update(self.sensor_files)
        return files

    def load(self) -> None:
        missing = [f"{label} ({path})" for label, path in self.core_files.items()
                   if not path.is_file()]
        if missing:
            self.load_error = "Missing from the model bundle: " + "; ".join(missing)
            return
        try:
            import numpy as np
        except ImportError as exc:
            self.load_error = f"numpy is required: {exc}"
            return
        try:
            interpreter_cls, runtime = _interpreter_class()
        except ImportError:
            self.load_error = (
                "No TFLite runtime found. Install tflite-runtime (Raspberry Pi) or "
                "tensorflow. TensorFlow has no wheel for Python 3.14 - use Python 3.12."
            )
            return

        try:
            self._np = np
            self._image_io = _load_module("_rsl_image_io", self.image_io_path)
            self._load_config()
            interpreter = self._build_interpreter(interpreter_cls)
            # One input is the image-only ensemble; two is the image+sensor fusion graph.
            self.image_only = len(interpreter.get_input_details()) == 1
            if not self.image_only:
                self._load_sensor_side()
            self._bind_contract(interpreter)
            self._load_metrics()
            self.name = ("Image-ensemble RSL model" if self.image_only
                         else "Multimodal RSL fusion model")
            self.runtime = runtime
            self.available = True
            self.load_error = None
            log.info("RSL model loaded from %s (%s, %s)", self.model_path, runtime,
                     "image only" if self.image_only else "image + sensor")
        except Exception as exc:  # anything wrong with the bundle degrades, never crashes
            self.available = False
            self._interpreter = None
            self.load_error = f"{type(exc).__name__}: {exc}"
            log.warning("RSL model failed to load: %s", self.load_error)

    def _load_config(self) -> None:
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.bounds = config["outlier_bounds"]
        self.envelope = config.get("input_envelope", {})
        # "11 - Day" -> 11
        self.max_days = float(str(config["target"]["definition"]).split("-")[0].strip())
        size = config.get("image_size") or [224, 224]
        self.image_size = (int(size[0]), int(size[1]))
        for channel in CHANNELS:
            if channel not in self.bounds:
                raise ValueError(f"preprocessing_config.json has no outlier bounds for {channel}")

    def _load_sensor_side(self) -> None:
        """Load the parts only a sensor-taking graph needs, and insist they are present."""
        for label, path in self.sensor_files.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"this export takes a sensor vector, so the bundle needs its {label} ({path})"
                )
        self._features = _load_module("_rsl_pi_features", self.features_path)
        self._load_scaler()

    def _load_scaler(self) -> None:
        with open(self.scaler_path, "rb") as handle:
            bundle = pickle.load(handle)
        if not isinstance(bundle, dict) or "scalers" not in bundle or "features" not in bundle:
            raise ValueError("sensor_scaler.pkl is not the expected {'scalers', 'features'} bundle")
        self.feature_names = list(bundle["features"][FEATURE_SET])
        self._scaler = bundle["scalers"][f"{SPLIT}__{FEATURE_SET}"]
        expected = getattr(self._scaler, "n_features_in_", len(self.feature_names))
        if int(expected) != len(self.feature_names):
            raise ValueError(
                f"scaler expects {expected} features but the bundle lists {len(self.feature_names)}"
            )

    def _build_interpreter(self, interpreter_cls):
        """Allocate with batch pinned to 1, before anything is known about the inputs."""
        interpreter = interpreter_cls(model_path=str(self.model_path))
        for detail in interpreter.get_input_details():
            shape = detail["shape"].copy()
            shape[0] = 1
            interpreter.resize_tensor_input(detail["index"], shape, strict=False)
        interpreter.allocate_tensors()
        return interpreter

    def _bind_contract(self, interpreter) -> None:
        """Refuse any graph whose inputs are not the ones the code below feeds it."""
        details = interpreter.get_input_details()
        by_rank = {len(d["shape"]): d for d in details}
        wanted = {4} if self.image_only else {4, 2}
        if set(by_rank) != wanted or len(details) != len(wanted):
            raise ValueError(
                "expected "
                + ("one image input (rank 4)" if self.image_only
                   else "an image (rank 4) and a sensor vector (rank 2)")
                + "; got " + str([(d["name"], list(d["shape"])) for d in details])
            )
        image, sensor = by_rank[4], by_rank.get(2)
        if tuple(int(v) for v in image["shape"][1:]) != (*self.image_size, 3):
            raise ValueError(f"image input shape {list(image['shape'])} does not match {self.image_size}")
        if sensor is not None and int(sensor["shape"][1]) != len(self.feature_names):
            raise ValueError(
                f"sensor input takes {int(sensor['shape'][1])} features; the scaler bundle has "
                f"{len(self.feature_names)}"
            )
        outputs = interpreter.get_output_details()
        if len(outputs) != 1 or int(self._np.prod(outputs[0]["shape"])) != 1:
            raise ValueError(f"expected one scalar output, got {[list(o['shape']) for o in outputs]}")

        def describe(detail):
            return {"name": detail["name"], "shape": [int(v) for v in detail["shape"]],
                    "dtype": detail["dtype"].__name__}

        self._interpreter = interpreter
        self._image_input, self._sensor_input, self._output = image, sensor, outputs[0]
        self.input_details = {"image": describe(image)}
        if sensor is not None:
            self.input_details["sensor"] = describe(sensor)
        self.input_details["output"] = describe(outputs[0])

    def _load_metrics(self) -> None:
        """Read the published error figures. Missing files are simply omitted."""
        results = self.bundle_dir / "results"

        def read(path: Path):
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None

        split_dir = results / f"{SPLIT}_{FEATURE_SET}"
        if self.image_only:
            primary = read(split_dir / "image_ensemble_metrics.json") or {}
            tflite = read(split_dir / "image_ensemble_tflite_report.json") or {}
        else:
            primary = read(split_dir / "fusion_metrics.json") or {}
            tflite = read(split_dir / "tflite_report.json") or {}
        ablations = read(results / "ablations.json") or {}
        ensemble = read(results / "image_ensemble_lodo.json") or {}
        comparison = read(split_dir / "comparison_metrics.json") or {}
        # The evidence behind the card's claims: how much of the within-session score is
        # duplicate retrieval, and what the alternatives actually measured.
        experiments = read(results / "experiments.json") or {}

        variant = self.model_path.stem.rsplit("_", 1)[-1]
        served = ((tflite.get("variants") or {}).get(variant) or {})
        lodo = ablations.get("leave_one_day_out") or {}
        ensemble_folds = ensemble.get("folds") or {}
        lodo_mae = {}
        for model_name in ("sensor", "image", "fusion"):
            values = [
                day["metrics"][model_name]["MAE"]
                for day in lodo.values()
                if model_name in (day.get("metrics") or {})
            ]
            if values:
                lodo_mae[model_name] = sum(values) / len(values)
        # The ensemble is measured by its own run, through the code path that is served.
        ensemble_values = [
            day["metrics"]["image_ensemble"]["MAE"]
            for day in ensemble_folds.values()
            if "image_ensemble" in (day.get("metrics") or {})
        ]
        if ensemble_values:
            lodo_mae["image_ensemble"] = sum(ensemble_values) / len(ensemble_values)
            lodo_mae["single_head_average"] = (ensemble.get("mean_MAE") or {}).get(
                "single_head_average")

        self.metrics = {
            "variant": variant,
            # both metric files hold {"metrics": {"train": .., "val": .., "test": ..}}
            "within_session": ((primary.get("metrics") or {}).get("test")),
            "ensemble_heads": ensemble.get("heads") if self.image_only else None,
            "served_variant": served.get("metrics"),
            "served_variant_size_mb": served.get("size_mb"),
            "served_variant_latency_ms": served.get("latency_ms_per_sample_x86_cpu"),
            "variants": {
                name: {
                    "size_mb": v.get("size_mb"),
                    "mae": (v.get("metrics") or {}).get("MAE"),
                    "r2": (v.get("metrics") or {}).get("R2"),
                    "latency_ms": v.get("latency_ms_per_sample_x86_cpu"),
                }
                for name, v in (tflite.get("variants") or {}).items()
            },
            "comparison": comparison,
            # Branch-by-branch within-session scores, so the card can show the
            # comparison it is implicitly making rather than only the served figure.
            "branches": (comparison.get("test_metrics") or {}),
            "graph_consistency_max_abs_diff": comparison.get("graph_consistency_max_abs_diff"),
            # A 1-NN lookup on *untrained* features. The closer this is to a trained
            # score, the less the within-session number means.
            "lookup_probe": {
                split_name: {
                    "image_mae": (((probe.get("frozen_imagenet_features") or {})
                                   .get("metrics") or {}).get("MAE")),
                    "sensor_mae": (((probe.get("scaled_sensor_vector") or {})
                                    .get("metrics") or {}).get("MAE")),
                    "median_minutes_to_nearest_train_row": (
                        (probe.get("minutes_to_nearest_train_row") or {}).get("median")),
                }
                for split_name, probe in (ablations.get("nearest_neighbour_probe") or {}).items()
            },
            "experiments": {
                name: {
                    "seeds": len(variant.get("seeds") or []),
                    "sensor": (variant.get("mean_MAE") or {}).get("sensor"),
                    "image": (variant.get("mean_MAE") or {}).get("image"),
                    "fusion": (variant.get("mean_MAE") or {}).get("fusion"),
                    "ensemble": (variant.get("ensemble_of_seeds_MAE") or {}).get("image"),
                }
                for name, variant in experiments.items()
            },
            "leave_one_day_out_mae": lodo_mae,
            "leave_one_day_out_days": {
                key: {
                    "true_rsl": value.get("true_RSL"),
                    "fusion_mae": ((value.get("metrics") or {}).get("fusion") or {}).get("MAE"),
                    "image_mae": ((value.get("metrics") or {}).get("image") or {}).get("MAE"),
                    "sensor_mae": ((value.get("metrics") or {}).get("sensor") or {}).get("MAE"),
                    "ensemble_mae": (((ensemble_folds.get(key) or {}).get("metrics") or {})
                                     .get("image_ensemble") or {}).get("MAE"),
                    "samples": value.get("held_out_samples"),
                }
                for key, value in sorted(
                    (lodo or ensemble_folds).items(), key=lambda kv: int(kv[0]))
            },
        }
        self.verification = {
            "features": read(results / "pi_feature_verification.json"),
            "serving": read(results / "serving_transform_verification.json"),
        }

    # -------------------------------------------------------------- helpers --
    @property
    def requires_gas(self) -> bool:
        """Whether a gas reading is a precondition for using this export at all.

        The image-only ensemble needs none of the three channels, which is most of the
        reason it is served: a photograph is a precondition the kitchen can actually meet.
        """
        return not self.image_only

    @property
    def new_batch_mae_days(self) -> float | None:
        """Mean error of the served graph on a day never seen in training.

        Every item a user tracks is a batch the model has never seen, so this - not the
        within-session figure - is the honest size of the uncertainty to show. The figure
        comes from whichever model is actually being served, never from a better-scoring
        one in the same bundle.
        """
        lodo = self.metrics.get("leave_one_day_out_mae", {})
        if self.image_only:
            return lodo.get("image_ensemble")
        return lodo.get("fusion")

    def envelope_violations(self, reading: Reading) -> dict[str, float]:
        values = {"Temperature": reading.temperature, "Humidity": reading.humidity, "Gas": reading.gas}
        out = {}
        for channel, value in values.items():
            env = self.envelope.get(channel)
            if env and not (env["min"] <= value <= env["max"]):
                out[channel] = value
        return out

    def describe_envelope(self) -> str:
        parts = []
        units = {"Temperature": " °C", "Humidity": " % RH", "Gas": " (gas)"}
        for channel in CHANNELS:
            env = self.envelope.get(channel)
            if env:
                parts.append(f"{env['min']:g}–{env['max']:g}{units[channel]}")
        return ", ".join(parts)

    def load_image_bytes(self, data: bytes):
        if not self.available:
            raise ModelUnavailable(self.load_error or "model not loaded")
        return self._image_io.load_image_bytes(data)

    def load_image_path(self, path: str | Path):
        return self.load_image_bytes(Path(path).read_bytes())

    def new_builder(self):
        return self._features.SensorFeatureBuilder(bounds=self.bounds)

    def builder_from_state(self, state: dict | None):
        return self._features.SensorFeatureBuilder.from_state(state, self.bounds)

    def _invoke(self, pixels, scaled=None) -> float:
        np = self._np

        def prepare(detail, value):
            if detail["dtype"] in (np.int8, np.uint8):
                scale, zero = detail["quantization"]
                info = np.iinfo(detail["dtype"])
                value = np.round(value / scale + zero).clip(info.min, info.max)
            return value.astype(detail["dtype"])

        with self._lock:
            self._interpreter.set_tensor(self._image_input["index"], prepare(self._image_input, pixels))
            if self._sensor_input is not None:
                if scaled is None:
                    raise ModelUnavailable("this export needs the 27-feature sensor vector")
                self._interpreter.set_tensor(self._sensor_input["index"],
                                             prepare(self._sensor_input, scaled))
            self._interpreter.invoke()
            raw = self._interpreter.get_tensor(self._output["index"])
        if self._output["dtype"] in (np.int8, np.uint8):
            scale, zero = self._output["quantization"]
            raw = (raw.astype("float32") - zero) * scale
        return float(np.asarray(raw).ravel()[0])

    # -------------------------------------------------------------- predict --
    def predict_scaled(self, pixels, scaled=None) -> float:
        """Forward pass from already-prepared inputs (held-out evaluation).

        ``scaled`` is the already-scaled 27-feature vector the fusion export needs, and is
        left out entirely for the image-only ensemble.
        """
        if not self.available:
            raise ModelUnavailable(self.load_error or "model not loaded")
        if scaled is None:
            return self._invoke(pixels)
        return self._invoke(pixels, scaled.astype("float32"))

    def scale(self, raw_vector):
        return self._scaler.transform(self._np.asarray(raw_vector, dtype="float64")).astype("float32")

    def _predict_image_only(self, image, history: list) -> RSLResult:
        """The served ensemble: the photo is the whole input.

        No sensor vector means no cold-start caveat and no envelope check - neither can
        apply to an estimate the readings never reached.
        """
        started = time.perf_counter()
        raw = self._invoke(image.pixels)
        elapsed = (time.perf_counter() - started) * 1000.0
        days = min(max(raw, 0.0), self.max_days)
        warnings = list(image.notes)
        if days != raw:
            warnings.append(f"Raw model output {raw:.2f} days was clamped to 0–{self.max_days:g}.")
        return RSLResult(
            raw_days=raw, days=days, reading_number=len(history), inference_ms=elapsed,
            decoder=image.decoder, warnings=warnings, adjustments={}, out_of_envelope={},
        )

    def predict(self, image, history: Iterable[Reading] = (), builder=None) -> RSLResult:
        """Predict from a photo and, for the fusion export, the session's readings.

        ``image`` is a loaded image (from :meth:`load_image_bytes`). The whole history is
        replayed through a fresh feature builder unless ``builder`` carries on an existing
        session, in which case ``history`` should hold only the new readings.

        The image-only ensemble does not read ``history`` at all; it is still accepted so
        both exports share one call signature, and only labels the result.
        """
        if not self.available:
            raise ModelUnavailable(self.load_error or "model not loaded")
        history = list(history)
        if self.image_only:
            return self._predict_image_only(image, history)
        if not history:
            raise ModelUnavailable("the trained model needs at least one sensor reading")
        for reading in history:
            for label, value in (("temperature", reading.temperature),
                                 ("humidity", reading.humidity), ("gas", reading.gas)):
                if value is None:
                    raise ModelUnavailable(f"a reading is missing its {label} value")

        fresh = builder is None
        builder = builder or self.new_builder()
        adjustments: dict[str, dict] = {}
        vector = None
        for reading in history:
            vector = builder.vector(
                {"Temperature": reading.temperature, "Humidity": reading.humidity, "Gas": reading.gas},
                reading.hours_since_previous,
                self.feature_names,
            )
            adjustments.update(builder.last_adjustments)

        started = time.perf_counter()
        raw = self._invoke(image.pixels, self.scale(vector))
        elapsed = (time.perf_counter() - started) * 1000.0
        days = min(max(raw, 0.0), self.max_days)

        current = history[-1]
        reading_number = len(history) if fresh else None
        violations = self.envelope_violations(current)

        warnings = list(image.notes)
        if reading_number == 1:
            warnings.append(
                "First reading of this session: the rolling sensor features have no history "
                "yet, so the estimate leans almost entirely on the photo."
            )
        for channel, value in violations.items():
            env = self.envelope[channel]
            warnings.append(
                f"{channel} {value:g} is outside every reading the model was trained on "
                f"({env['min']:g}–{env['max']:g}); the estimate is an extrapolation."
            )
        for channel, change in adjustments.items():
            how = "clipped to" if change["how"] == "clipped" else "replaced by the last in-range value,"
            warnings.append(
                f"An outlying {channel} reading ({change['raw']:g}) was {how} "
                f"{change['used']:.2f} for the smoothed features, as in training."
            )
        if days != raw:
            warnings.append(f"Raw model output {raw:.2f} days was clamped to 0–{self.max_days:g}.")

        return RSLResult(
            raw_days=raw, days=days,
            reading_number=reading_number if reading_number is not None else 0,
            inference_ms=elapsed, decoder=image.decoder, warnings=warnings,
            adjustments=adjustments, out_of_envelope=violations,
        )

    # --------------------------------------------------------------- report --
    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "available": self.available,
            "load_error": self.load_error,
            "runtime": self.runtime,
            "image_only": self.image_only,
            "bundle_dir": str(self.bundle_dir),
            "files": {label: {"path": str(path), "exists": path.is_file()}
                      for label, path in self.required_files.items()},
            "feature_names": list(self.feature_names),
            "feature_count": len(self.feature_names),
            "image_size": list(self.image_size),
            "max_days": self.max_days,
            "envelope": self.envelope,
            "outlier_bounds": self.bounds,
            "inputs": getattr(self, "input_details", {}),
            "metrics": self.metrics,
            "new_batch_mae_days": self.new_batch_mae_days,
            "verification": self.verification,
        }
