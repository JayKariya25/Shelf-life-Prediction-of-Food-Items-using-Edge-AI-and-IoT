"""Freshness / remaining-shelf-life estimation.

Two estimators live behind one interface:

``KineticBaselinePredictor``
    A temperature-and-humidity kinetic model (Q10 formulation) parameterised by
    published postharvest storage guidance. It uses **no image data** and it is
    **not** a trained model. Every response it produces is tagged
    ``model_kind="heuristic-baseline"`` so the UI can say so out loud.

``TFLiteModelPredictor``
    Loads a trained multimodal model exported to TFLite when one is present on
    disk and a runtime is importable. Until the team trains and drops a model in
    ``models/``, this predictor stays inactive and the app degrades to the
    baseline rather than pretending.

The registry picks the trained model when it is available and falls back
otherwise. Nothing here fabricates accuracy figures or confidence scores that
the underlying method cannot support.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Freshness vocabulary used across the app. Matches the project specification.
FRESH = "Fresh"
MODERATE = "Moderately Fresh"
SPOILED = "Spoiled"
FRESHNESS_CLASSES = (FRESH, MODERATE, SPOILED)

# Fraction of the item's modelled life still remaining at each class boundary.
MODERATE_THRESHOLD = 0.55
SPOILED_THRESHOLD = 0.08

# The baseline is a coarse model; this is the half-width of the reported
# interval, expressed as a fraction of the point estimate. It is a stated
# modelling assumption, not a measured error bar.
BASELINE_RELATIVE_UNCERTAINTY = 0.35

MAX_MODELLED_HOURS = 24 * 45  # refuse to extrapolate beyond ~6 weeks


@dataclass(slots=True)
class PredictionContext:
    """Everything an estimator may look at for one item."""

    food_key: str = "other"
    food_name: str = "Other produce"
    ref_shelf_life_hours: float = 168.0
    ref_temperature_c: float = 20.0
    q10: float = 2.4
    ideal_temp_min_c: float | None = None
    ideal_temp_max_c: float | None = None
    ideal_humidity_min: float | None = None
    ideal_humidity_max: float | None = None
    temperature_c: float | None = None
    humidity_pct: float | None = None
    gas_ppm: float | None = None
    hours_stored: float = 0.0
    image_path: str | None = None
    # Absolute path to the item's photograph. The trained multimodal model needs
    # the pixels; the kinetic baseline ignores this entirely.
    image_full_path: str | None = None
    # Environment actually experienced while stored, as
    # ``(duration_hours, temperature_c, humidity_pct, gas_ppm)`` segments. When
    # present the baseline integrates degradation across it instead of assuming
    # the current reading held for the whole storage period.
    history: list[tuple[float, float | None, float | None, float | None]] = field(
        default_factory=list
    )


@dataclass(slots=True)
class Prediction:
    """Estimator output. Serialised straight to JSON by the API."""

    freshness_class: str
    remaining_hours: float
    remaining_hours_low: float
    remaining_hours_high: float
    model_name: str
    model_version: str
    model_kind: str            # "heuristic-baseline" | "trained-model"
    freshness_confidence: float | None = None
    inference_ms: float = 0.0
    image_used: bool = False
    rationale: str = ""
    # Set when a trained model was configured but could not serve this request,
    # so the UI can explain the downgrade instead of silently showing baseline
    # numbers under a "trained model" banner.
    fallback_reason: str | None = None
    factors: dict[str, float] = field(default_factory=dict)

    @property
    def days(self) -> int:
        return int(self.remaining_hours // 24)

    @property
    def hours(self) -> int:
        return int(round(self.remaining_hours % 24))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["days"] = self.days
        data["hours"] = self.hours
        data["remaining_hours"] = round(self.remaining_hours, 2)
        data["remaining_hours_low"] = round(self.remaining_hours_low, 2)
        data["remaining_hours_high"] = round(self.remaining_hours_high, 2)
        data["inference_ms"] = round(self.inference_ms, 2)
        data["status_color"] = status_color(self.freshness_class)
        return data


def status_color(freshness_class: str) -> str:
    return {FRESH: "ok", MODERATE: "warn", SPOILED: "crit"}.get(freshness_class, "muted")


def classify(fraction_remaining: float) -> str:
    """Map fraction of modelled life remaining onto the freshness vocabulary."""
    if fraction_remaining <= SPOILED_THRESHOLD:
        return SPOILED
    if fraction_remaining <= MODERATE_THRESHOLD:
        return MODERATE
    return FRESH


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def humidity_factor(
    humidity: float | None, ideal_min: float | None, ideal_max: float | None
) -> float:
    """Multiplier <= 1 for humidity outside the produce's ideal band.

    Too damp accelerates microbial growth; too dry drives transpiration losses
    and shrivelling. Damp is penalised harder because it is the dominant
    spoilage route for the produce this project targets.
    """
    if humidity is None or ideal_min is None or ideal_max is None:
        return 1.0
    humidity = _clamp(humidity, 0.0, 100.0)
    if ideal_min <= humidity <= ideal_max:
        return 1.0
    if humidity > ideal_max:
        return max(0.45, 1.0 - 0.012 * (humidity - ideal_max))
    return max(0.60, 1.0 - 0.006 * (ideal_min - humidity))


def chilling_factor(temperature: float | None, ideal_min: float | None) -> float:
    """Penalty for chilling injury in cold-sensitive produce.

    Tomato, banana, mango, cucumber and capsicum are damaged below roughly
    7-13 C. A pure Q10 extrapolation would happily predict weeks of life in a
    fridge, which is wrong for these items, so the penalty is applied only when
    the profile's ideal minimum is itself above 7 C.
    """
    if temperature is None or ideal_min is None or ideal_min < 7.0:
        return 1.0
    if temperature >= ideal_min:
        return 1.0
    return max(0.30, 1.0 - 0.08 * (ideal_min - temperature))


def is_chilling_sensitive(ideal_min: float | None) -> bool:
    """Produce whose ideal storage floor sits above 7 C is cold-sensitive."""
    return ideal_min is not None and ideal_min >= 7.0


def effective_temperature(temperature: float, ideal_min: float | None) -> float:
    """Temperature used in the Q10 term.

    For chilling-sensitive produce, cooling below the chilling threshold buys no
    further slowdown -- it only causes injury, which
    :func:`chilling_factor` accounts for separately. Clamping here stops the Q10
    extrapolation from rewarding a fridge that is actively damaging the item.
    """
    if is_chilling_sensitive(ideal_min) and temperature < ideal_min:
        return ideal_min
    return temperature


def gas_factor(gas_ppm: float | None) -> float:
    """Optional MQ-135 style VOC reading treated as a spoilage-gas proxy.

    The sensor is not part of the core BMP280 + SHT31 build, so this is applied
    only when a reading is actually present.
    """
    if gas_ppm is None:
        return 1.0
    if gas_ppm <= 150:
        return 1.0
    if gas_ppm <= 220:
        return _clamp(1.0 - (gas_ppm - 150) * (0.30 / 70.0), 0.70, 1.0)
    return max(0.35, 0.70 - (gas_ppm - 220) * 0.004)


class KineticBaselinePredictor:
    """Q10 kinetic shelf-life baseline.

    ``shelf_life(T) = ref_shelf_life * q10 ** ((ref_temp - T) / 10)``

    is the standard Q10 formulation for temperature-dependent reaction rates,
    then adjusted by multiplicative humidity, chilling and gas factors. It gives
    the project a defensible, reproducible reference line to beat once the
    multimodal model is trained.
    """

    name = "Q10 kinetic baseline"
    version = "1.0.0"
    kind = "heuristic-baseline"
    uses_image = False

    def life_hours(
        self,
        ctx: PredictionContext,
        temperature: float | None,
        humidity: float | None,
        gas: float | None,
    ) -> tuple[float, dict[str, float]]:
        """Total modelled shelf life, in hours, under one steady condition."""
        modelled_temp = _clamp(
            temperature if temperature is not None else ctx.ref_temperature_c, -2.0, 45.0
        )
        q10 = _clamp(ctx.q10 or 2.4, 1.2, 4.0)
        base = max(ctx.ref_shelf_life_hours, 1.0)
        kinetic_temp = effective_temperature(modelled_temp, ctx.ideal_temp_min_c)
        exponent = (ctx.ref_temperature_c - kinetic_temp) / 10.0

        f_humidity = humidity_factor(humidity, ctx.ideal_humidity_min, ctx.ideal_humidity_max)
        f_chill = chilling_factor(temperature, ctx.ideal_temp_min_c)
        f_gas = gas_factor(gas)

        total = base * (q10 ** exponent) * f_humidity * f_chill * f_gas
        detail = {
            "modelled_temperature_c": round(modelled_temp, 2),
            "kinetic_temperature_c": round(kinetic_temp, 2),
            "humidity_factor": round(f_humidity, 3),
            "chilling_factor": round(f_chill, 3),
            "gas_factor": round(f_gas, 3),
            "q10": round(q10, 2),
        }
        return _clamp(total, 1.0, MAX_MODELLED_HOURS), detail

    def consumed_fraction(self, ctx: PredictionContext, fallback_total: float) -> float:
        """Fraction of shelf life already used up.

        With an environment history the degradation rate ``1 / life_hours`` is
        integrated segment by segment, so an item that spent two days in a warm
        room and then went into the fridge is charged for the warm days. Without
        history it falls back to assuming the current condition held throughout.
        """
        if not ctx.history:
            return max(0.0, ctx.hours_stored) / fallback_total if fallback_total > 0 else 1.0

        consumed = 0.0
        covered = 0.0
        for duration, temperature, humidity, gas in ctx.history:
            if duration <= 0:
                continue
            segment_life, _ = self.life_hours(ctx, temperature, humidity, gas)
            consumed += duration / segment_life
            covered += duration
        # Any storage time before the first recorded reading is charged at the
        # current condition rather than treated as free.
        uncovered = max(0.0, ctx.hours_stored - covered)
        if uncovered > 0 and fallback_total > 0:
            consumed += uncovered / fallback_total
        return consumed

    def predict(self, ctx: PredictionContext) -> Prediction:
        started = time.perf_counter()

        temperature = ctx.temperature_c
        total_hours, detail = self.life_hours(
            ctx, temperature, ctx.humidity_pct, ctx.gas_ppm
        )
        modelled_temp = detail["modelled_temperature_c"]
        kinetic_temp = detail["kinetic_temperature_c"]
        f_humidity = detail["humidity_factor"]
        f_chill = detail["chilling_factor"]
        f_gas = detail["gas_factor"]

        consumed = self.consumed_fraction(ctx, total_hours)
        fraction = _clamp(1.0 - consumed, 0.0, 1.0)
        elapsed = max(0.0, ctx.hours_stored)
        # Remaining time is the unused fraction spent at the *current* rate,
        # which is what the user is asking about: "how long from now?"
        remaining = fraction * total_hours
        freshness = classify(fraction)

        spread = remaining * BASELINE_RELATIVE_UNCERTAINTY
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        rationale = self._explain(
            ctx, modelled_temp, kinetic_temp, total_hours, f_humidity, f_chill, f_gas
        )

        return Prediction(
            freshness_class=freshness,
            remaining_hours=remaining,
            remaining_hours_low=max(0.0, remaining - spread),
            remaining_hours_high=remaining + spread,
            model_name=self.name,
            model_version=self.version,
            model_kind=self.kind,
            # A rule-based estimator has no calibrated posterior. Reporting a
            # number here would be fabricating one, so it stays None.
            freshness_confidence=None,
            inference_ms=elapsed_ms,
            image_used=False,
            rationale=rationale,
            factors={
                "modelled_total_hours": round(total_hours, 2),
                "hours_stored": round(elapsed, 2),
                "fraction_remaining": round(fraction, 4),
                "fraction_consumed": round(consumed, 4),
                "history_segments": len(ctx.history),
                **detail,
            },
        )

    @staticmethod
    def _explain(
        ctx: PredictionContext,
        modelled_temp: float,
        kinetic_temp: float,
        total_hours: float,
        f_humidity: float,
        f_chill: float,
        f_gas: float,
    ) -> str:
        parts = [
            f"{ctx.food_name}: {ctx.ref_shelf_life_hours:.0f} h reference life at "
            f"{ctx.ref_temperature_c:.0f} °C, Q10 = {ctx.q10:.1f}, scaled to "
            f"{modelled_temp:.1f} °C."
        ]
        if f_humidity < 0.999:
            direction = "above" if (ctx.humidity_pct or 0) > (ctx.ideal_humidity_max or 0) else "below"
            parts.append(
                f"Humidity {ctx.humidity_pct:.0f}% is {direction} the "
                f"{ctx.ideal_humidity_min:.0f}-{ctx.ideal_humidity_max:.0f}% ideal band "
                f"(x{f_humidity:.2f})."
            )
        if f_chill < 0.999:
            parts.append(
                f"Stored below the {ctx.ideal_temp_min_c:.0f} °C chilling threshold "
                f"for this produce, so the rate model is held at "
                f"{kinetic_temp:.0f} °C and an injury penalty applied (x{f_chill:.2f})."
            )
        if f_gas < 0.999:
            parts.append(f"Elevated spoilage-gas reading (x{f_gas:.2f}).")
        parts.append(f"Modelled total life {total_hours / 24:.1f} days at the current condition.")
        if ctx.history:
            parts.append(
                f"Degradation integrated over {len(ctx.history)} recorded environment "
                f"segments covering {ctx.hours_stored:.1f} h of storage."
            )
        return " ".join(parts)


class TFLiteModelPredictor:
    """Adapter for a trained multimodal model exported to TFLite.

    Activates only when *both* a model file exists and a TFLite runtime can be
    imported. Any failure during load leaves the predictor inactive so the app
    falls back to the baseline instead of crashing on the edge device.
    """

    kind = "trained-model"
    uses_image = True

    def __init__(self, model_path: str, labels_path: str = "") -> None:
        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path) if labels_path else None
        self.name = self.model_path.stem or "trained model"
        self.version = "unknown"
        self.available = False
        self.load_error: str | None = None
        self._interpreter = None
        self._labels: list[str] = list(FRESHNESS_CLASSES)
        self._load()

    def _load(self) -> None:
        if not self.model_path.exists():
            self.load_error = f"No model file at {self.model_path}"
            return
        interpreter_cls = None
        try:  # tflite-runtime is the lightweight option on a Raspberry Pi
            from tflite_runtime.interpreter import Interpreter as interpreter_cls  # type: ignore
        except ImportError:
            try:
                from tensorflow.lite import Interpreter as interpreter_cls  # type: ignore
            except ImportError:
                self.load_error = (
                    "Neither tflite-runtime nor tensorflow is installed; "
                    "install tflite-runtime on the Pi to enable the trained model."
                )
                return
        try:
            self._interpreter = interpreter_cls(model_path=str(self.model_path))
            self._interpreter.allocate_tensors()
            if self.labels_path and self.labels_path.exists():
                payload = json.loads(self.labels_path.read_text(encoding="utf-8"))
                self._labels = list(payload.get("labels", FRESHNESS_CLASSES))
                self.version = str(payload.get("version", "unknown"))
                self.name = str(payload.get("name", self.name))
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on a real model file
            self.load_error = f"{type(exc).__name__}: {exc}"
            self._interpreter = None
            self.available = False

    def predict(self, ctx: PredictionContext) -> Prediction:  # pragma: no cover
        if not self.available or self._interpreter is None:
            raise RuntimeError("Trained model is not loaded")
        raise NotImplementedError(
            "Wire the trained model's input tensors here once the export format "
            "is fixed: image tensor + [temperature, humidity] sensor tensor."
        )


class PredictorRegistry:
    """Chooses the best estimator that can actually serve each request.

    The trained multimodal model needs an image *and* a gas reading. When either
    is missing for a particular item, this falls back to the kinetic baseline and
    records why, rather than fabricating the missing input.
    """

    def __init__(
        self,
        model_path: str = "",
        labels_path: str = "",
        keras_model_path: str = "",
        scaler_path: str = "",
        config_path: str = "",
    ) -> None:
        self.baseline = KineticBaselinePredictor()
        self.trained: TFLiteModelPredictor | None = None
        self.multimodal = None

        if keras_model_path:
            from .multimodal import MultimodalShelfLifeModel

            self.multimodal = MultimodalShelfLifeModel(
                keras_model_path, scaler_path, config_path
            )
            if not self.multimodal.available:
                log.info(
                    "Trained multimodal model unavailable, using baseline: %s",
                    self.multimodal.load_error,
                )

        if model_path:
            candidate = TFLiteModelPredictor(model_path, labels_path)
            self.trained = candidate
            if not candidate.available:
                log.info("TFLite model unavailable, using baseline: %s", candidate.load_error)

    # --- selection --------------------------------------------------------
    @property
    def has_trained_model(self) -> bool:
        return bool(
            (self.multimodal is not None and self.multimodal.available)
            or (self.trained is not None and self.trained.available)
        )

    @property
    def active(self):
        """The best estimator in principle, ignoring per-request inputs."""
        if self.multimodal is not None and self.multimodal.available:
            return self.multimodal
        if self.trained is not None and self.trained.available:
            return self.trained
        return self.baseline

    def blockers(self, ctx: PredictionContext) -> list[str]:
        """Why the trained model cannot serve this particular context."""
        reasons: list[str] = []
        if self.multimodal is None or not self.multimodal.available:
            return ["No trained model is loaded."]
        if not ctx.image_full_path or not Path(ctx.image_full_path).is_file():
            reasons.append("this item has no photograph on file")
        if ctx.gas_ppm is None:
            reasons.append("no gas reading is available (the model was trained on Temperature, Humidity and Gas)")
        return reasons

    def predict(self, ctx: PredictionContext) -> Prediction:
        if self.multimodal is not None and self.multimodal.available:
            blockers = self.blockers(ctx)
            if not blockers:
                try:
                    return self._predict_multimodal(ctx)
                except Exception as exc:
                    log.exception("Trained model failed, falling back to baseline")
                    return self._baseline_with_reason(
                        ctx, f"the trained model raised {type(exc).__name__}: {exc}"
                    )
            return self._baseline_with_reason(ctx, " and ".join(blockers))

        if self.trained is not None and self.trained.available:
            try:
                return self.trained.predict(ctx)
            except Exception as exc:
                log.exception("TFLite model failed, falling back to baseline")
                return self._baseline_with_reason(ctx, f"the TFLite model raised {exc}")

        return self.baseline.predict(ctx)

    def _baseline_with_reason(self, ctx: PredictionContext, reason: str) -> Prediction:
        prediction = self.baseline.predict(ctx)
        if self.multimodal is not None or self.trained is not None:
            prediction.fallback_reason = (
                f"Estimated with the kinetic baseline because {reason}."
            )
        return prediction

    def _predict_multimodal(self, ctx: PredictionContext) -> Prediction:
        model = self.multimodal
        image = model.open_image(ctx.image_full_path)
        output = model.predict(image, ctx.temperature_c, ctx.humidity_pct, ctx.gas_ppm)

        # The network regresses remaining shelf life in days; clamp at zero for
        # display without touching the raw value used in the rationale.
        raw_days = output.remaining_days
        remaining_hours = max(0.0, raw_days) * 24.0

        # The class is derived from the regression output, not separately
        # predicted; the rationale says so.
        total = remaining_hours + max(0.0, ctx.hours_stored)
        fraction = remaining_hours / total if total > 0 else 0.0
        freshness = classify(fraction)

        # Only report an interval if the training run published an error metric.
        mae_hours = None
        for key in ("test_mae", "mae", "val_mae"):
            if key in model.metrics:
                mae_hours = float(model.metrics[key]) * 24.0
                break
        low = max(0.0, remaining_hours - mae_hours) if mae_hours is not None else None
        high = remaining_hours + mae_hours if mae_hours is not None else None

        rationale = (
            f"{model.name} predicted {raw_days:.2f} days of remaining shelf life from the "
            f"item photograph and the sensor triple "
            f"(T={ctx.temperature_c:.1f}, RH={ctx.humidity_pct:.1f}, Gas={ctx.gas_ppm:.1f}). "
            f"The freshness class is derived from that regression output, not predicted "
            f"separately."
        )
        if mae_hours is not None:
            rationale += f" The interval is the model's held-out MAE of {mae_hours / 24:.2f} days."
        else:
            rationale += " No held-out error metric is recorded, so no interval is shown."

        return Prediction(
            freshness_class=freshness,
            remaining_hours=remaining_hours,
            remaining_hours_low=low,
            remaining_hours_high=high,
            model_name=model.name,
            model_version=model.version,
            model_kind=model.kind,
            # A single-output regressor emits no class probability.
            freshness_confidence=None,
            inference_ms=output.inference_ms,
            image_used=True,
            rationale=rationale,
            factors={
                "predicted_days_raw": round(raw_days, 4),
                "hours_stored": round(max(0.0, ctx.hours_stored), 2),
                "fraction_remaining": round(fraction, 4),
                "temperature_c": ctx.temperature_c,
                "humidity_pct": ctx.humidity_pct,
                "gas_ppm": ctx.gas_ppm,
            },
        )

    # --- reporting --------------------------------------------------------
    def status(self) -> dict[str, Any]:
        active = self.active
        status = {
            "active_model": active.name,
            "active_version": active.version,
            "active_kind": active.kind,
            "uses_image": bool(getattr(active, "uses_image", False)),
            "requires_gas": bool(getattr(active, "requires_gas", False)),
            "trained_model_configured": self.multimodal is not None or self.trained is not None,
            "trained_model_available": self.has_trained_model,
            "trained_model_error": None,
            "baseline_name": self.baseline.name,
            "baseline_version": self.baseline.version,
            "supports_calibrated_confidence": False,
        }
        if self.multimodal is not None:
            status["trained_model_error"] = self.multimodal.load_error
            status["multimodal"] = self.multimodal.status()
        elif self.trained is not None:
            status["trained_model_error"] = self.trained.load_error
        return status
