"""Freshness / remaining-shelf-life estimation.

Two estimators live behind one registry:

``RSLModel`` (see :mod:`shelflife.rsl_model`)
    The project's trained multimodal model: an item photograph plus the session's
    Temperature / Humidity / Gas readings -> remaining shelf life in days. Tagged
    ``model_kind="trained-model"``.

``KineticBaselinePredictor``
    A temperature-and-humidity kinetic model (Q10 formulation) parameterised by
    published postharvest storage guidance. It uses **no image data** and it is
    **not** a trained model; every result is tagged ``model_kind="heuristic-baseline"``.

For each item the registry uses the trained model only when it can be used
*honestly* - a recent photo, gas readings, and readings inside the range the model
was trained on - and otherwise falls back to the baseline and records why. Nothing
here fabricates accuracy figures or confidence scores the method cannot support.
"""

from __future__ import annotations

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
    # When that photograph was taken, as hours before the newest reading.
    image_age_hours: float | None = None
    # The session's sensor readings, oldest first, already thinned to the training
    # cadence - the trained model replays them to build its rolling features.
    rsl_history: list = field(default_factory=list)
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
    remaining_hours_low: float | None
    remaining_hours_high: float | None
    model_name: str
    model_version: str
    model_kind: str            # "heuristic-baseline" | "trained-model"
    # What this estimator read, for the item page. Set by each predictor.
    model_inputs: str = ""
    freshness_confidence: float | None = None
    inference_ms: float = 0.0
    image_used: bool = False
    rationale: str = ""
    # Set when a trained model was configured but could not serve this request,
    # so the UI can explain the downgrade instead of silently showing baseline
    # numbers under a "trained model" banner.
    fallback_reason: str | None = None
    notes: list[str] = field(default_factory=list)
    factors: dict[str, Any] = field(default_factory=dict)

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
        for key in ("remaining_hours_low", "remaining_hours_high"):
            value = getattr(self, key)
            data[key] = round(value, 2) if value is not None else None
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
    """Always 1.0: the baseline does not use the gas channel.

    An earlier version penalised readings above ~150 as if they were MQ-135 ppm. The
    project's gas sensor reports raw values of roughly 150-195 throughout an entire
    spoilage run, in uncalibrated units, so those thresholds penalised *every* normal
    reading. Gas is used by the trained model, which learned its scale from the data.
    """
    return 1.0


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
            model_inputs="temperature and humidity only; the photo is not used",
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
        parts.append(f"Modelled total life {total_hours / 24:.1f} days at the current condition.")
        if ctx.history:
            parts.append(
                f"Degradation integrated over {len(ctx.history)} recorded environment "
                f"segments covering {ctx.hours_stored:.1f} h of storage."
            )
        return " ".join(parts)


class PredictorRegistry:
    """Chooses the estimator that can honestly serve each request."""

    def __init__(
        self,
        rsl_bundle_dir: str = "",
        rsl_model_file: str = "models/image_ensemble_blocked_full_float16.tflite",
        max_photo_age_hours: float = 6.0,
        trained_food_keys: tuple[str, ...] = ("banana",),
    ) -> None:
        self.baseline = KineticBaselinePredictor()
        self.max_photo_age_hours = max_photo_age_hours
        self.trained_food_keys = tuple(k.lower() for k in trained_food_keys)
        self.rsl = None
        if rsl_bundle_dir:
            from .rsl_model import RSLModel

            self.rsl = RSLModel(rsl_bundle_dir, rsl_model_file)
            if not self.rsl.available:
                log.info("Trained RSL model unavailable, using baseline: %s", self.rsl.load_error)

    # --- selection --------------------------------------------------------
    @property
    def has_trained_model(self) -> bool:
        return self.rsl is not None and self.rsl.available

    @property
    def active(self):
        """The best estimator in principle, ignoring per-request inputs."""
        return self.rsl if self.has_trained_model else self.baseline

    def blockers(self, ctx: PredictionContext) -> list[str]:
        """Why the trained model cannot serve this particular item right now."""
        if not self.has_trained_model:
            detail = self.rsl.load_error if self.rsl is not None else None
            return [f"no trained model is loaded ({detail})" if detail
                    else "no trained model is loaded"]
        reasons: list[str] = []
        # The model saw one food. Scoring another one would be a confident number about
        # something it has never seen, so it is refused rather than quietly served.
        if self.trained_food_keys and ctx.food_key.lower() not in self.trained_food_keys:
            trained = " or ".join(self.trained_food_keys)
            reasons.append(
                f"the trained model was fitted on {trained} only, and this item is "
                f"{ctx.food_name.lower()} (its photograph would be scored by a model that "
                f"has never seen this food)"
            )
        if not ctx.image_full_path or not Path(ctx.image_full_path).is_file():
            reasons.append("this item has no photograph on file")
        elif ctx.image_age_hours is not None and ctx.image_age_hours > self.max_photo_age_hours:
            reasons.append(
                f"its latest photograph is {ctx.image_age_hours:.1f} h older than the newest "
                f"reading (the model estimates from how the food looks now, so it needs one "
                f"under {self.max_photo_age_hours:g} h old)"
            )
        # The served image-only ensemble reads nothing but the photograph, so the three
        # sensor preconditions below do not apply to it.
        if self.rsl.image_only:
            return reasons
        if not ctx.rsl_history:
            if ctx.gas_ppm is None:
                reasons.append(
                    "no gas reading is available (the model was trained on Temperature, "
                    "Humidity and Gas)"
                )
            else:
                reasons.append("no sensor readings have been recorded for this item yet")
        else:
            violations = self.rsl.envelope_violations(ctx.rsl_history[-1])
            if violations:
                listed = ", ".join(f"{k} {v:g}" for k, v in violations.items())
                reasons.append(
                    f"the current reading ({listed}) is outside the range the model was "
                    f"trained on ({self.rsl.describe_envelope()})"
                )
        return reasons

    def predict(self, ctx: PredictionContext) -> Prediction:
        if self.rsl is None:
            return self.baseline.predict(ctx)
        blockers = self.blockers(ctx)
        if blockers:
            return self._baseline_with_reason(ctx, " and ".join(blockers))
        try:
            return self._predict_rsl(ctx)
        except Exception as exc:
            log.exception("Trained model failed, falling back to baseline")
            return self._baseline_with_reason(
                ctx, f"the trained model raised {type(exc).__name__}: {exc}"
            )

    def _baseline_with_reason(self, ctx: PredictionContext, reason: str) -> Prediction:
        prediction = self.baseline.predict(ctx)
        if self.rsl is not None:
            prediction.fallback_reason = f"Estimated with the kinetic baseline because {reason}."
        return prediction

    def _predict_rsl(self, ctx: PredictionContext) -> Prediction:
        model = self.rsl
        image = model.load_image_path(ctx.image_full_path)
        result = model.predict(image, ctx.rsl_history)

        remaining_hours = result.days * 24.0
        # The model is trained on RSL = 11 - Day, so the whole life is max_days by
        # construction and the fraction left follows directly.
        fraction = result.days / model.max_days if model.max_days else 0.0
        freshness = classify(fraction)

        # Every tracked item is a batch the model has never seen, so the honest
        # interval is the leave-one-day-out error, not the within-session one.
        mae_days = model.new_batch_mae_days
        if mae_days is not None:
            spread = mae_days * 24.0
            low = max(0.0, remaining_hours - spread)
            high = min(model.max_days * 24.0, remaining_hours + spread)
        else:
            low = high = None

        if model.image_only:
            heads = (model.metrics.get("ensemble_heads") or 0)
            rationale = (
                f"{model.name} ({model.version}) estimated {result.days:.2f} days from the "
                f"item photograph alone"
                + (f", averaged over {heads} independently trained heads" if heads else "")
                + ". The sensor readings are not an input to it: on a batch it had never "
                  "seen, the sensor branch described the room rather than the food, so the "
                  "photograph-only model is both more accurate and usable without a gas "
                  "sensor. The freshness class is derived from that estimate, not predicted "
                  "separately."
            )
        else:
            latest = ctx.rsl_history[-1]
            rationale = (
                f"{model.name} ({model.version}) estimated {result.days:.2f} days from the item "
                f"photograph and {len(ctx.rsl_history)} sensor reading"
                f"{'' if len(ctx.rsl_history) == 1 else 's'} (latest T {latest.temperature:.2f}, "
                f"RH {latest.humidity:.2f}, gas {latest.gas:g}). The freshness class is derived "
                f"from that estimate, not predicted separately."
            )
        if mae_days is not None:
            rationale += (
                f" The ±{mae_days:.2f}-day band is the model's mean error on days it never saw "
                f"in training - the right figure for a new batch."
            )
        else:
            rationale += " No held-out error figure was found, so no band is shown."

        return Prediction(
            freshness_class=freshness,
            remaining_hours=remaining_hours,
            remaining_hours_low=low,
            remaining_hours_high=high,
            model_name=model.name,
            model_version=model.version,
            model_kind=model.kind,
            model_inputs=("photo only" if model.image_only
                          else "photo + sensor history"),
            # A single-output regressor emits no class probability.
            freshness_confidence=None,
            inference_ms=result.inference_ms,
            image_used=True,
            rationale=rationale,
            notes=list(result.warnings),
            factors={
                "predicted_days_raw": round(result.raw_days, 4),
                "readings_in_session": result.reading_number,
                "fraction_remaining": round(fraction, 4),
                "image_decoder": result.decoder,
                "image_age_hours": ctx.image_age_hours,
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
            "trained_food_keys": list(self.trained_food_keys),
            "trained_model_configured": self.rsl is not None,
            "trained_model_available": self.has_trained_model,
            "trained_model_error": self.rsl.load_error if self.rsl is not None else None,
            "baseline_name": self.baseline.name,
            "baseline_version": self.baseline.version,
            "supports_calibrated_confidence": False,
            "max_photo_age_hours": self.max_photo_age_hours,
        }
        if self.rsl is not None:
            status["rsl"] = self.rsl.status()
        return status
