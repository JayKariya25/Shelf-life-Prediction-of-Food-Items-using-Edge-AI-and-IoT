"""The estimator's maths and its honesty guarantees."""

from __future__ import annotations

import unittest

import tests.support  # noqa: F401  (side effect: puts the repo on sys.path)

from shelflife.inference import (
    FRESH,
    MODERATE,
    SPOILED,
    KineticBaselinePredictor,
    PredictionContext,
    PredictorRegistry,
    TFLiteModelPredictor,
    chilling_factor,
    classify,
    effective_temperature,
    gas_factor,
    humidity_factor,
)

TOMATO = dict(
    food_key="tomato", food_name="Tomato", ref_shelf_life_hours=168.0,
    ref_temperature_c=20.0, q10=2.4, ideal_temp_min_c=10.0, ideal_temp_max_c=15.0,
    ideal_humidity_min=85.0, ideal_humidity_max=95.0,
)
APPLE = dict(
    food_key="apple", food_name="Apple", ref_shelf_life_hours=480.0,
    ref_temperature_c=20.0, q10=2.2, ideal_temp_min_c=0.0, ideal_temp_max_c=4.0,
    ideal_humidity_min=90.0, ideal_humidity_max=95.0,
)


class Q10Maths(unittest.TestCase):
    def setUp(self):
        self.model = KineticBaselinePredictor()

    def test_life_at_reference_temperature_equals_reference(self):
        total, _ = self.model.life_hours(
            PredictionContext(**TOMATO), 20.0, 90.0, None
        )
        self.assertAlmostEqual(total, 168.0, places=4)

    def test_ten_degrees_colder_multiplies_life_by_q10(self):
        ctx = PredictionContext(**APPLE)  # ideal_min = 0, so no chilling clamp at 10 C
        warm, _ = self.model.life_hours(ctx, 20.0, 92.0, None)
        cold, _ = self.model.life_hours(ctx, 10.0, 92.0, None)
        self.assertAlmostEqual(cold / warm, 2.2, places=4)

    def test_warmer_storage_always_shortens_life(self):
        ctx = PredictionContext(**TOMATO)
        lives = [self.model.life_hours(ctx, t, 90.0, None)[0] for t in (12, 18, 24, 30, 36)]
        self.assertEqual(lives, sorted(lives, reverse=True))

    def test_life_is_clamped_to_a_sane_range(self):
        ctx = PredictionContext(**APPLE)
        very_cold, _ = self.model.life_hours(ctx, -30.0, 92.0, None)
        self.assertLessEqual(very_cold, 24 * 45)
        very_hot, _ = self.model.life_hours(ctx, 90.0, 92.0, None)
        self.assertGreaterEqual(very_hot, 1.0)


class Factors(unittest.TestCase):
    def test_humidity_inside_the_band_is_neutral(self):
        self.assertEqual(humidity_factor(90, 85, 95), 1.0)

    def test_damp_penalised_harder_than_dry(self):
        """Equal deviations either side of the band are not equally costly."""
        damp = humidity_factor(80, 60, 70)   # 10 points above the band
        dry = humidity_factor(50, 60, 70)    # 10 points below it
        self.assertLess(damp, dry)

    def test_humidity_factor_never_collapses_to_zero(self):
        self.assertGreaterEqual(humidity_factor(100, 60, 65), 0.45)
        self.assertGreaterEqual(humidity_factor(0, 90, 95), 0.60)

    def test_missing_humidity_is_neutral(self):
        self.assertEqual(humidity_factor(None, 85, 95), 1.0)

    def test_chilling_only_applies_to_cold_sensitive_produce(self):
        self.assertEqual(chilling_factor(1.0, 0.0), 1.0)   # apple, tolerant
        self.assertLess(chilling_factor(1.0, 10.0), 1.0)   # tomato, sensitive

    def test_cooling_below_the_threshold_buys_no_further_slowdown(self):
        self.assertEqual(effective_temperature(2.0, 10.0), 10.0)
        self.assertEqual(effective_temperature(12.0, 10.0), 12.0)
        self.assertEqual(effective_temperature(2.0, 0.0), 2.0)

    def test_refrigerating_a_tomato_does_not_beat_ideal_storage(self):
        model = KineticBaselinePredictor()
        ctx = PredictionContext(**TOMATO)
        ideal, _ = model.life_hours(ctx, 12.0, 90.0, None)
        fridge, _ = model.life_hours(ctx, 3.0, 90.0, None)
        self.assertLess(fridge, ideal)

    def test_gas_factor_is_neutral_without_a_reading(self):
        self.assertEqual(gas_factor(None), 1.0)
        self.assertEqual(gas_factor(100), 1.0)
        self.assertLess(gas_factor(300), 0.7)


class Classification(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(classify(1.0), FRESH)
        self.assertEqual(classify(0.56), FRESH)
        self.assertEqual(classify(0.55), MODERATE)
        self.assertEqual(classify(0.09), MODERATE)
        self.assertEqual(classify(0.08), SPOILED)
        self.assertEqual(classify(0.0), SPOILED)


class HistoryIntegration(unittest.TestCase):
    def setUp(self):
        self.model = KineticBaselinePredictor()

    def test_warm_spell_consumes_more_life_than_steady_cold(self):
        warm_then_cold = self.model.predict(PredictionContext(
            **TOMATO, temperature_c=12.0, humidity_pct=90.0, hours_stored=72.0,
            history=[(48.0, 32.0, 90.0, None), (24.0, 12.0, 90.0, None)],
        ))
        always_cold = self.model.predict(PredictionContext(
            **TOMATO, temperature_c=12.0, humidity_pct=90.0, hours_stored=72.0,
            history=[(72.0, 12.0, 90.0, None)],
        ))
        self.assertLess(warm_then_cold.remaining_hours, always_cold.remaining_hours)
        self.assertGreater(
            warm_then_cold.factors["fraction_consumed"],
            always_cold.factors["fraction_consumed"],
        )

    def test_storage_time_before_the_first_reading_is_still_charged(self):
        """A 100-hour-old item with 10 hours of history must not look brand new."""
        partial = self.model.predict(PredictionContext(
            **TOMATO, temperature_c=20.0, humidity_pct=90.0, hours_stored=100.0,
            history=[(10.0, 20.0, 90.0, None)],
        ))
        self.assertGreater(partial.factors["fraction_consumed"], 0.5)

    def test_no_history_falls_back_to_the_current_condition(self):
        prediction = self.model.predict(PredictionContext(
            **TOMATO, temperature_c=20.0, humidity_pct=90.0, hours_stored=84.0,
        ))
        self.assertAlmostEqual(prediction.factors["fraction_consumed"], 0.5, places=2)

    def test_remaining_never_goes_negative(self):
        prediction = self.model.predict(PredictionContext(
            **TOMATO, temperature_c=35.0, humidity_pct=95.0, hours_stored=5000.0,
        ))
        self.assertEqual(prediction.remaining_hours, 0.0)
        self.assertEqual(prediction.freshness_class, SPOILED)


class Honesty(unittest.TestCase):
    """Guards on the project's core constraint: never overclaim."""

    def test_baseline_reports_no_calibrated_confidence(self):
        prediction = KineticBaselinePredictor().predict(
            PredictionContext(**TOMATO, temperature_c=22.0, humidity_pct=80.0)
        )
        self.assertIsNone(prediction.freshness_confidence)

    def test_baseline_never_claims_to_have_used_the_image(self):
        prediction = KineticBaselinePredictor().predict(
            PredictionContext(**TOMATO, temperature_c=22.0, humidity_pct=80.0,
                              image_path="some_photo.jpg")
        )
        self.assertFalse(prediction.image_used)

    def test_registry_reports_baseline_when_no_model_is_configured(self):
        status = PredictorRegistry().status()
        self.assertEqual(status["active_kind"], "heuristic-baseline")
        self.assertFalse(status["uses_image"])
        self.assertFalse(status["supports_calibrated_confidence"])
        self.assertFalse(status["trained_model_configured"])

    def test_missing_model_file_degrades_instead_of_crashing(self):
        registry = PredictorRegistry("/nonexistent/model.tflite")
        self.assertEqual(registry.active.kind, "heuristic-baseline")
        self.assertTrue(registry.status()["trained_model_configured"])
        self.assertFalse(registry.status()["trained_model_available"])
        self.assertIn("No model file", registry.status()["trained_model_error"])

    def test_unloadable_model_still_produces_a_prediction(self):
        registry = PredictorRegistry("/nonexistent/model.tflite")
        prediction = registry.predict(
            PredictionContext(**TOMATO, temperature_c=22.0, humidity_pct=80.0)
        )
        self.assertEqual(prediction.model_kind, "heuristic-baseline")

    def test_tflite_adapter_reports_why_it_is_unavailable(self):
        predictor = TFLiteModelPredictor("/nope/model.tflite")
        self.assertFalse(predictor.available)
        self.assertIsNotNone(predictor.load_error)


class Serialisation(unittest.TestCase):
    def test_days_and_hours_split(self):
        prediction = KineticBaselinePredictor().predict(
            PredictionContext(**APPLE, temperature_c=20.0, humidity_pct=92.0)
        )
        payload = prediction.to_dict()
        self.assertEqual(payload["days"] * 24 + payload["hours"], round(payload["remaining_hours"]))
        self.assertIn(payload["status_color"], {"ok", "warn", "crit"})


if __name__ == "__main__":
    unittest.main()
