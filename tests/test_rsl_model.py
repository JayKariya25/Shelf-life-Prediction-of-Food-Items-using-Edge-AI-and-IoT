"""The trained Remaining-Shelf-Life model, in the website.

Three layers:

* always run - dataset parsing, metrics, graceful degradation without a runtime;
* fixture   - TFLite models with the real exports' exact contracts, bundled with the
  real ``pi_features.py`` and ``image_io.py`` (needs TensorFlow). Both shapes are
  covered: the image-only ensemble that is served, and the older image+sensor
  fusion graph, which the loader still supports;
* real      - the actual bundle in ``ml/rsl_multimodal`` (needs TensorFlow, and the
  gitignored ``data_images/`` for the accuracy checks).

Run everything with the model environment:
    .venv-model/bin/python -m unittest tests.test_rsl_model
"""

from __future__ import annotations

import io
import json
import math
import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from tests.support import AppTestCase

from shelflife.dataset import HeldOutSplit, regression_metrics
from shelflife.db import iso_now, query_one, utcnow
from shelflife.inference import PredictionContext, PredictorRegistry
from shelflife.rsl_model import ModelUnavailable, Reading, RSLModel

REPO = Path(__file__).resolve().parent.parent
REAL_BUNDLE = REPO / "ml" / "rsl_multimodal"

try:
    import numpy  # noqa: F401
    import sklearn  # noqa: F401
    import tensorflow  # noqa: F401
    from PIL import Image  # noqa: F401

    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False

needs_runtime = unittest.skipUnless(HAS_RUNTIME, "TensorFlow runtime not installed (use .venv-model)")
SERVED = "models/image_ensemble_blocked_full_float16.tflite"
FUSION = "models/fusion_blocked_full_float16.tflite"

needs_real_bundle = unittest.skipUnless(
    HAS_RUNTIME and (REAL_BUNDLE / SERVED).is_file(),
    "real model bundle or runtime not available",
)
needs_real_fusion = unittest.skipUnless(
    HAS_RUNTIME and (REAL_BUNDLE / FUSION).is_file(),
    "the fusion export is not in the bundle",
)
needs_real_images = unittest.skipUnless(
    HAS_RUNTIME and (REAL_BUNDLE / "data_images").is_dir(),
    "real training images not present (they are gitignored)",
)

TOMATO = dict(
    food_key="tomato", food_name="Tomato", ref_shelf_life_hours=168.0,
    ref_temperature_c=20.0, q10=2.4, ideal_temp_min_c=10.0, ideal_temp_max_c=15.0,
    ideal_humidity_min=85.0, ideal_humidity_max=95.0,
)
# The trained model was fitted on a single 11-day banana run, so banana is the only
# produce it may serve. Every trained-model case below uses it.
BANANA = dict(
    food_key="banana", food_name="Banana", ref_shelf_life_hours=120.0,
    ref_temperature_c=20.0, q10=2.6, ideal_temp_min_c=13.0, ideal_temp_max_c=16.0,
    ideal_humidity_min=85.0, ideal_humidity_max=95.0,
)
IN_RANGE = Reading(28.5, 58.0, 185.0, 0.0)

_FIXTURE: dict[str, Path] = {}


def fixture_bundle() -> Path:
    """Build the fixture bundle once per test run."""
    if "path" not in _FIXTURE:
        from tools.make_rsl_fixture import main as build

        root = Path(tempfile.mkdtemp(prefix="rsl-fixture-"))
        _FIXTURE["path"] = build(root)
    return _FIXTURE["path"]


def tearDownModule():
    if "path" in _FIXTURE:
        shutil.rmtree(_FIXTURE["path"], ignore_errors=True)


def jpeg_bytes(colour=(120, 160, 90), size=(224, 224)) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, "JPEG", quality=95)
    return buffer.getvalue()


def png_bytes(colour=(120, 160, 90), size=(224, 224)) -> bytes:
    """A PNG upload makes the image loader report that it re-encoded the frame."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, "PNG")
    return buffer.getvalue()


# =============================================================================
# always run
# =============================================================================
class Metrics(unittest.TestCase):
    def test_known_values(self):
        result = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual((result["mae"], result["rmse"], result["r2"]), (0.0, 0.0, 1.0))
        self.assertEqual(result["within_half_day"], 1.0)

    def test_bias_sign_shows_over_prediction(self):
        self.assertGreater(regression_metrics([1.0, 2.0], [2.0, 3.0])["bias"], 0)
        self.assertLess(regression_metrics([2.0, 3.0], [1.0, 2.0])["bias"], 0)

    def test_tolerance_bands(self):
        result = regression_metrics([0.0, 0.0, 0.0, 0.0], [0.2, 0.7, 1.5, 0.0])
        self.assertEqual(result["within_half_day"], 0.5)
        self.assertEqual(result["within_one_day"], 0.75)

    def test_empty_input_is_safe(self):
        self.assertEqual(regression_metrics([], [])["n"], 0)

    def test_r2_is_none_without_variance(self):
        self.assertIsNone(regression_metrics([5.0, 5.0], [4.0, 6.0])["r2"])


class SplitParsing(unittest.TestCase):
    HEADER = "image_file,Timestamp,Day,RemainingShelfLife,split_blocked,split_random,Temperature,Humidity,Gas,Extra\n"

    def bundle(self, body: str) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "artifacts").mkdir()
        (tmp / "data_images").mkdir()
        (tmp / "artifacts" / "master_dataset.csv").write_text(body, encoding="utf-8")
        return tmp

    def test_missing_file_reports_the_path(self):
        split = HeldOutSplit("/nowhere")
        self.assertFalse(split.available)
        self.assertIn("/nowhere", split.error)

    def test_missing_columns_are_named(self):
        split = HeldOutSplit(self.bundle("image_file,Day\na.jpg,1\n"))
        self.assertIn("missing columns", split.error)

    def test_only_blocked_test_rows_are_kept(self):
        split = HeldOutSplit(self.bundle(
            self.HEADER
            + "a.jpg,t,1,10,train,test,28,58,185,1\n"
            + "b.jpg,t,2,9,test,train,28,58,185,1\n"
            + "c.jpg,t,3,8,dropped,test,28,58,185,1\n"
        ))
        self.assertEqual(len(split), 1)
        self.assertEqual(split.rows[0]["image_file"], "b.jpg")
        self.assertEqual(split.counts, {"train": 1, "test": 1, "dropped": 1})

    def test_feature_columns_exclude_metadata(self):
        split = HeldOutSplit(self.bundle(self.HEADER + "b.jpg,t,2,9,test,train,28,58,185,1\n"))
        self.assertEqual(split.feature_columns, ["Temperature", "Humidity", "Gas", "Extra"])

    def test_unparseable_rows_are_skipped(self):
        split = HeldOutSplit(self.bundle(
            self.HEADER
            + "a.jpg,t,1,10,test,test,28,58,185,1\n"
            + "b.jpg,t,2,9,test,test,warm,58,185,1\n"
        ))
        self.assertEqual(len(split), 1)

    def test_vector_names_a_missing_feature(self):
        split = HeldOutSplit(self.bundle(self.HEADER + "b.jpg,t,2,9,test,train,28,58,185,1\n"))
        with self.assertRaises(KeyError):
            split.vector(split.rows[0], ["Temperature", "Not_A_Column"])

    def test_missing_image_is_reported_not_fatal(self):
        split = HeldOutSplit(self.bundle(self.HEADER + "gone.jpg,t,2,9,test,train,28,58,185,1\n"))
        self.assertIsNone(split.rows[0]["image_path"])
        self.assertEqual(split.summary()["rows_missing_images"], 1)

    def test_out_of_range_index(self):
        split = HeldOutSplit(self.bundle(self.HEADER + "b.jpg,t,2,9,test,train,28,58,185,1\n"))
        self.assertIsNone(split.get(1))
        self.assertIsNone(split.get(-1))


class GracefulDegradation(unittest.TestCase):
    def test_missing_bundle_lists_every_file(self):
        model = RSLModel("/nowhere")
        self.assertFalse(model.available)
        for label in ("model", "config", "image loader"):
            self.assertIn(label, model.load_error)

    def test_the_sensor_half_is_not_demanded_before_the_graph_is_known(self):
        """The scaler and feature builder are only needed by a sensor-taking export."""
        error = RSLModel("/nowhere").load_error
        self.assertNotIn("scaler", error)
        self.assertNotIn("feature builder", error)

    def test_status_is_serialisable_when_unavailable(self):
        json.dumps(RSLModel("/nowhere").status())

    def test_predicting_without_a_model_raises(self):
        with self.assertRaises(ModelUnavailable):
            RSLModel("/nowhere").predict(None, [IN_RANGE])
        with self.assertRaises(ModelUnavailable):
            RSLModel("/nowhere").predict(None)


# =============================================================================
# fixture: the real contract, the real feature and image code
# =============================================================================
class BundleCopy(unittest.TestCase):
    """Helper: a writable copy of the fixture bundle, to break on purpose."""

    def copy_bundle(self) -> Path:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        target = tmp / "bundle"
        shutil.copytree(fixture_bundle(), target)
        return target


@needs_runtime
class FixtureContract(BundleCopy):
    """The served export: one image input and nothing else."""

    def setUp(self):
        self.model = RSLModel(fixture_bundle())

    def test_loads(self):
        self.assertTrue(self.model.available, self.model.load_error)
        self.assertTrue(self.model.image_only)
        self.assertEqual(self.model.max_days, 11.0)

    def test_input_contract_is_recorded(self):
        inputs = self.model.input_details
        self.assertEqual(inputs["image"]["shape"], [1, 224, 224, 3])
        self.assertNotIn("sensor", inputs)
        self.assertEqual(inputs["output"]["shape"], [1, 1])

    def test_the_sensor_half_of_the_bundle_is_not_touched(self):
        self.assertEqual(self.model.feature_names, [])
        self.assertNotIn("scaler", self.model.required_files)
        self.assertNotIn("feature builder", self.model.required_files)
        self.assertFalse(self.model.requires_gas)

    def test_loads_without_a_scaler_or_feature_builder_at_all(self):
        bundle = self.copy_bundle()
        (bundle / "artifacts" / "sensor_scaler.pkl").unlink()
        (bundle / "scripts" / "pi_features.py").unlink()
        model = RSLModel(bundle)
        self.assertTrue(model.available, model.load_error)

    def test_published_metrics_are_read(self):
        self.assertAlmostEqual(self.model.metrics["served_variant"]["MAE"], 0.1468)
        self.assertAlmostEqual(self.model.metrics["within_session"]["MAE"], 0.1468)
        self.assertEqual(self.model.metrics["ensemble_heads"], 3)

    def test_the_band_is_the_served_models_own_error(self):
        """Not the best figure in the bundle - the one belonging to what is running."""
        lodo = self.model.metrics["leave_one_day_out_mae"]
        self.assertAlmostEqual(lodo["fusion"], 1.25)
        self.assertAlmostEqual(lodo["image_ensemble"], 0.80)
        self.assertAlmostEqual(self.model.new_batch_mae_days, 0.80)

    def test_config_without_outlier_bounds_is_refused(self):
        bundle = self.copy_bundle()
        path = bundle / "artifacts" / "preprocessing_config.json"
        config = json.loads(path.read_text())
        del config["outlier_bounds"]["Gas"]
        path.write_text(json.dumps(config))
        model = RSLModel(bundle)
        self.assertFalse(model.available)
        self.assertIn("Gas", model.load_error)

    def test_a_non_model_file_is_refused(self):
        bundle = self.copy_bundle()
        (bundle / SERVED).write_bytes(b"not a model")
        model = RSLModel(bundle)
        self.assertFalse(model.available)
        self.assertIsNotNone(model.load_error)

    def test_missing_results_do_not_block_loading(self):
        bundle = self.copy_bundle()
        shutil.rmtree(bundle / "results")
        model = RSLModel(bundle)
        self.assertTrue(model.available, model.load_error)
        self.assertIsNone(model.new_batch_mae_days)


@needs_runtime
class FusionContract(BundleCopy):
    """The older two-input export. Still loadable, and still held to its contract."""

    def setUp(self):
        self.model = RSLModel(fixture_bundle(), FUSION)

    def test_loads(self):
        self.assertTrue(self.model.available, self.model.load_error)
        self.assertFalse(self.model.image_only)
        self.assertTrue(self.model.requires_gas)
        self.assertEqual(len(self.model.feature_names), 27)

    def test_input_contract_is_recorded(self):
        inputs = self.model.input_details
        self.assertEqual(inputs["image"]["shape"], [1, 224, 224, 3])
        self.assertEqual(inputs["sensor"]["shape"], [1, 27])
        self.assertEqual(inputs["output"]["shape"], [1, 1])

    def test_its_own_published_metrics_are_read(self):
        self.assertAlmostEqual(self.model.metrics["served_variant"]["MAE"], 0.2636)
        self.assertAlmostEqual(self.model.new_batch_mae_days, 1.25)
        self.assertAlmostEqual(self.model.metrics["within_session"]["MAE"], 0.2662)

    def test_scaler_feature_mismatch_is_refused(self):
        import pickle

        bundle = self.copy_bundle()
        path = bundle / "artifacts" / "sensor_scaler.pkl"
        data = pickle.loads(path.read_bytes())
        data["features"]["full"] = data["features"]["full"][:-1]
        path.write_bytes(pickle.dumps(data))
        model = RSLModel(bundle, FUSION)
        self.assertFalse(model.available)
        self.assertIn("features", model.load_error)

    def test_a_missing_scaler_is_named(self):
        bundle = self.copy_bundle()
        (bundle / "artifacts" / "sensor_scaler.pkl").unlink()
        model = RSLModel(bundle, FUSION)
        self.assertFalse(model.available)
        self.assertIn("scaler", model.load_error)


@needs_runtime
class FixturePrediction(unittest.TestCase):
    """The served export reads the photograph and nothing else."""

    def setUp(self):
        self.model = RSLModel(fixture_bundle())
        self.image = self.model.load_image_bytes(jpeg_bytes())

    def test_prediction_is_finite_and_in_range(self):
        result = self.model.predict(self.image)
        self.assertTrue(math.isfinite(result.raw_days))
        self.assertGreaterEqual(result.days, 0.0)
        self.assertLessEqual(result.days, 11.0)

    def test_deterministic(self):
        self.assertEqual(self.model.predict(self.image).raw_days,
                         self.model.predict(self.image).raw_days)

    def test_no_readings_are_needed(self):
        result = self.model.predict(self.image)
        self.assertEqual(result.reading_number, 0)
        self.assertEqual(result.warnings, [])

    def test_readings_do_not_change_the_estimate(self):
        """If they did, the page would be crediting the sensors with something."""
        alone = self.model.predict(self.image).raw_days
        with_history = self.model.predict(self.image, [
            Reading(27.6, 54.0, 175.0, 0.0),
            Reading(45.0, 58.0, 185.0, 2 / 60),
        ]).raw_days
        self.assertEqual(alone, with_history)

    def test_no_cold_start_or_envelope_caveats_are_invented(self):
        result = self.model.predict(self.image, [Reading(45.0, 58.0, 185.0, 0.0)])
        self.assertEqual(result.out_of_envelope, {})
        self.assertEqual(result.adjustments, {})
        self.assertFalse(any("First reading" in w for w in result.warnings))
        self.assertFalse(any("extrapolation" in w for w in result.warnings))

    def test_a_missing_channel_is_not_an_error_here(self):
        """Nothing reads the channels, so an absent one cannot break the forward pass."""
        result = self.model.predict(self.image, [Reading(28.5, 58.0, None, 0.0)])
        self.assertTrue(math.isfinite(result.raw_days))


@needs_runtime
class FusionPrediction(unittest.TestCase):
    """The two-input export: the sensor history is replayed into 27 features."""

    def setUp(self):
        self.model = RSLModel(fixture_bundle(), FUSION)
        self.image = self.model.load_image_bytes(jpeg_bytes())

    def test_prediction_is_finite_and_in_range(self):
        result = self.model.predict(self.image, [IN_RANGE])
        self.assertTrue(math.isfinite(result.raw_days))
        self.assertGreaterEqual(result.days, 0.0)
        self.assertLessEqual(result.days, 11.0)

    def test_deterministic(self):
        first = self.model.predict(self.image, [IN_RANGE]).raw_days
        second = self.model.predict(self.image, [IN_RANGE]).raw_days
        self.assertEqual(first, second)

    def test_cold_start_is_flagged(self):
        result = self.model.predict(self.image, [IN_RANGE])
        self.assertEqual(result.reading_number, 1)
        self.assertTrue(any("First reading" in w for w in result.warnings))

    def test_history_changes_the_features(self):
        cold = self.model.predict(self.image, [Reading(28.5, 58.0, 185.0, 0.0)])
        warm = self.model.predict(self.image, [
            Reading(27.6, 54.0, 175.0, 0.0),
            Reading(28.0, 56.0, 180.0, 2 / 60),
            Reading(28.5, 58.0, 185.0, 2 / 60),
        ])
        self.assertEqual(warm.reading_number, 3)
        self.assertNotEqual(cold.raw_days, warm.raw_days)
        self.assertFalse(any("First reading" in w for w in warm.warnings))

    def test_out_of_envelope_reading_is_flagged(self):
        result = self.model.predict(self.image, [Reading(45.0, 58.0, 185.0, 0.0)])
        self.assertEqual(result.out_of_envelope, {"Temperature": 45.0})
        self.assertTrue(any("extrapolation" in w for w in result.warnings))

    def test_cold_start_outlier_is_clipped_and_reported(self):
        result = self.model.predict(self.image, [Reading(45.0, 58.0, 185.0, 0.0)])
        change = result.adjustments["Temperature"]
        self.assertEqual(change["how"], "clipped")
        self.assertAlmostEqual(change["used"], 29.335)

    def test_later_outlier_is_forward_filled(self):
        result = self.model.predict(self.image, [IN_RANGE, Reading(45.0, 58.0, 185.0, 2 / 60)])
        change = result.adjustments["Temperature"]
        self.assertEqual(change["how"], "forward_filled")
        self.assertAlmostEqual(change["used"], 28.5)

    def test_missing_channel_is_refused(self):
        with self.assertRaises(ModelUnavailable):
            self.model.predict(self.image, [Reading(28.5, 58.0, None, 0.0)])

    def test_empty_history_is_refused(self):
        with self.assertRaises(ModelUnavailable):
            self.model.predict(self.image, [])

    def test_continuing_a_session_advances_the_builder(self):
        builder = self.model.new_builder()
        self.model.predict(self.image, [IN_RANGE], builder=builder)
        self.model.predict(self.image, [Reading(28.6, 58.2, 186.0, 2 / 60)], builder=builder)
        self.assertEqual(len(builder.smoothed["Temperature"]), 2)


@needs_runtime
class FixtureImages(unittest.TestCase):
    def setUp(self):
        self.model = RSLModel(fixture_bundle())

    def test_training_form_jpeg_is_decoded_directly(self):
        loaded = self.model.load_image_bytes(jpeg_bytes())
        self.assertEqual(loaded.pixels.shape, (1, 224, 224, 3))
        self.assertFalse(loaded.resized)
        self.assertTrue(loaded.exact)

    def test_large_frame_is_resized_like_training(self):
        loaded = self.model.load_image_bytes(jpeg_bytes(size=(2592, 1944)))
        self.assertTrue(loaded.resized)
        self.assertEqual(loaded.source_size, (2592, 1944))
        self.assertEqual(loaded.pixels.shape, (1, 224, 224, 3))

    def test_png_is_reencoded_and_says_so(self):
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", (224, 224), (10, 200, 30)).save(buffer, "PNG")
        loaded = self.model.load_image_bytes(buffer.getvalue())
        self.assertTrue(any("re-encoded" in note for note in loaded.notes))

    def test_corrupt_upload_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.model.load_image_bytes(b"definitely not an image")

    def test_empty_upload_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.model.load_image_bytes(b"")

    def test_simplejpeg_matches_tensorflow_when_available(self):
        """The Raspberry Pi decoder must agree with training bit for bit."""
        try:
            import simplejpeg
        except ImportError:
            self.skipTest("simplejpeg not installed")
        import numpy as np
        import tensorflow as tf

        data = jpeg_bytes(colour=(200, 40, 90), size=(224, 224))
        self.assertTrue(np.array_equal(
            tf.io.decode_jpeg(data, channels=3).numpy(),
            simplejpeg.decode_jpeg(data, colorspace="RGB", fastdct=True, fastupsample=False),
        ))


class RegistryContext(unittest.TestCase):
    """Shared set-up for the two export shapes. Holds no tests of its own."""

    model_file = SERVED

    def setUp(self):
        self.registry = PredictorRegistry(str(fixture_bundle()), self.model_file,
                                          max_photo_age_hours=6.0)
        self.photo = str(fixture_bundle() / "data_images" / "fixture_000.jpg")

    def context(self, **overrides):
        base = dict(
            **BANANA, temperature_c=28.5, humidity_pct=58.0, gas_ppm=185.0,
            hours_stored=24.0, image_full_path=self.photo, image_age_hours=0.1,
            rsl_history=[Reading(28.4, 57.8, 184.0, 0.0), Reading(28.5, 58.0, 185.0, 2 / 60)],
        )
        base.update(overrides)
        return PredictionContext(**base)


@needs_runtime
class FixtureRegistry(RegistryContext):
    """The served export: a recent photograph is the only thing it asks for."""

    def test_trained_model_serves_a_complete_request(self):
        prediction = self.registry.predict(self.context())
        self.assertEqual(prediction.model_kind, "trained-model")
        self.assertTrue(prediction.image_used)
        self.assertIsNone(prediction.fallback_reason)
        self.assertIsNone(prediction.freshness_confidence)
        self.assertEqual(prediction.model_inputs, "photo only")

    def test_interval_is_the_new_batch_error(self):
        prediction = self.registry.predict(self.context())
        hours = prediction.remaining_hours
        self.assertAlmostEqual(prediction.remaining_hours_high - hours,
                               min(0.80 * 24, 11 * 24 - hours), places=3)
        self.assertIn("never saw in training", prediction.rationale)

    def test_freshness_class_follows_days_over_eleven(self):
        from shelflife.inference import classify

        prediction = self.registry.predict(self.context())
        self.assertEqual(prediction.freshness_class, classify(prediction.remaining_hours / 24 / 11))

    def test_no_photo_falls_back(self):
        prediction = self.registry.predict(self.context(image_full_path=None))
        self.assertEqual(prediction.model_kind, "heuristic-baseline")
        self.assertIn("no photograph", prediction.fallback_reason)

    def test_stale_photo_falls_back(self):
        prediction = self.registry.predict(self.context(image_age_hours=30.0))
        self.assertEqual(prediction.model_kind, "heuristic-baseline")
        self.assertIn("30.0 h older", prediction.fallback_reason)

    def test_no_gas_is_no_longer_a_blocker(self):
        """The whole point of serving the image-only export: no gas sensor needed."""
        prediction = self.registry.predict(self.context(rsl_history=[], gas_ppm=None))
        self.assertEqual(prediction.model_kind, "trained-model")
        self.assertIsNone(prediction.fallback_reason)

    def test_a_reading_outside_the_trained_range_is_no_longer_a_blocker(self):
        prediction = self.registry.predict(self.context(
            rsl_history=[Reading(22.0, 58.0, 185.0, 0.0)]))
        self.assertEqual(prediction.model_kind, "trained-model")

    def test_another_food_falls_back_however_good_the_photo(self):
        """It was trained on bananas. A tomato is not a banana, photo or no photo."""
        prediction = self.registry.predict(self.context(**TOMATO))
        self.assertEqual(prediction.model_kind, "heuristic-baseline")
        self.assertIn("fitted on banana only", prediction.fallback_reason)
        self.assertIn("tomato", prediction.fallback_reason)

    def test_the_photograph_is_the_only_precondition(self):
        prediction = self.registry.predict(self.context(
            image_full_path=None, rsl_history=[], gas_ppm=None))
        self.assertIn("no photograph", prediction.fallback_reason)
        self.assertNotIn("no gas reading", prediction.fallback_reason)

    def test_the_rationale_says_the_readings_are_not_an_input(self):
        rationale = self.registry.predict(self.context()).rationale
        self.assertIn("photograph alone", rationale)
        self.assertIn("not an input", rationale)

    def test_status_reports_the_trained_model(self):
        status = self.registry.status()
        self.assertTrue(status["trained_model_available"])
        self.assertEqual(status["active_kind"], "trained-model")
        self.assertFalse(status["requires_gas"])
        self.assertTrue(status["rsl"]["image_only"])
        self.assertEqual(status["rsl"]["feature_count"], 0)


@needs_runtime
class FusionRegistry(RegistryContext):
    """The fusion export keeps its stricter gating, and its own error band."""

    model_file = FUSION

    def test_it_is_the_two_input_graph(self):
        self.assertFalse(self.registry.rsl.image_only)
        self.assertTrue(self.registry.status()["requires_gas"])

    def test_trained_model_serves_a_complete_request(self):
        prediction = self.registry.predict(self.context())
        self.assertEqual(prediction.model_kind, "trained-model")
        self.assertEqual(prediction.model_inputs, "photo + sensor history")

    def test_no_gas_falls_back(self):
        prediction = self.registry.predict(self.context(rsl_history=[], gas_ppm=None))
        self.assertEqual(prediction.model_kind, "heuristic-baseline")
        self.assertIn("no gas reading", prediction.fallback_reason)

    def test_out_of_envelope_falls_back(self):
        prediction = self.registry.predict(self.context(
            rsl_history=[Reading(22.0, 58.0, 185.0, 0.0)]))
        self.assertEqual(prediction.model_kind, "heuristic-baseline")
        self.assertIn("outside the range the model was trained on", prediction.fallback_reason)
        self.assertIn("Temperature 22", prediction.fallback_reason)

    def test_every_blocker_is_reported_at_once(self):
        prediction = self.registry.predict(self.context(
            image_full_path=None, rsl_history=[], gas_ppm=None))
        self.assertIn("no photograph", prediction.fallback_reason)
        self.assertIn("no gas reading", prediction.fallback_reason)

    def test_its_band_is_its_own_leave_one_day_out_error(self):
        prediction = self.registry.predict(self.context())
        hours = prediction.remaining_hours
        self.assertAlmostEqual(prediction.remaining_hours_high - hours,
                               min(1.25 * 24, 11 * 24 - hours), places=3)


# =============================================================================
# the website, end to end, with the fixture bundle
# =============================================================================
@needs_runtime
class MonitoringWithTheModel(AppTestCase):
    sensor_source = "auto"

    @classmethod
    def setUpClass(cls):
        cls.rsl_bundle = str(fixture_bundle())

    def setUp(self):
        super().setUp()
        self.register_and_login()
        self.token = self.make_device()
        self.auth = {"Authorization": f"Bearer {self.token}"}
        stored = (utcnow() - timedelta(hours=2)).replace(microsecond=0).isoformat()
        # Banana: the only produce the trained model was fitted on, so these tests
        # exercise the trained path rather than the food-type fallback.
        self.item = self.create_item(label="Monitored bananas", food_key="banana",
                                     stored_at=stored)

    def post_readings(self, count=6, spacing_seconds=120, **values):
        start = utcnow() - timedelta(seconds=spacing_seconds * count)
        for i in range(count):
            payload = {"temperature_c": values.get("temperature", 28.4 + 0.01 * i),
                       "humidity_pct": values.get("humidity", 58.0),
                       "gas_ppm": values.get("gas", 184.0 + i),
                       "recorded_at": (start + timedelta(seconds=spacing_seconds * (i + 1))).isoformat()}
            response = self.client.post("/api/device/v1/readings", json=payload, headers=self.auth)
            self.assertEqual(response.status_code, 201, response.get_data(as_text=True))

    def capture(self, captured_at=None, size=(640, 480), image_format="JPEG"):
        blob = (jpeg_bytes(size=size) if image_format == "JPEG"
                else png_bytes(size=size))
        name = "frame.jpg" if image_format == "JPEG" else "frame.png"
        data = {"image": (io.BytesIO(blob), name),
                "item_id": str(self.item["id"])}
        if captured_at:
            data["captured_at"] = captured_at
        response = self.client.post("/api/device/v1/captures", data=data,
                                    content_type="multipart/form-data", headers=self.auth)
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        return response.get_json()["data"]

    def predict(self):
        response = self.post_json(f"/api/v1/items/{self.item['id']}/predict", {})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.get_json()["data"]

    def test_without_a_photo_the_item_explains_the_fallback(self):
        self.post_readings()
        item = self.predict()["item"]
        self.assertEqual(item["model_kind"], "heuristic-baseline")
        self.assertIn("no photograph", item["fallback_reason"])

    def test_readings_plus_a_fresh_capture_use_the_trained_model(self):
        self.post_readings()
        self.capture()
        data = self.predict()
        self.assertEqual(data["prediction"]["model_kind"], "trained-model")
        self.assertEqual(data["item"]["model_kind"], "trained-model")
        self.assertIsNone(data["item"]["fallback_reason"])
        self.assertEqual(data["prediction"]["factors"]["readings_in_session"], 6)

    def test_capture_links_the_item_to_the_device(self):
        self.capture()
        with self.app.app_context():
            row = query_one("SELECT device_id, image_captured_at FROM items WHERE id = ?",
                            (self.item["id"],))
        self.assertIsNotNone(row["device_id"])
        self.assertIsNotNone(row["image_captured_at"])

    def test_an_old_capture_falls_back(self):
        self.post_readings()
        self.capture(captured_at=(utcnow() - timedelta(hours=20)).isoformat())
        item = self.predict()["item"]
        self.assertEqual(item["model_kind"], "heuristic-baseline")
        self.assertIn("photograph", item["fallback_reason"])

    def test_readings_without_gas_still_use_the_trained_model(self):
        """No gas sensor is wired up in this project; the served model does not need one."""
        start = utcnow() - timedelta(minutes=10)
        for i in range(3):
            self.client.post("/api/device/v1/readings", headers=self.auth, json={
                "temperature_c": 28.4, "humidity_pct": 58.0,
                "recorded_at": (start + timedelta(minutes=2 * i)).isoformat()})
        self.capture()
        item = self.predict()["item"]
        self.assertEqual(item["model_kind"], "trained-model")
        self.assertIsNone(item["fallback_reason"])

    def test_a_warm_room_no_longer_forces_a_fallback(self):
        self.post_readings(temperature=22.0)
        self.capture()
        item = self.predict()["item"]
        self.assertEqual(item["model_kind"], "trained-model")

    def test_with_no_readings_at_all_a_photo_is_enough(self):
        self.capture()
        item = self.predict()["item"]
        self.assertEqual(item["model_kind"], "trained-model")

    def test_a_tomato_item_is_served_by_the_baseline_and_says_so(self):
        """The site lets users track tomatoes; the model has only ever seen bananas."""
        stored = (utcnow() - timedelta(hours=2)).replace(microsecond=0).isoformat()
        tomato = self.create_item(label="Monitored tomatoes", food_key="tomato",
                                  stored_at=stored)
        self.client.post("/api/device/v1/captures", headers=self.auth,
                         content_type="multipart/form-data",
                         data={"image": (io.BytesIO(jpeg_bytes()), "f.jpg"),
                               "item_id": str(tomato["id"])})
        response = self.post_json(f"/api/v1/items/{tomato['id']}/predict", {})
        item = response.get_json()["data"]["item"]
        self.assertEqual(item["model_kind"], "heuristic-baseline")
        self.assertIn("fitted on banana only", item["fallback_reason"])
        page = self.client.get(f"/items/{tomato['id']}").get_data(as_text=True)
        self.assertIn("fitted on banana only", page)

    def test_readings_before_the_item_was_stored_are_ignored(self):
        with self.app.app_context():
            from shelflife import services

            raw = services.get_item(1, self.item["id"])
            old = (utcnow() - timedelta(hours=5)).isoformat()
            self.client.post("/api/device/v1/readings", headers=self.auth, json={
                "temperature_c": 28.4, "humidity_pct": 58.0, "gas_ppm": 184.0, "recorded_at": old})
            history, _ = services.rsl_history_for_item(
                1, raw, utcnow() - timedelta(hours=2), utcnow())
        self.assertEqual(history, [])

    def test_a_fast_stream_is_thinned_to_the_training_cadence(self):
        self.post_readings(count=12, spacing_seconds=30)
        with self.app.app_context():
            from shelflife import services

            raw = services.get_item(1, self.item["id"])
            history, _ = services.rsl_history_for_item(
                1, raw, utcnow() - timedelta(hours=2), utcnow())
        self.assertLessEqual(len(history), 4)
        self.assertTrue(all(r.hours_since_previous * 3600 >= 90 for r in history[1:]))

    def test_model_notes_are_stored_and_shown(self):
        """A caveat the image loader raises has to reach the page, not be swallowed."""
        self.post_readings(count=1)
        self.capture(image_format="PNG")
        item = self.predict()["item"]
        self.assertTrue(any("re-encoded" in note for note in item["model_notes"]),
                        item["model_notes"])
        page = self.client.get(f"/items/{self.item['id']}").get_data(as_text=True)
        self.assertIn("re-encoded", page)

    def test_item_page_names_the_trained_model(self):
        self.post_readings()
        self.capture()
        self.predict()
        page = self.client.get(f"/items/{self.item['id']}").get_data(as_text=True)
        self.assertIn("Image-ensemble RSL model", page)
        # The line under the estimate says what this run actually read. Earlier baseline
        # runs are also on the page, in the history JSON, so match the rendered line.
        self.assertIn("&middot; photo only", page)
        self.assertNotIn("&middot; photo + sensor history", page)

    def test_capture_rejects_a_future_timestamp(self):
        response = self.client.post(
            "/api/device/v1/captures",
            data={"image": (io.BytesIO(jpeg_bytes()), "f.jpg"), "item_id": str(self.item["id"]),
                  "captured_at": (utcnow() + timedelta(days=1)).isoformat()},
            content_type="multipart/form-data", headers=self.auth)
        self.assertEqual(response.status_code, 422)

    def test_capture_for_someone_elses_item_is_refused(self):
        response = self.client.post(
            "/api/device/v1/captures",
            data={"image": (io.BytesIO(jpeg_bytes()), "f.jpg"), "item_id": "99999"},
            content_type="multipart/form-data", headers=self.auth)
        self.assertEqual(response.status_code, 404)

    def test_devices_are_asked_to_report_every_two_minutes(self):
        data = self.client.get("/api/device/v1/config", headers=self.auth).get_json()["data"]
        self.assertEqual(data["report_interval_seconds"], 120)


@needs_runtime
class DeveloperConsoleWithTheModel(AppTestCase):
    @classmethod
    def setUpClass(cls):
        cls.rsl_bundle = str(fixture_bundle())

    def setUp(self):
        super().setUp()
        from shelflife.blueprints.admin import clear_playground
        from shelflife.dataset import clear_cache

        clear_cache()
        clear_playground()
        self.addCleanup(clear_cache)
        self.addCleanup(clear_playground)
        self.register_and_login_admin()

    def playground(self, temperature="28.5", session="s1", reset=False, image=None, **extra):
        data = {"image": (io.BytesIO(image or jpeg_bytes()), "x.jpg"),
                "temperature": temperature, "humidity": "58", "gas": "185", "session": session}
        if reset:
            data["reset"] = "1"
        data.update(extra)
        return self.client.post("/admin/api/predict", data=data,
                                content_type="multipart/form-data",
                                headers={"X-CSRF-Token": self.api_token()})

    def test_pages_render_with_the_model(self):
        for path in ("/admin/", "/admin/model", "/admin/playground", "/admin/evaluation",
                     "/admin/system"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
        self.assertIn("Trained model loaded", self.client.get("/admin/").get_data(as_text=True))

    def test_model_card_lists_the_contract(self):
        body = self.client.get("/admin/model").get_data(as_text=True)
        self.assertIn("1, 224, 224, 3", body)
        self.assertIn("25.83", body)
        # The served export has no sensor input, so the card must not advertise one.
        self.assertNotIn("1, 27", body)
        self.assertNotIn("Temperature_Gas_Interaction", body)

    def test_model_card_shows_the_evidence_for_its_own_claims(self):
        """The card asserts the served model is best; the bundle's evidence must be on it."""
        body = self.client.get("/admin/model").get_data(as_text=True)
        # the 1-NN lookup probe: how much of the within-session score is retrieval
        self.assertIn("lookup", body.lower())
        self.assertIn("0.081", body)
        self.assertIn("No within-day split measures generalisation", body)
        # the branch-by-branch within-session comparison
        self.assertIn("Each branch, within a session", body)
        # the multi-seed variant study
        self.assertIn("What else was tried", body)
        self.assertIn("sensor_dropout", body)

    def test_playground_counts_readings_and_says_they_are_not_an_input(self):
        first = self.playground().get_json()["data"]
        second = self.playground(temperature="28.6").get_json()["data"]
        self.assertEqual((first["reading_number"], second["reading_number"]), (1, 2))
        self.assertTrue(any("photograph only" in w for w in first["warnings"]))
        self.assertFalse(any("First reading" in w for w in first["warnings"]))

    def test_playground_readings_do_not_move_the_number(self):
        image = jpeg_bytes(colour=(90, 140, 70))
        cold = self.playground(image=image, temperature="28.5").get_json()["data"]
        warm = self.playground(image=image, temperature="45").get_json()["data"]
        self.assertEqual(cold["predicted_days_raw"], warm["predicted_days_raw"])

    def test_playground_reset_starts_over(self):
        self.playground()
        self.playground()
        self.assertEqual(self.playground(reset=True).get_json()["data"]["reading_number"], 1)

    def test_playground_sessions_are_independent(self):
        self.playground(session="a")
        self.assertEqual(self.playground(session="b").get_json()["data"]["reading_number"], 1)

    def test_a_failed_request_does_not_advance_the_session(self):
        self.playground()
        bad = self.playground(image=b"not an image")
        self.assertEqual(bad.status_code, 415)
        self.assertEqual(self.playground().get_json()["data"]["reading_number"], 2)

    def test_playground_does_not_flag_an_envelope_it_does_not_read(self):
        data = self.playground(temperature="45").get_json()["data"]
        self.assertEqual(data["out_of_envelope"], {})

    def test_playground_rejects_bad_numbers(self):
        for value in ("warm", "nan", "inf"):
            response = self.playground(temperature=value)
            self.assertEqual(response.status_code, 422, value)

    def test_reset_endpoint(self):
        self.playground()
        response = self.post_json("/admin/api/playground/reset", {"session": "s1"})
        self.assertTrue(response.get_json()["data"]["existed"])

    def test_evaluation_scores_the_held_out_split(self):
        data = self.post_json("/admin/api/evaluate", {}).get_json()["data"]
        self.assertEqual(data["metrics"]["n"], 4)
        self.assertTrue(data["complete"])
        self.assertAlmostEqual(data["published"]["mae"], 0.1468)
        self.assertEqual(len(data["points"]), 4)

    def test_single_sample_scoring(self):
        data = self.post_json("/admin/api/test-samples/0/predict", {}).get_json()["data"]
        self.assertEqual(data["actual_days"], 10.0)
        self.assertAlmostEqual(data["absolute_error"],
                               abs(data["actual_days"] - data["predicted_days_raw"]), places=3)

    def test_samples_list_and_images(self):
        data = self.client.get("/admin/api/test-samples").get_json()["data"]
        self.assertEqual(data["count"], 4)
        image = self.client.get("/admin/api/test-samples/0/image")
        self.assertEqual(image.status_code, 200)
        self.assertEqual(self.client.get("/admin/api/test-samples/99/image").status_code, 404)

    def test_reload_keeps_the_model(self):
        data = self.post_json("/admin/api/reload-model", {}).get_json()["data"]
        self.assertTrue(data["model"]["trained_model_available"])

    def test_plain_users_cannot_run_the_model(self):
        self.logout()
        self.register_and_login(email="plain@example.com")
        self.assertEqual(self.playground().status_code, 403)
        self.assertEqual(self.post_json("/admin/api/evaluate", {}).status_code, 403)


# =============================================================================
# the real bundle
# =============================================================================
@needs_real_bundle
class RealBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RSLModel(REAL_BUNDLE)

    def test_loads_the_recommended_export(self):
        self.assertTrue(self.model.available, self.model.load_error)
        self.assertEqual(self.model.metrics["variant"], "float16")
        self.assertTrue(self.model.image_only)
        self.assertEqual(self.model.metrics["ensemble_heads"], 6)

    def test_envelope_matches_the_training_data(self):
        self.assertEqual(self.model.envelope["Temperature"]["min"], 25.83)
        self.assertEqual(self.model.envelope["Temperature"]["max"], 29.31)
        self.assertEqual(self.model.envelope["Gas"]["max"], 193.0)

    def test_new_batch_error_is_the_leave_one_day_out_figure(self):
        """The served ensemble's own measured figure - results/image_ensemble_lodo.json."""
        self.assertAlmostEqual(self.model.new_batch_mae_days, 0.848, places=3)

    def test_the_served_model_beats_the_alternatives_in_the_bundle(self):
        """The reason this export is the one being served, asserted rather than claimed."""
        lodo = self.model.metrics["leave_one_day_out_mae"]
        self.assertLess(lodo["image_ensemble"], lodo["fusion"])
        self.assertLess(lodo["image_ensemble"], lodo["image"])
        self.assertLess(lodo["image_ensemble"], lodo["single_head_average"])

    def test_the_feature_gates_passed(self):
        self.assertTrue(self.model.verification["features"]["passed"])
        serving = self.model.verification["serving"]
        self.assertTrue(serving["passed"])
        self.assertEqual(serving["originals_bit_identical"], serving["originals_checked"])


@needs_real_images
class RealBundleAccuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RSLModel(REAL_BUNDLE)
        cls.split = HeldOutSplit(REAL_BUNDLE)

    def test_the_website_reproduces_the_published_test_error(self):
        errors = []
        for row in self.split.rows:
            image = self.model.load_image_path(row["image_path"])
            errors.append(abs(self.model.predict_scaled(image.pixels) - row["actual"]))
        mae = sum(errors) / len(errors)
        self.assertEqual(len(errors), 271)
        self.assertAlmostEqual(mae, self.model.metrics["served_variant"]["MAE"], places=4)

    def test_the_documented_example(self):
        image = self.model.load_image_path(REAL_BUNDLE / "data_images" / "20260606_200106.jpg")
        self.assertAlmostEqual(self.model.predict(image).days, 9.55, places=2)

    def test_the_readings_cannot_move_it(self):
        image = self.model.load_image_path(REAL_BUNDLE / "data_images" / "20260606_200106.jpg")
        alone = self.model.predict(image).days
        extreme = self.model.predict(image, [Reading(45.0, 58.0, 185.0, 0.0)]).days
        self.assertEqual(alone, extreme)


@needs_real_fusion
class RealFusionAccuracy(unittest.TestCase):
    """The fusion export is kept in the bundle, so its behaviour is still pinned down."""

    @classmethod
    def setUpClass(cls):
        cls.model = RSLModel(REAL_BUNDLE, FUSION)
        cls.split = HeldOutSplit(REAL_BUNDLE)

    def setUp(self):
        if not (REAL_BUNDLE / "data_images").is_dir():
            self.skipTest("real training images not present (they are gitignored)")

    def test_it_reproduces_its_own_published_test_error(self):
        errors = []
        for row in self.split.rows:
            image = self.model.load_image_path(row["image_path"])
            scaled = self.model.scale(self.split.vector(row, self.model.feature_names))
            errors.append(abs(self.model.predict_scaled(image.pixels, scaled) - row["actual"]))
        mae = sum(errors) / len(errors)
        self.assertEqual(len(errors), 271)
        self.assertAlmostEqual(mae, self.model.metrics["served_variant"]["MAE"], places=3)

    def test_the_documented_example(self):
        image = self.model.load_image_path(REAL_BUNDLE / "data_images" / "20260606_200106.jpg")
        result = self.model.predict(image, [Reading(28.74, 58.04, 185.0, 0.0)])
        self.assertAlmostEqual(result.days, 9.59, places=2)

    def test_a_45_degree_cold_start_is_clamped_and_flagged(self):
        image = self.model.load_image_path(REAL_BUNDLE / "data_images" / "20260606_200106.jpg")
        result = self.model.predict(image, [Reading(45.0, 58.0, 185.0, 0.0)])
        self.assertLess(result.raw_days, 0)
        self.assertEqual(result.days, 0.0)
        self.assertIn("Temperature", result.out_of_envelope)


@needs_runtime
class BandLabelling(AppTestCase):
    """The uncertainty band must not be described as an assumption when it is measured."""

    sensor_source = "auto"

    @classmethod
    def setUpClass(cls):
        cls.rsl_bundle = str(fixture_bundle())

    def test_baseline_items_say_the_band_is_assumed(self):
        self.register_and_login()
        item = self.create_item()
        page = self.client.get(f"/items/{item['id']}").get_data(as_text=True)
        self.assertIn("assumed band", page)
        self.assertNotIn("measured on unseen days", page)

    def test_trained_items_say_the_band_was_measured(self):
        self.register_and_login()
        token = self.make_device()
        auth = {"Authorization": f"Bearer {token}"}
        stored = (utcnow() - timedelta(hours=2)).replace(microsecond=0).isoformat()
        item = self.create_item(label="Banded", food_key="banana", stored_at=stored)
        start = utcnow() - timedelta(minutes=10)
        for i in range(4):
            self.client.post("/api/device/v1/readings", headers=auth, json={
                "temperature_c": 28.4, "humidity_pct": 58.0, "gas_ppm": 184.0,
                "recorded_at": (start + timedelta(minutes=2 * i)).isoformat()})
        self.client.post("/api/device/v1/captures", headers=auth,
                         content_type="multipart/form-data",
                         data={"image": (io.BytesIO(jpeg_bytes()), "f.jpg"),
                               "item_id": str(item["id"])})
        self.post_json(f"/api/v1/items/{item['id']}/predict", {})
        page = self.client.get(f"/items/{item['id']}").get_data(as_text=True)
        self.assertIn("measured on unseen days", page)
        self.assertNotIn("assumed band", page)


if __name__ == "__main__":
    unittest.main()
