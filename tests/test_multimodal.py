"""Trained multimodal model integration.

The heavy tests build a real Keras model with the *same input contract* as the
project's trained export and exercise the whole path: load, validate,
preprocess, infer, and fall back. They are skipped automatically when the model
runtime is not installed, so the suite still passes on the Flask-only
environment.

Run them with the model environment:
    .venv-model/bin/python -m unittest tests.test_multimodal
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from tests.support import AppTestCase

from shelflife.dataset import HeldOutSplit, regression_metrics
from shelflife.multimodal import MultimodalShelfLifeModel

try:
    import numpy  # noqa: F401
    import tensorflow  # noqa: F401
    from PIL import Image  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    import joblib  # noqa: F401

    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False

needs_runtime = unittest.skipUnless(
    HAS_RUNTIME, "model runtime (tensorflow/sklearn/pillow/joblib) not installed"
)


class GracefulDegradation(unittest.TestCase):
    """Behaviour on an install with no artifacts - the common starting state."""

    def test_missing_artifacts_report_clearly_and_do_not_raise(self):
        model = MultimodalShelfLifeModel("nope/model.keras", "nope/scaler.pkl", "nope/config.json")
        self.assertFalse(model.available)
        self.assertIn("Missing artifact", model.load_error)

    def test_status_is_serialisable_when_unavailable(self):
        model = MultimodalShelfLifeModel("nope/model.keras", "nope/scaler.pkl")
        payload = json.dumps(model.status())
        self.assertIn("available", payload)

    def test_predicting_without_a_model_raises_model_unavailable(self):
        from shelflife.multimodal import ModelUnavailable

        model = MultimodalShelfLifeModel("nope/model.keras", "nope/scaler.pkl")
        with self.assertRaises(ModelUnavailable):
            model.predict(None, 25.0, 60.0, 180.0)


class Metrics(unittest.TestCase):
    def test_known_values(self):
        result = regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertEqual(result["mae"], 0.0)
        self.assertEqual(result["rmse"], 0.0)
        self.assertEqual(result["r2"], 1.0)

    def test_bias_sign_shows_over_prediction(self):
        over = regression_metrics([1.0, 2.0], [2.0, 3.0])
        under = regression_metrics([2.0, 3.0], [1.0, 2.0])
        self.assertGreater(over["bias"], 0)
        self.assertLess(under["bias"], 0)

    def test_empty_input_is_safe(self):
        self.assertEqual(regression_metrics([], [])["n"], 0)

    def test_r2_is_none_when_actuals_have_no_variance(self):
        self.assertIsNone(regression_metrics([5.0, 5.0], [4.0, 6.0])["r2"])


class SplitParsing(unittest.TestCase):
    def _write(self, body: str) -> Path:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path = Path(self.tmp.name) / "splits.csv"
        path.write_text(body, encoding="utf-8")
        return path

    def test_missing_file_reports_the_path(self):
        split = HeldOutSplit("/nowhere/splits.csv", "/nowhere/images")
        self.assertFalse(split.available)
        self.assertIn("/nowhere/splits.csv", split.error)

    def test_missing_columns_are_named(self):
        path = self._write("split,Imagepath\ntest,a.jpg\n")
        split = HeldOutSplit(path, "/tmp")
        self.assertIn("missing columns", split.error)

    def test_only_test_rows_are_kept(self):
        path = self._write(
            "split,Imagepath,resolved_image_path,Temperature,Humidity,Gas,RemainingShelfLife\n"
            "train,a.jpg,,20,60,150,5\n"
            "test,b.jpg,,21,61,151,4\n"
            "val,c.jpg,,22,62,152,3\n"
            "test,d.jpg,,23,63,153,2\n"
        )
        split = HeldOutSplit(path, "/tmp")
        self.assertEqual(len(split), 2)
        self.assertEqual(split.counts, {"train": 1, "test": 2, "val": 1})

    def test_unparseable_rows_are_skipped_not_fatal(self):
        path = self._write(
            "split,Imagepath,resolved_image_path,Temperature,Humidity,Gas,RemainingShelfLife\n"
            "test,a.jpg,,20,60,150,5\n"
            "test,b.jpg,,not-a-number,61,151,4\n"
        )
        split = HeldOutSplit(path, "/tmp")
        self.assertEqual(len(split), 1)

    def test_index_out_of_range_returns_none(self):
        path = self._write(
            "split,Imagepath,resolved_image_path,Temperature,Humidity,Gas,RemainingShelfLife\n"
            "test,a.jpg,,20,60,150,5\n"
        )
        split = HeldOutSplit(path, "/tmp")
        self.assertIsNotNone(split.get(0))
        self.assertIsNone(split.get(1))
        self.assertIsNone(split.get(-1))


@needs_runtime
class RealModelContract(unittest.TestCase):
    """Exercises the adapter against a genuine Keras model."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tools.make_fixture_model import main as build

        build(root)
        cls.root = root
        cls.model = MultimodalShelfLifeModel(
            root / "models" / "best_multimodal_model.keras",
            root / "artifacts" / "sensor_scaler.pkl",
            root / "artifacts" / "config.json",
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def image(self):
        return self.model.open_image(self.root / "data" / "Images" / "fixture_000.jpg")

    def test_model_loads(self):
        self.assertTrue(self.model.available, self.model.load_error)

    def test_input_contract_is_validated(self):
        self.assertEqual(sorted(self.model.input_names), ["image", "sensor"])
        self.assertEqual(self.model.input_shapes["image"], (None, 224, 224, 3))
        self.assertEqual(self.model.input_shapes["sensor"], (None, 3))

    def test_metrics_are_read_from_config(self):
        self.assertIn("test_mae", self.model.metrics)

    def test_image_preprocessing_shape_and_dtype(self):
        tensor = self.model.preprocess_image(self.image())
        self.assertEqual(tensor.shape, (1, 224, 224, 3))
        self.assertEqual(tensor.dtype.name, "float32")

    def test_sensor_preprocessing_uses_the_fitted_scaler(self):
        tensor = self.model.preprocess_sensor(28.0, 60.0, 185.0)
        self.assertEqual(tensor.shape, (1, 3))
        # A fitted StandardScaler maps mid-range inputs near zero, never identity.
        self.assertLess(abs(float(tensor[0, 0])), 5.0)
        self.assertNotAlmostEqual(float(tensor[0, 0]), 28.0)

    def test_prediction_is_a_finite_number_of_days(self):
        import math

        output = self.model.predict(self.image(), 28.0, 60.0, 185.0)
        self.assertTrue(math.isfinite(output.remaining_days))
        self.assertGreater(output.inference_ms, 0)

    def test_prediction_is_deterministic(self):
        first = self.model.predict(self.image(), 28.0, 60.0, 185.0).remaining_days
        second = self.model.predict(self.image(), 28.0, 60.0, 185.0).remaining_days
        self.assertAlmostEqual(first, second, places=5)

    def test_sensor_values_change_the_output(self):
        cold = self.model.predict(self.image(), 5.0, 60.0, 120.0).remaining_days
        hot = self.model.predict(self.image(), 40.0, 95.0, 300.0).remaining_days
        self.assertNotAlmostEqual(cold, hot, places=4)

    def test_missing_sensor_input_is_refused_not_imputed(self):
        from shelflife.multimodal import ModelUnavailable

        for triple in ((None, 60.0, 185.0), (28.0, None, 185.0), (28.0, 60.0, None)):
            with self.assertRaises(ModelUnavailable):
                self.model.predict(self.image(), *triple)

    def test_corrupt_image_raises_a_clear_error(self):
        bad = self.root / "corrupt.jpg"
        bad.write_bytes(b"this is not an image")
        with self.assertRaises(ValueError):
            self.model.open_image(bad)

    def test_mismatched_sensor_order_is_rejected(self):
        config = self.root / "artifacts" / "config.json"
        original = config.read_text(encoding="utf-8")
        payload = json.loads(original)
        payload["sensor_columns"] = ["Humidity", "Temperature", "Gas"]
        config.write_text(json.dumps(payload), encoding="utf-8")
        try:
            model = MultimodalShelfLifeModel(
                self.root / "models" / "best_multimodal_model.keras",
                self.root / "artifacts" / "sensor_scaler.pkl",
                config,
            )
            self.assertFalse(model.available)
            self.assertIn("sensor_columns", model.load_error)
        finally:
            config.write_text(original, encoding="utf-8")

    def test_a_model_with_the_wrong_signature_is_refused(self):
        import tensorflow as tf

        wrong = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(3,))])
        path = self.root / "wrong.keras"
        wrong.save(path)
        model = MultimodalShelfLifeModel(
            path, self.root / "artifacts" / "sensor_scaler.pkl"
        )
        self.assertFalse(model.available)
        self.assertIn("inputs", model.load_error.lower())


@needs_runtime
class RegistrySelection(unittest.TestCase):
    """Which estimator serves a request, and why."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tools.make_fixture_model import main as build

        build(root)
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def registry(self):
        from shelflife.inference import PredictorRegistry

        return PredictorRegistry(
            keras_model_path=str(self.root / "models" / "best_multimodal_model.keras"),
            scaler_path=str(self.root / "artifacts" / "sensor_scaler.pkl"),
            config_path=str(self.root / "artifacts" / "config.json"),
        )

    def context(self, **overrides):
        from shelflife.inference import PredictionContext

        base = dict(
            food_key="tomato", food_name="Tomato", ref_shelf_life_hours=168.0,
            ref_temperature_c=20.0, q10=2.4, ideal_temp_min_c=10.0, ideal_temp_max_c=15.0,
            ideal_humidity_min=85.0, ideal_humidity_max=95.0,
            temperature_c=26.0, humidity_pct=70.0, gas_ppm=180.0, hours_stored=24.0,
            image_full_path=str(self.root / "data" / "Images" / "fixture_000.jpg"),
        )
        base.update(overrides)
        return PredictionContext(**base)

    def test_trained_model_is_used_when_every_input_is_present(self):
        prediction = self.registry().predict(self.context())
        self.assertEqual(prediction.model_kind, "trained-model")
        self.assertTrue(prediction.image_used)
        self.assertIsNone(prediction.fallback_reason)

    def test_status_reports_the_trained_model_as_active(self):
        status = self.registry().status()
        self.assertTrue(status["trained_model_available"])
        self.assertEqual(status["active_kind"], "trained-model")
        self.assertTrue(status["uses_image"])

    def test_missing_gas_falls_back_and_says_so(self):
        prediction = self.registry().predict(self.context(gas_ppm=None))
        self.assertEqual(prediction.model_kind, "heuristic-baseline")
        self.assertIn("gas", prediction.fallback_reason.lower())

    def test_missing_image_falls_back_and_says_so(self):
        prediction = self.registry().predict(self.context(image_full_path=None))
        self.assertEqual(prediction.model_kind, "heuristic-baseline")
        self.assertIn("photograph", prediction.fallback_reason.lower())

    def test_image_path_that_does_not_exist_falls_back(self):
        prediction = self.registry().predict(self.context(image_full_path="/nope/missing.jpg"))
        self.assertEqual(prediction.model_kind, "heuristic-baseline")

    def test_both_inputs_missing_reports_both_reasons(self):
        prediction = self.registry().predict(self.context(gas_ppm=None, image_full_path=None))
        self.assertIn("and", prediction.fallback_reason)

    def test_trained_prediction_never_reports_a_fake_confidence(self):
        prediction = self.registry().predict(self.context())
        self.assertIsNone(prediction.freshness_confidence)

    def test_interval_comes_from_the_published_mae(self):
        prediction = self.registry().predict(self.context())
        # config.json in the fixture publishes test_mae = 0.82 days
        self.assertIsNotNone(prediction.remaining_hours_low)
        spread = prediction.remaining_hours_high - prediction.remaining_hours
        self.assertAlmostEqual(spread, 0.82 * 24, places=2)

    def test_no_interval_is_invented_without_a_published_metric(self):
        config = self.root / "artifacts" / "config.json"
        original = config.read_text(encoding="utf-8")
        payload = json.loads(original)
        payload.pop("metrics", None)
        config.write_text(json.dumps(payload), encoding="utf-8")
        try:
            prediction = self.registry().predict(self.context())
            self.assertIsNone(prediction.remaining_hours_low)
            self.assertIn("No held-out error metric", prediction.rationale)
        finally:
            config.write_text(original, encoding="utf-8")

    def test_negative_output_is_floored_at_zero_hours(self):
        prediction = self.registry().predict(self.context())
        self.assertGreaterEqual(prediction.remaining_hours, 0.0)

    def test_rationale_names_the_model_and_the_inputs(self):
        rationale = self.registry().predict(self.context()).rationale
        self.assertIn("Gas=", rationale)
        self.assertIn("derived from that regression output", rationale)


@needs_runtime
class AdminConsoleWithModel(AppTestCase):
    """The developer console driving a real model end to end."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tools.make_fixture_model import main as build

        build(Path(cls.fixture.name))

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        super().setUp()
        root = Path(self.fixture.name)
        self.app.config.update(
            KERAS_MODEL_PATH=str(root / "models" / "best_multimodal_model.keras"),
            SCALER_PATH=str(root / "artifacts" / "sensor_scaler.pkl"),
            MODEL_CONFIG_PATH=str(root / "artifacts" / "config.json"),
            DATASET_SPLITS_PATH=str(root / "artifacts" / "dataset_splits.csv"),
            DATASET_IMAGE_DIR=str(root / "data" / "Images"),
        )
        from shelflife.dataset import clear_cache

        clear_cache()
        self.addCleanup(clear_cache)
        self.register_and_login_admin()
        # Reload so the app picks up the fixture paths set above.
        response = self.post_json("/admin/api/reload-model", {})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))

    def test_console_reports_the_model_as_loaded(self):
        body = self.client.get("/admin/").get_data(as_text=True)
        self.assertIn("Trained model loaded", body)

    def test_playground_runs_inference_on_an_upload(self):
        import io

        image_bytes = (Path(self.fixture.name) / "data" / "Images" / "fixture_001.jpg").read_bytes()
        response = self.client.post(
            "/admin/api/predict",
            data={
                "image": (io.BytesIO(image_bytes), "sample.jpg"),
                "temperature": "28.0", "humidity": "60.0", "gas": "185.0",
            },
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": self.api_token()},
        )
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        data = response.get_json()["data"]
        self.assertIn("predicted_days", data)
        self.assertGreaterEqual(data["predicted_days"], 0.0)
        self.assertGreater(data["inference_ms"], 0)

    def test_playground_rejects_a_non_numeric_sensor_value(self):
        import io

        response = self.client.post(
            "/admin/api/predict",
            data={
                "image": (io.BytesIO(b"\xff\xd8\xff"), "x.jpg"),
                "temperature": "warm", "humidity": "60", "gas": "185",
            },
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": self.api_token()},
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "invalid_sensor")

    def test_test_samples_are_listed(self):
        data = self.client.get("/admin/api/test-samples").get_json()["data"]
        self.assertGreater(data["count"], 0)
        self.assertTrue(all(s["has_image"] for s in data["samples"]))

    def test_a_held_out_sample_can_be_scored(self):
        response = self.post_json("/admin/api/test-samples/0/predict", {})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()["data"]
        self.assertIn("actual_days", data)
        self.assertIn("predicted_days", data)
        self.assertAlmostEqual(
            data["absolute_error"],
            abs(data["actual_days"] - data["predicted_days_raw"]),
            places=3,
        )

    def test_out_of_range_sample_index_is_404(self):
        self.assertEqual(self.post_json("/admin/api/test-samples/9999/predict", {}).status_code, 404)

    def test_evaluation_reports_metrics_over_the_split(self):
        response = self.post_json("/admin/api/evaluate", {"limit": 4})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()["data"]
        self.assertGreater(data["metrics"]["n"], 0)
        self.assertIsNotNone(data["metrics"]["mae"])
        self.assertEqual(len(data["points"]), data["metrics"]["n"])

    def test_test_images_are_served_by_index_only(self):
        response = self.client.get("/admin/api/test-samples/0/image")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["Content-Type"].startswith("image/"))
        self.assertEqual(self.client.get("/admin/api/test-samples/9999/image").status_code, 404)

    def test_user_items_use_the_trained_model_when_inputs_are_present(self):
        import io

        image_bytes = (Path(self.fixture.name) / "data" / "Images" / "fixture_002.jpg").read_bytes()
        upload = self.client.post(
            "/api/v1/uploads",
            data={"image": (io.BytesIO(image_bytes), "item.jpg")},
            content_type="multipart/form-data",
            headers={"X-CSRF-Token": self.api_token()},
        )
        stored = upload.get_json()["data"]["image_path"]
        item = self.create_item(label="Photographed tomatoes", image_path=stored)
        # No gas sensor in the simulated feed, so this must fall back and say so.
        self.assertEqual(item["model_kind"], "heuristic-baseline")
        self.assertIn("gas", (item["fallback_reason"] or "").lower())


if __name__ == "__main__":
    unittest.main()
