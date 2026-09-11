# Model artifacts

Drop the trained export here. The application picks it up automatically and
switches every prediction over to it; while these files are absent it serves the
documented Q10 kinetic baseline and says so in the UI.

```
capstone_project/
├── models/
│   └── best_multimodal_model.keras     <- the trained Keras model
├── artifacts/
│   ├── sensor_scaler.pkl               <- the fitted StandardScaler
│   ├── config.json                     <- contract + published metrics
│   └── dataset_splits.csv              <- held-out split metadata (admin only)
└── data/
    └── Images/                         <- images referenced by the splits file
```

`config.json` must declare the feature order, and may publish metrics:

```json
{
  "name": "Multimodal MobileNetV3 + sensor MLP",
  "version": "1.0.0",
  "sensor_columns": ["Temperature", "Humidity", "Gas"],
  "image_size": [224, 224],
  "metrics": {"test_mae": 0.82, "test_rmse": 1.14, "test_r2": 0.71}
}
```

The loader validates the model signature on startup and refuses to run anything
whose inputs are not exactly `image (None, 224, 224, 3)` and
`sensor (None, 3)` with a single scalar output. A mismatch is reported on the
**Model card** page instead of being silently tolerated.

Without a `metrics` block, predictions are shown with **no** uncertainty
interval rather than an invented one.

Run `flask --app app model-check` to print the loaded contract.
