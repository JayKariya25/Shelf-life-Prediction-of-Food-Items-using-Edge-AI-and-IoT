"""Generate the three walkthrough notebooks and execute them so they ship with outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)

BOOT = """import sys, warnings
sys.path.insert(0, "../scripts")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt, tensorflow as tf
import common as C, datasets as D, models as M
tf.get_logger().setLevel("ERROR")
print("TensorFlow", tf.__version__)"""


def nb(cells):
    n = nbf.v4.new_notebook(cells=cells)
    n.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return n


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


# --------------------------------------------------------------------------------------
def notebook_sensor():
    return nb([
        md("""
# 01 - Sensor branch

Trains the **sensor-only** RSL model on the engineered features produced by
`01sensor_preprocessing.ipynb` and `02_sensor_feature_engineering.ipynb`.

Target: `RemainingShelfLife = 11 - Day`, in days.

This notebook is the walkthrough; `scripts/train_sensor.py` is the same thing as a
runnable script.
"""),
        code(BOOT),
        md("""
## 1. Which columns may the model see?

`sensor_features.csv` has 58 columns and several of them are clocks rather than sensors.
Anything that encodes elapsed time reconstructs `RSL = 11 - Day` arithmetically, so the
model would score near-perfectly and learn nothing about spoilage - and a Raspberry Pi
handed an arbitrary sample could not supply those columns anyway.
"""),
        code("""
raw = pd.read_csv(C.SENSOR_FEATURES_CSV)
num = raw.select_dtypes("number")
corr = num.corrwith(raw["Day"]).abs().sort_values(ascending=False)
print("Top correlations with Day (the target is 11 - Day):")
print(corr.head(10).round(4).to_string())
print()
print("Excluded as leakage:", C.LEAKAGE_COLUMNS)
print()
print(f"Model inputs ({len(C.feature_columns('full'))}):")
print(C.feature_columns("full"))
"""),
        md("""
The `*_Level` band columns are excluded for a separate reason - they are not a function of
the reading they describe, so they cannot be recomputed on a Pi.
"""),
        code("""
for ch in ["Temperature", "Humidity", "Gas"]:
    g = raw.groupby(f"{ch}_Level")[ch].agg(["min", "max", "count"]).sort_values("min")
    print(ch); print(g.to_string()); print()
"""),
        md("""
## 2. Splits

Frames arrive roughly every 2 minutes, so neighbouring frames are near-duplicates. The
**blocked** split assigns whole 30-minute blocks to train/val/test so a frame and its twin
cannot straddle the boundary. The **random** split is kept only for comparison.
"""),
        code("""
df, X, F, y, (tr, va, te) = D.load_all("blocked", "full")
print(f"samples: {len(df)}   sensor features: {X.shape[1]}")
print(f"train {tr.sum()}   val {va.sum()}   test {te.sum()}")
display(df.groupby("Day")[["split_blocked"]].value_counts().unstack(fill_value=0))
"""),
        md("""
## 3. Architecture

The penultimate layer, `sensor_embedding`, is the 64-d vector the fusion stage will
consume. The final `sensor_rsl` unit lets this branch stand alone as a complete model.
"""),
        code("""
C.set_seeds()
sensor = M.compile_regressor(M.build_sensor_branch(X.shape[1]))
sensor.summary()
"""),
        code("""
history = sensor.fit(
    X[tr], y[tr], validation_data=(X[va], y[va]),
    epochs=200, batch_size=64, callbacks=M.default_callbacks(), verbose=0,
)
print(f"stopped after {len(history.history['loss'])} epochs")
h = pd.DataFrame(history.history)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
h[["loss", "val_loss"]].plot(ax=ax[0], title="Huber loss"); ax[0].grid(alpha=.3)
h[["mae", "val_mae"]].plot(ax=ax[1], title="MAE (days)"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
"""),
        md("## 4. Held-out performance"),
        code("""
for part, mask in [("train", tr), ("val", va), ("test", te)]:
    m = C.regression_metrics(y[mask], sensor.predict(X[mask], verbose=0))
    print(f"{part:>5}  MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}  R2 {m['R2']:.4f}")
"""),
        code("""
pred = sensor.predict(X[te], verbose=0).ravel()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(y[te], pred, s=10, alpha=.35, color="#2f6f9f")
lims = [y[te].min()-.5, y[te].max()+.5]
ax[0].plot(lims, lims, "--", color="#c0392b"); ax[0].set_xlim(lims); ax[0].set_ylim(lims)
ax[0].set_xlabel("Actual RSL (days)"); ax[0].set_ylabel("Predicted"); ax[0].grid(alpha=.3)
ax[1].hist(pred - y[te], bins=40, color="#2f6f9f"); ax[1].axvline(0, color="#c0392b", ls="--")
ax[1].set_xlabel("Error (days)"); ax[1].grid(alpha=.3)
plt.suptitle("Sensor branch - blocked test split"); plt.tight_layout(); plt.show()
"""),
        md("""
Temperature, humidity and a single gas channel carry real but limited information about
spoilage stage: around 0.75 days of mean error even within a monitoring session. The image
branch (notebook 02) does considerably better, and notebook 03 shows the sensor branch is
also the one that generalises worst to a day it has never seen.
"""),
    ])


# --------------------------------------------------------------------------------------
def notebook_image():
    return nb([
        md("""
# 02 - Image branch

Trains the **image-only** RSL model on the 224x224 images produced by
`ImagePreprocessing.ipynb`.

MobileNetV3Small (ImageNet) is used as a **frozen** feature extractor. Because it never
updates, its pooled 576-d output is constant for a given image, so it is computed once and
cached - which turns training from hours of CPU convolutions into seconds and makes the
whole pipeline cheap to re-run.
"""),
        code(BOOT),
        md("## 1. The cached backbone features"),
        code("""
df, X, F, y, (tr, va, te) = D.load_all("blocked", "full")
print(f"backbone feature matrix: {F.shape}  (one 576-d row per image)")
print(f"train {tr.sum()}   val {va.sum()}   test {te.sum()}")
"""),
        code("""
from PIL import Image
sample = df.groupby("Day").head(1).sort_values("Day")
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for ax, (_, r) in zip(axes.ravel(), sample.iterrows()):
    ax.imshow(Image.open(C.IMAGE_DIR / r.image_file)); ax.axis("off")
    ax.set_title(f"Day {int(r.Day)} - RSL {r.RemainingShelfLife:.0f}d", fontsize=9)
for ax in axes.ravel()[len(sample):]:
    ax.axis("off")
plt.suptitle("One frame per monitoring day"); plt.tight_layout(); plt.show()
"""),
        md("""
## 2. Head architecture

Only this head is trained. `image_embedding` (128-d) is what the fusion stage consumes;
`image_rsl` makes the branch a complete standalone model.
"""),
        code("""
C.set_seeds()
head = M.compile_regressor(M.build_image_head(F.shape[1]))
head.summary()
"""),
        code("""
history = head.fit(
    F[tr], y[tr], validation_data=(F[va], y[va]),
    epochs=200, batch_size=64, callbacks=M.default_callbacks(), verbose=0,
)
print(f"stopped after {len(history.history['loss'])} epochs")
h = pd.DataFrame(history.history)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
h[["loss", "val_loss"]].plot(ax=ax[0], title="Huber loss"); ax[0].grid(alpha=.3)
h[["mae", "val_mae"]].plot(ax=ax[1], title="MAE (days)"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
"""),
        md("## 3. Held-out performance"),
        code("""
for part, mask in [("train", tr), ("val", va), ("test", te)]:
    m = C.regression_metrics(y[mask], head.predict(F[mask], verbose=0))
    print(f"{part:>5}  MAE {m['MAE']:.3f}  RMSE {m['RMSE']:.3f}  R2 {m['R2']:.4f}")
"""),
        code("""
pred = head.predict(F[te], verbose=0).ravel()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(y[te], pred, s=10, alpha=.35, color="#2f6f9f")
lims = [y[te].min()-.5, y[te].max()+.5]
ax[0].plot(lims, lims, "--", color="#c0392b"); ax[0].set_xlim(lims); ax[0].set_ylim(lims)
ax[0].set_xlabel("Actual RSL (days)"); ax[0].set_ylabel("Predicted"); ax[0].grid(alpha=.3)
ax[1].hist(pred - y[te], bins=40, color="#2f6f9f"); ax[1].axvline(0, color="#c0392b", ls="--")
ax[1].set_xlabel("Error (days)"); ax[1].grid(alpha=.3)
plt.suptitle("Image branch - blocked test split"); plt.tight_layout(); plt.show()
"""),
        md("""
## 4. Worst cases

Worth a look before the viva: the frames the image branch gets most wrong are the ones to
be able to explain.
"""),
        code("""
err = np.abs(pred - y[te])
worst = df.loc[te].assign(pred=pred, err=err).nlargest(5, "err")
fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
for ax, (_, r) in zip(axes, worst.iterrows()):
    ax.imshow(Image.open(C.IMAGE_DIR / r.image_file)); ax.axis("off")
    ax.set_title(f"true {r.RemainingShelfLife:.0f}d\\npred {r.pred:.1f}d", fontsize=9)
plt.suptitle("Largest image-branch errors"); plt.tight_layout(); plt.show()
"""),
    ])


# --------------------------------------------------------------------------------------
def notebook_fusion():
    return nb([
        md("""
# 03 - Fusion, evaluation and Raspberry Pi export

Both branches are now **frozen**. Only the fusion head is trained, on

    concat[ image_embedding (128) | sensor_embedding (64) | image RSL | sensor RSL ]

Appending each branch's own scalar prediction matters: it gives the head a ready-made
baseline, so fusing can refine the stronger branch instead of having to relearn it.
"""),
        code(BOOT),
        code("""
df, X, F, y, (tr, va, te) = D.load_all("blocked", "full")
sensor = tf.keras.models.load_model(C.MODELS / "sensor_model_blocked_full.keras", compile=False)
image  = tf.keras.models.load_model(C.MODELS / "image_head_blocked_full.keras", compile=False)
sensor.trainable = False
image.trainable = False

def embed(model, layer, A):
    return tf.keras.Model(model.input, model.get_layer(layer).output).predict(A, verbose=0)

E = np.concatenate([
    embed(image, "image_embedding", F),
    embed(sensor, "sensor_embedding", X),
    image.predict(F, verbose=0),
    sensor.predict(X, verbose=0),
], axis=1).astype("float32")
print("fusion input:", E.shape)
"""),
        code("""
C.set_seeds()
fusion = M.compile_regressor(M.build_fusion_head())
fusion.summary()
history = fusion.fit(E[tr], y[tr], validation_data=(E[va], y[va]),
                     epochs=200, batch_size=64, callbacks=M.default_callbacks(), verbose=0)
print(f"stopped after {len(history.history['loss'])} epochs")
"""),
        md("## 1. Does fusing actually help?"),
        code("""
rows = {
    "Sensor only": sensor.predict(X[te], verbose=0).ravel(),
    "Image only":  image.predict(F[te], verbose=0).ravel(),
    "Fusion":      fusion.predict(E[te], verbose=0).ravel(),
}
table = pd.DataFrame({k: C.regression_metrics(y[te], v) for k, v in rows.items()}).T
display(table[["MAE", "RMSE", "R2", "WithinHalfDay", "WithinOneDay"]].round(4))

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
lims = [y[te].min()-.5, y[te].max()+.5]
for a, (name, p) in zip(ax, rows.items()):
    a.scatter(y[te], p, s=10, alpha=.35, color="#2f6f9f")
    a.plot(lims, lims, "--", color="#c0392b"); a.set_xlim(lims); a.set_ylim(lims)
    a.set_title(f"{name}  MAE {C.regression_metrics(y[te], p)['MAE']:.3f} d")
    a.set_xlabel("Actual RSL (days)"); a.grid(alpha=.3)
ax[0].set_ylabel("Predicted RSL (days)")
plt.tight_layout(); plt.show()
"""),
        md("""
## 2. The deployable graph

The saved multimodal model takes raw 0-255 pixels and the scaled sensor vector directly.
It must be numerically identical to the staged pipeline above - checked here rather than
assumed.
"""),
        code("""
multimodal = tf.keras.models.load_model(C.MODELS / "fusion_model_blocked_full.keras", compile=False)
files = df.loc[te, "image_file"].tolist()[:64]
imgs = np.stack([
    tf.image.resize(tf.io.decode_jpeg(tf.io.read_file(str(C.IMAGE_DIR / f)), channels=3),
                    C.IMAGE_SIZE).numpy() for f in files
])
e2e = multimodal.predict([imgs, X[te][:64]], verbose=0).ravel()
staged = fusion.predict(E[te][:64], verbose=0).ravel()
print("max |assembled graph - staged pipeline| =", float(np.max(np.abs(e2e - staged))))
"""),
        md("""
## 3. Raspberry Pi export

Four TFLite conversions, each re-scored on the test set rather than assumed equivalent.
"""),
        code("""
import json
rep = json.loads((C.RESULTS / "blocked_full" / "tflite_report.json").read_text())
display(pd.DataFrame({
    k: {"size_MB": v["size_mb"], "MAE": round(v["metrics"]["MAE"], 3),
        "R2": round(v["metrics"]["R2"], 4),
        "ms/sample": v["latency_ms_per_sample_x86_cpu"], "backend": v["cpu_backend"]}
    for k, v in rep["variants"].items()
}).T)
print("recommended:", rep["recommended_variant"])
"""),
        md("""
`float16` is the deployment artifact: half the size of float32, accuracy unchanged, still
XNNPACK-accelerated on a Pi 4/5.

Full-integer `int8` loses far more than usual here. MobileNetV3's hard-swish and
squeeze-excite blocks quantize badly, and this task needs sub-day resolution over a narrow
output range, so quantization noise is large relative to the signal - and the XNNPACK
delegate refuses the fully-quantized graph, making it slower as well as less accurate. If
int8 becomes mandatory (Coral Edge TPU), the route is quantization-aware training or a
MobileNetV2 backbone, not a different converter flag.

## 4. How much of the test score is just lookup?

Frames arrive every ~2 minutes, so the honest first question is whether the test set can be
answered from memory. This probe copies each test row's label from its nearest train row
using *untrained* ImageNet features - no learning at all. If it rivals the trained model,
the split is leaking.
"""),
        code("""
abl = json.loads((C.RESULTS / "ablations.json").read_text())
probe = abl["nearest_neighbour_probe"]
display(pd.DataFrame({
    s: {"median min to nearest train row": p["minutes_to_nearest_train_row"]["median"],
        "1-NN image MAE": round(p["frozen_imagenet_features"]["metrics"]["MAE"], 3),
        "1-NN sensor MAE": round(p["scaled_sensor_vector"]["metrics"]["MAE"], 3)}
    for s, p in probe.items()
}).T)
"""),
        md("""
On the random split a nearest-neighbour lookup scores ~0.02 days: that test set is
answerable from memory, so any model score on it is meaningless. The blocked+embargo split
is harder but still not clean - within one continuous experiment, same-day frames look
alike however far apart in time. **No within-day split measures generalisation here.**

## 5. A day the model has never seen

Leave-one-day-out removes every frame of one day from training, and validates on a
*different* held-out day so early stopping is also chosen across days.
"""),
        code("""
lodo = abl["leave_one_day_out"]
display(pd.DataFrame({
    d: {"n": v["held_out_samples"], "true_RSL": v["true_RSL"],
        "fusion_pred": round(v["mean_predicted_RSL"], 2),
        "sensor_MAE": round(v["metrics"]["sensor"]["MAE"], 3),
        "image_MAE": round(v["metrics"]["image"]["MAE"], 3),
        "fusion_MAE": round(v["metrics"]["fusion"]["MAE"], 3)}
    for d, v in sorted(lodo.items(), key=lambda kv: int(kv[0]))
}).T)
print("mean across held-out days:", abl["leave_one_day_out_summary"]["mean_MAE"])
"""),
        md("""
**The ordering flips.** Within a session fusion wins; across sessions the image branch alone
wins and fusion is worse, because the sensor branch generalises poorly to a new day and the
fusion head has learned to trust it. Temperature, humidity and one gas channel describe the
ambient conditions of *that run* more than they describe spoilage; the camera sees the food.

So: deploy the fusion model for continuous monitoring of a batch already under observation,
and treat the image branch as the more reliable predictor for a cold read on an unfamiliar
batch. Closing that gap needs more monitoring runs, not more model capacity.

## 6. Running it on the Pi

```bash
pip install tflite-runtime pillow numpy scikit-learn
python scripts/predict_pi.py --image capture.jpg --temperature 28.7 --humidity 58.0 --gas 185
```
"""),
    ])


def main() -> None:
    builders = {
        "01_sensor_model.ipynb": notebook_sensor,
        "02_image_model.ipynb": notebook_image,
        "03_fusion_and_deployment.ipynb": notebook_fusion,
    }
    for name, build in builders.items():
        path = NB_DIR / name
        notebook = build()
        print(f"executing {name} ...")
        NotebookClient(notebook, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(NB_DIR)}}).execute()
        nbf.write(notebook, path)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
