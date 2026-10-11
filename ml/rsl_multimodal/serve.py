"""HTTP API that turns the trained RSL model into something a website can call.

    python serve.py                  # http://127.0.0.1:8001
    python serve.py --port 5000 --host 0.0.0.0

The default port is 8001 so it does not collide with the monitoring website, which
serves on 8000 and runs this model in-process anyway.

Endpoints
---------
GET  /                     built-in test page: upload a photo, type 3 readings, see the RSL
GET  /api/health           model + version info, for uptime checks
POST /api/predict          the one your website calls
POST /api/session/reset    clear a session's rolling history
GET  /api/sessions         what sessions are currently open

Why sessions matter
-------------------
The sensor branch uses rolling features - change since the last reading, rate of change,
a 5-sample rolling mean and std, drift since the start of monitoring. Those need history.
Pass a stable `session_id` per food batch (e.g. the tray id, or the Pi's device id) and
the server keeps that batch's rolling window. The very first reading of a session has no
history, so its prediction is the weakest one; the response flags this in `warnings`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))
from image_io import load_image_bytes  # noqa: E402
from pi_features import CHANNELS, SensorFeatureBuilder  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
DEFAULT_MODEL = MODELS / "fusion_blocked_full_float16.tflite"
IMAGE_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

# Shelf-life bands shown to end users. Tune these to whatever your product needs.
BANDS = [
    (5.0, "fresh", "Good for several more days"),
    (2.0, "use_soon", "Use within a couple of days"),
    (0.0, "expiring", "Use immediately"),
]


# --------------------------------------------------------------------------------------
# Model wrapper
# --------------------------------------------------------------------------------------
class RSLModel:
    """Loads the TFLite model once and serves single predictions, thread-safely."""

    def __init__(self, model_path: Path, split: str = "blocked", feature_set: str = "full"):
        self.model_path = model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found - run scripts/export_tflite.py first."
            )

        with open(ARTIFACTS / "sensor_scaler.pkl", "rb") as fh:
            bundle = pickle.load(fh)
        self.feature_names = bundle["features"][feature_set]
        self.scaler = bundle["scalers"][f"{split}__{feature_set}"]

        config = json.loads((ARTIFACTS / "preprocessing_config.json").read_text())
        self.bounds = config["outlier_bounds"]
        self.envelope = config.get("input_envelope", {})
        self.max_shelf_life = float(config["target"]["definition"].split("-")[0].strip())

        self.interpreter = self._load_interpreter(model_path)
        details = {len(d["shape"]): d for d in self.interpreter.get_input_details()}
        self.image_input, self.sensor_input = details[4], details[2]
        self.output = self.interpreter.get_output_details()[0]
        self._lock = threading.Lock()   # one interpreter, many requests

    @staticmethod
    def _load_interpreter(model_path: Path):
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter
        interp = Interpreter(model_path=str(model_path))
        for d in interp.get_input_details():
            shape = d["shape"].copy()
            shape[0] = 1
            interp.resize_tensor_input(d["index"], shape, strict=False)
        interp.allocate_tensors()
        return interp

    @staticmethod
    def _quantize(detail, value: np.ndarray) -> np.ndarray:
        if detail["dtype"] in (np.int8, np.uint8):
            scale, zero = detail["quantization"]
            info = np.iinfo(detail["dtype"])
            value = np.round(value / scale + zero).clip(info.min, info.max)
        return value.astype(detail["dtype"])

    def predict(self, image: np.ndarray, sensor_scaled: np.ndarray) -> float:
        with self._lock:
            self.interpreter.set_tensor(self.image_input["index"],
                                        self._quantize(self.image_input, image))
            self.interpreter.set_tensor(self.sensor_input["index"],
                                        self._quantize(self.sensor_input, sensor_scaled))
            self.interpreter.invoke()
            raw = self.interpreter.get_tensor(self.output["index"])
        if self.output["dtype"] in (np.int8, np.uint8):
            scale, zero = self.output["quantization"]
            raw = (raw.astype("float32") - zero) * scale
        return float(np.asarray(raw).ravel()[0])


# --------------------------------------------------------------------------------------
# Session store
# --------------------------------------------------------------------------------------
class SessionStore:
    """One rolling-feature builder per food batch, with idle expiry."""

    def __init__(self, bounds: dict, ttl_seconds: int = 7 * 24 * 3600):
        self.bounds = bounds
        self.ttl = ttl_seconds
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> dict:
        """Return the session entry, creating it if needed. Does not count a reading."""
        now = time.time()
        with self._lock:
            self._sessions = {k: v for k, v in self._sessions.items()
                              if now - v["last_seen"] < self.ttl}
            entry = self._sessions.get(session_id)
            if entry is None:
                entry = {"builder": SensorFeatureBuilder(bounds=self.bounds),
                         "count": 0, "started": now, "last_seen": now,
                         "lock": threading.Lock()}
                self._sessions[session_id] = entry
            entry["last_seen"] = now
            return entry

    def reset(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def listing(self) -> list[dict]:
        with self._lock:
            return [{"session_id": k, "readings": v["count"],
                     "age_seconds": round(time.time() - v["started"], 1)}
                    for k, v in self._sessions.items()]


def band_for(rsl: float) -> tuple[str, str]:
    for threshold, status, message in BANDS:
        if rsl >= threshold:
            return status, message
    return BANDS[-1][1], BANDS[-1][2]


def decode_image(raw: bytes):
    """Training-faithful decode: LANCZOS + JPEG q95 + TF-identical IDCT (see image_io.py).

    The first version used PIL with BILINEAR resizing, which cost 0.014 days of MAE on
    224x224 uploads and 0.055 days on full-resolution camera frames.
    """
    try:
        return load_image_bytes(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------
MODEL: "RSLModel | None" = None
SESSIONS: "SessionStore | None" = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the model once, at startup, not on the first request."""
    global MODEL, SESSIONS
    MODEL = RSLModel(Path(application.state.model_path))
    SESSIONS = SessionStore(MODEL.bounds)
    print(f"loaded {MODEL.model_path.name}  "
          f"({len(MODEL.feature_names)} sensor features) - ready")
    yield


app = FastAPI(title="Remaining Shelf Life API", version="1.0", lifespan=lifespan)

# Your website is served from a different origin than this API, so the browser needs
# permission to call it. Replace ["*"] with your site's origin before going public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL.model_path.name,
        "sensor_features": len(MODEL.feature_names),
        "image_size": list(IMAGE_SIZE),
        "open_sessions": len(SESSIONS.listing()),
    }


@app.get("/api/sessions")
def sessions() -> dict:
    return {"sessions": SESSIONS.listing()}


@app.post("/api/session/reset")
def reset_session(session_id: str = Form("default")) -> dict:
    return {"session_id": session_id, "existed": SESSIONS.reset(session_id)}


@app.post("/api/predict")
async def predict(
    image: UploadFile = File(..., description="JPEG/PNG photo of the food"),
    temperature: float = Form(..., description="deg C"),
    humidity: float = Form(..., description="% RH"),
    gas: float = Form(..., description="gas sensor reading"),
    minutes_since_previous: float = Form(2.0),
    session_id: str = Form("default"),
) -> JSONResponse:
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "empty image upload")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"image larger than {MAX_UPLOAD_BYTES // 1024 // 1024} MB")

    loaded = decode_image(raw)
    pixels = loaded.pixels
    warnings: list[str] = list(loaded.notes)
    if session_id == "default":
        warnings.append(
            "No session_id was sent, so this reading joined the shared 'default' session. "
            "Readings from different food batches would mix their rolling history - send a "
            "stable id per batch."
        )

    entry = SESSIONS.get(session_id)
    readings = {"Temperature": temperature, "Humidity": humidity, "Gas": gas}
    with entry["lock"]:
        builder = entry["builder"]
        snapshot = builder.state_dict()
        try:
            vector = builder.vector(readings, minutes_since_previous / 60.0, MODEL.feature_names)
            scaled = MODEL.scaler.transform(vector.astype("float64")).astype("float32")
            started = time.perf_counter()
            rsl = MODEL.predict(pixels, scaled)
            elapsed_ms = (time.perf_counter() - started) * 1000
        except Exception:
            # Leave the session exactly as it was, so a failed request does not advance
            # the rolling window.
            entry["builder"] = SensorFeatureBuilder.from_state(snapshot, SESSIONS.bounds)
            raise
        entry["count"] += 1
        reading_number = entry["count"]
        adjustments = dict(builder.last_adjustments)

    rsl_clamped = float(np.clip(rsl, 0.0, MODEL.max_shelf_life))
    status, message = band_for(rsl_clamped)

    if reading_number == 1:
        warnings.append(
            "First reading of this session: the rolling sensor features have no history "
            "yet, so this prediction leans almost entirely on the image. Accuracy improves "
            "from the second reading onward."
        )
    for channel, value in readings.items():
        env = MODEL.envelope.get(channel)
        if env and not (env["min"] <= value <= env["max"]):
            warnings.append(
                f"UNRELIABLE: {channel} = {value} is outside every reading the model was "
                f"trained on ({env['min']:g} to {env['max']:g}). The raw value still reaches "
                f"the network, so this prediction is an extrapolation."
            )
    for channel, change in adjustments.items():
        how = ("clipped to the outlier bound" if change["how"] == "clipped"
               else "replaced by the last in-range reading")
        warnings.append(
            f"{channel} = {change['raw']} is an outlier; its smoothed and rolling features used "
            f"{change['used']:.2f} ({how}), as training did."
        )
    if rsl != rsl_clamped:
        warnings.append(f"raw model output {rsl:.2f} clamped to the valid 0-"
                        f"{MODEL.max_shelf_life:.0f} day range")

    return JSONResponse({
        "remaining_shelf_life_days": round(rsl_clamped, 2),
        "status": status,
        "message": message,
        "session_id": session_id,
        "reading_number": reading_number,
        "readings": readings,
        "inference_ms": round(elapsed_ms, 1),
        "model": MODEL.model_path.name,
        "image_decoder": loaded.decoder,
        "raw_model_output": round(rsl, 4),
        "warnings": warnings,
    })


# --------------------------------------------------------------------------------------
# Built-in test page
# --------------------------------------------------------------------------------------
TEST_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RSL model - test page</title>
<style>
  :root { color-scheme: light dark; --bg:#fbfbf9; --fg:#1c1c1a; --muted:#6b6b66;
          --line:#e2e1dc; --card:#fff; --accent:#2f6f9f; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#16161a; --fg:#eceae6; --muted:#9a988f; --line:#2e2e34; --card:#1e1e23; }
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 ui-sans-serif,
         system-ui,-apple-system,"Segoe UI",sans-serif; padding:2rem 1rem; }
  main { max-width:760px; margin:0 auto }
  h1 { font-size:1.35rem; margin:0 0 .25rem }
  p.sub { color:var(--muted); margin:0 0 1.5rem }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:1.25rem; margin-bottom:1rem }
  label { display:block; font-weight:600; font-size:.82rem; margin-bottom:.3rem }
  input { width:100%; padding:.5rem .6rem; border:1px solid var(--line); border-radius:6px;
          background:var(--bg); color:var(--fg); font:inherit }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.9rem }
  button { margin-top:1rem; padding:.6rem 1.2rem; border:0; border-radius:6px;
           background:var(--accent); color:#fff; font:inherit; font-weight:600; cursor:pointer }
  button:disabled { opacity:.5; cursor:default }
  #preview { max-width:220px; border-radius:8px; margin-top:.8rem; display:none }
  .rsl { font-size:2.6rem; font-weight:700; margin:.2rem 0 }
  .pill { display:inline-block; padding:.15rem .6rem; border-radius:99px; font-size:.78rem;
          font-weight:600; border:1px solid var(--line) }
  .fresh { background:#e7f4ea; color:#1d6b34 } .use_soon { background:#fdf3e0; color:#8a5a10 }
  .expiring { background:#fbe9e9; color:#912b2b }
  pre { background:var(--bg); border:1px solid var(--line); border-radius:6px; padding:.8rem;
        overflow-x:auto; font-size:.78rem; margin:.8rem 0 0 }
  .warn { border-left:3px solid #d3a24a; padding:.5rem .75rem; margin-top:.7rem;
          font-size:.85rem; color:var(--muted) }
  code { font-size:.85em }
</style></head><body><main>
<h1>Remaining Shelf Life — test page</h1>
<p class="sub">Upload a food photo, enter the three sensor readings, get an RSL estimate.
This page calls the same <code>/api/predict</code> endpoint your website should call.</p>

<form class="card" id="form">
  <label for="image">Photo</label>
  <input type="file" id="image" accept="image/*" required>
  <img id="preview" alt="">
  <div class="grid" style="margin-top:1rem">
    <div><label for="temperature">Temperature (°C)</label>
      <input type="number" step="0.01" id="temperature" value="28.74" required></div>
    <div><label for="humidity">Humidity (% RH)</label>
      <input type="number" step="0.01" id="humidity" value="58.04" required></div>
    <div><label for="gas">Gas</label>
      <input type="number" step="1" id="gas" value="185" required></div>
    <div><label for="session_id">Session / batch id</label>
      <input type="text" id="session_id" value="test-batch"></div>
  </div>
  <button type="submit" id="go">Estimate shelf life</button>
</form>

<div class="card" id="result" style="display:none"></div>
</main><script>
const $ = id => document.getElementById(id);
$("image").addEventListener("change", e => {
  const f = e.target.files[0];
  if (!f) return;
  $("preview").src = URL.createObjectURL(f);
  $("preview").style.display = "block";
});
$("form").addEventListener("submit", async e => {
  e.preventDefault();
  const file = $("image").files[0];
  if (!file) return;
  $("go").disabled = true; $("go").textContent = "Estimating…";
  const fd = new FormData();
  fd.append("image", file);
  for (const k of ["temperature", "humidity", "gas", "session_id"]) fd.append(k, $(k).value);
  try {
    const res = await fetch("/api/predict", { method: "POST", body: fd });
    const data = await res.json();
    const box = $("result");
    box.style.display = "block";
    if (!res.ok) { box.innerHTML = "<strong>Error</strong><pre>" +
        JSON.stringify(data, null, 2) + "</pre>"; return; }
    box.innerHTML =
      '<span class="pill ' + data.status + '">' + data.status.replace("_", " ") + "</span>" +
      '<div class="rsl">' + data.remaining_shelf_life_days + " <span style='font-size:1rem;font-weight:400'>days left</span></div>" +
      "<div style='color:var(--muted)'>" + data.message + " · reading #" + data.reading_number +
      " of session <code>" + data.session_id + "</code> · " + data.inference_ms + " ms</div>" +
      (data.warnings || []).map(w => '<div class="warn">' + w + "</div>").join("") +
      "<pre>" + JSON.stringify(data, null, 2) + "</pre>";
  } catch (err) {
    $("result").style.display = "block";
    $("result").innerHTML = "<strong>Could not reach the API.</strong><pre>" + err + "</pre>";
  } finally { $("go").disabled = false; $("go").textContent = "Estimate shelf life"; }
});
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def test_page() -> str:
    return TEST_PAGE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1",
                    help="use 0.0.0.0 to accept connections from other machines")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = ap.parse_args()

    app.state.model_path = str(args.model)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
else:
    app.state.model_path = str(DEFAULT_MODEL)
