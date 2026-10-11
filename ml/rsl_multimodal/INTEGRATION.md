# Connecting the RSL model to your website

> **The Shelf-Life Monitor website already runs this model in-process** — it loads the
> float16 TFLite model, the scaler and `scripts/pi_features.py` / `scripts/image_io.py`
> directly, keeps each tracked item's rolling history in its database, and needs no
> separate service. This document is for any *other* frontend that wants to call the
> model over HTTP.

The model is Python — it cannot run inside the browser. The shape that works:

```
  Pi / phone camera ──┐
                      ├──> your website (frontend) ──HTTP──> serve.py ──> RSL model
  temp / humidity / gas ┘                                       │
                                                    returns {"remaining_shelf_life_days": 9.59}
```

`serve.py` is that middle service. Start it, and your website calls one endpoint.

---

## 1. Start the API

`serve.py` is the reference API for the **fusion** export: it takes an image plus
Temperature/Humidity/Gas and keeps a session's rolling features. It was written before the
image-only ensemble became the recommended artifact and has not been ported to it, so it
still loads `fusion_blocked_full_float16.tflite`. For the recommended export, use
`scripts/predict_pi.py --image capture.jpg` (no readings needed), or the monitoring
website, which serves it over HTTP and supports both contracts.

```bash
cd ~/Downloads/"CAPSTONE DATASET"/RSL_Multimodal_Model
source .venv/bin/activate
pip install -r requirements.txt
python serve.py
```

```
loaded fusion_blocked_full_float16.tflite  (27 sensor features) - ready
Uvicorn running on http://127.0.0.1:8001

(Port 8001, so it does not collide with the monitoring website on 8000.)
```

The model loads once at startup (~10 s), then each prediction takes about **3 ms**.

## 2. Test it with no code at all

Open **http://127.0.0.1:8001** in your browser. There is a built-in page: pick a photo,
type the three readings, press the button, see the RSL. Use it to sanity-check the model
before wiring anything up.

To test from the terminal instead:

```bash
curl -X POST http://127.0.0.1:8001/api/predict \
  -F "image=@data_images/20260606_200106.jpg" \
  -F "temperature=28.74" -F "humidity=58.04" -F "gas=185" \
  -F "session_id=tray-1"
```

```json
{
  "remaining_shelf_life_days": 9.59,
  "status": "fresh",
  "message": "Good for several more days",
  "session_id": "tray-1",
  "reading_number": 1,
  "inference_ms": 3.2,
  "image_decoder": "tensorflow",
  "raw_model_output": 9.5904,
  "warnings": ["First reading of this session: ..."]
}
```

---

## 3. Call it from your website

### JavaScript (any frontend — React, Vue, plain HTML)

```js
async function estimateShelfLife(imageFile, temperature, humidity, gas, batchId) {
  const form = new FormData();
  form.append("image", imageFile);          // File from <input type="file"> or a canvas blob
  form.append("temperature", temperature);
  form.append("humidity", humidity);
  form.append("gas", gas);
  form.append("session_id", batchId);       // stable per food batch — see section 4

  const res = await fetch("http://127.0.0.1:8001/api/predict", {
    method: "POST",
    body: form,                              // do NOT set Content-Type; the browser does it
  });
  if (!res.ok) throw new Error((await res.json()).detail);
  return res.json();                         // { remaining_shelf_life_days, status, ... }
}
```

### Python backend (Flask / Django proxying to the model)

```python
import requests

def estimate(image_bytes, temperature, humidity, gas, batch_id):
    r = requests.post(
        "http://127.0.0.1:8001/api/predict",
        files={"image": ("capture.jpg", image_bytes, "image/jpeg")},
        data={"temperature": temperature, "humidity": humidity,
              "gas": gas, "session_id": batch_id},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()
```

### From the Raspberry Pi itself

```bash
curl -X POST http://<server-ip>:8001/api/predict \
  -F "image=@$(ls -t /home/pi/captures/*.jpg | head -1)" \
  -F "temperature=$TEMP" -F "humidity=$HUM" -F "gas=$GAS" \
  -F "session_id=pi-01"
```

Or skip the server entirely and run the model on the Pi — see `QUICKSTART.md` section 6.

---

## 4. `session_id` — the one thing that is easy to get wrong

The sensor branch does not only look at the current reading. It uses **change since the
last reading, rate of change, a 5-sample rolling mean and standard deviation, and drift
since monitoring began.** Those need history.

- Pass a **stable id per food batch** — a tray id, a device id, an order number.
- Send readings for that batch to the **same** `session_id` every time, in time order.
- Start a **new** id when you start monitoring a new batch (or `POST /api/session/reset`).

If every request uses a fresh id, every prediction is a cold start: all the change and
rate features are zero, and you are effectively running the image branch alone. It still
returns a number, and the response says so in `warnings` — but you lose the sensor half.

Pass `minutes_since_previous` if your sampling interval is not 2 minutes.

---

## 5. What the response means

| Field | Meaning |
|---|---|
| `remaining_shelf_life_days` | The prediction, clamped to 0–11 days |
| `status` | `fresh` (≥5 d), `use_soon` (2–5 d), `expiring` (<2 d) — bands are in `serve.py`, change them freely |
| `reading_number` | How many readings this session has seen. 1 = cold start |
| `warnings` | Cold start, missing `session_id`, `UNRELIABLE` out-of-envelope readings, outlier cleaning, clamped output. **Show or log these** |
| `raw_model_output` | The unclamped network output, for debugging |
| `image_decoder` | `tensorflow` or `simplejpeg` = exact; `pil` = approximate (install `simplejpeg`) |
| `inference_ms` | Model time only, excluding upload |

---

## 6. Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Built-in test page |
| GET | `/api/health` | Model name, feature count, open sessions — for uptime checks |
| POST | `/api/predict` | The prediction call |
| POST | `/api/session/reset` | Clear one session's history (form field `session_id`) |
| GET | `/api/sessions` | List open sessions |
| GET | `/docs` | Auto-generated interactive API docs (FastAPI) |

---

## 7. Before you show this to real users

**Accuracy depends on what the model has seen.** For the fusion export this file serves:
about **0.27 days** within a batch already being monitored, and about **1.17 days** on a
batch it has never seen. For the recommended image-only ensemble: **0.15** and **0.85**.
Display a range, not a false-precision number — "about 9 days" beats "9.59 days".

**It was trained on one food item, one camera, one 11-day run.** Photos of a different
food, or from a different camera or lighting setup, are outside what it has seen and the
prediction is not trustworthy. This is the honest limitation to state in your report.

**Out-of-range sensor readings are flagged, not rejected.** Every raw reading the model was
trained on sits in 25.8–29.3 °C, 53–67 % RH and gas 152–193. Outside that the API still
answers, but the raw value reaches the network and the output is extrapolation — a 45 °C
cold start returns −7.4 days before clamping. Those responses carry an `UNRELIABLE`
warning: surface it, or fall back to another estimate, rather than hiding it.

**CORS is wide open** (`allow_origins=["*"]` in `serve.py`). Fine for local development.
Before deploying anywhere public, set it to your site's origin.

**There is no authentication and no rate limit.** If this goes on a public server, put it
behind your existing backend rather than exposing it to the internet directly.

**Sessions live in memory.** Restarting `serve.py` clears all rolling history. For a real
deployment, persist `builder.state_dict()` per batch to your database.

---

## 8. Running it somewhere other than your laptop

```bash
python serve.py --host 0.0.0.0 --port 8001
```

`--host 0.0.0.0` lets the Pi and other machines on your network reach it. For anything
longer-lived, run it under a process manager and put nginx in front:

```bash
pip install gunicorn
gunicorn serve:app -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001 --workers 1
```

Keep `--workers 1` unless you also move sessions out of memory — each worker would
otherwise hold its own copy of the rolling history.
