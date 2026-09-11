/* Inference playground: upload an image, supply the sensor triple, run the model. */
import { api, escapeHtml, formatNumber, toast } from "./core.js";

const form = document.getElementById("playgroundForm");
const fileInput = document.getElementById("pgImage");
const dropZone = document.getElementById("dropZone");
const prompt = document.getElementById("dropPrompt");
const preview = document.getElementById("preview");
const result = document.getElementById("pgResult");

let objectUrl = null;

function showPreview(file) {
    if (!file) return;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    preview.src = objectUrl;
    preview.hidden = false;
    prompt.hidden = true;
}

dropZone?.addEventListener("click", () => fileInput.click());
dropZone?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        fileInput.click();
    }
});
fileInput?.addEventListener("change", () => showPreview(fileInput.files[0]));

["dragenter", "dragover"].forEach((name) =>
    dropZone?.addEventListener(name, (event) => {
        event.preventDefault();
        dropZone.classList.add("is-over");
    })
);
["dragleave", "drop"].forEach((name) =>
    dropZone?.addEventListener(name, (event) => {
        event.preventDefault();
        dropZone.classList.remove("is-over");
    })
);
dropZone?.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (!file) return;
    // Assigning through DataTransfer keeps the native <input type=file> as the
    // single source of truth for the form submission.
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    showPreview(file);
});

form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = fileInput.files[0];
    if (!file) {
        toast("Choose an image first.", "error");
        return;
    }

    const button = document.getElementById("pgSubmit");
    const label = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span>Running inference';

    try {
        const payload = new FormData();
        payload.append("image", file);
        ["temperature", "humidity", "gas"].forEach((name) => {
            payload.append(name, form.querySelector(`[name="${name}"]`).value);
        });

        const data = await api("/admin/api/predict", { method: "POST", formData: payload });
        result.innerHTML = `
            <div class="stat" style="border:0;box-shadow:none;padding:0">
                <div class="stat__label">Predicted remaining shelf life</div>
                <div class="stat__value">${formatNumber(data.predicted_days, 2)}<small>days</small></div>
                <div class="stat__meta">${formatNumber(data.inference_ms, 1)} ms on this machine</div>
            </div>
            <hr class="divider">
            <dl class="kv">
                <dt>Raw model output</dt><dd class="mono">${formatNumber(data.predicted_days_raw, 4)} days</dd>
                <dt>Temperature</dt><dd>${formatNumber(data.sensor.temperature, 2)}</dd>
                <dt>Humidity</dt><dd>${formatNumber(data.sensor.humidity, 2)}</dd>
                <dt>Gas</dt><dd>${formatNumber(data.sensor.gas, 2)}</dd>
                <dt>Model</dt><dd>${escapeHtml(data.model.name)} v${escapeHtml(data.model.version)}</dd>
            </dl>
            ${data.clamped
                ? '<div class="banner banner--warn" style="margin-top:0.9rem">' +
                  '<div class="banner__body small">The raw output was negative and has been ' +
                  'floored at zero for display. The unmodified value is shown above.</div></div>'
                : ""}
            <p class="tiny muted" style="margin-top:0.9rem">
                Inference only &mdash; the saved model and scaler were not modified.
            </p>`;
    } catch (error) {
        result.innerHTML = `<div class="banner banner--crit"><div class="banner__body">
            ${escapeHtml(error.message || "Inference failed.")}</div></div>`;
    } finally {
        button.disabled = false;
        button.innerHTML = label;
    }
});
