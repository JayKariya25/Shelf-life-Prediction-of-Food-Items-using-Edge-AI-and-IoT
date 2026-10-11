/* Inference playground: add readings to a session and run the trained model. */
import { api, escapeHtml, formatNumber, toast } from "./core.js";

const form = document.getElementById("playgroundForm");
const fileInput = document.getElementById("pgImage");
const dropZone = document.getElementById("dropZone");
const prompt = document.getElementById("dropPrompt");
const preview = document.getElementById("preview");
const result = document.getElementById("pgResult");
const sessionInput = document.getElementById("pgSession");

let objectUrl = null;
const history = [];

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
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    showPreview(file);
});

function renderHistory() {
    if (!history.length) return "";
    const rows = history
        .slice(-8)
        .map(
            (h) => `<tr>
                <td>#${h.reading}</td>
                <td class="table__num">${formatNumber(h.temperature, 2)}</td>
                <td class="table__num">${formatNumber(h.humidity, 2)}</td>
                <td class="table__num">${formatNumber(h.gas, 1)}</td>
                <td class="table__num strong">${formatNumber(h.days, 2)}</td>
            </tr>`
        )
        .join("");
    return `
        <hr class="divider">
        <h4>This session</h4>
        <div class="table-wrap" style="margin-top:0.5rem">
            <table class="table">
                <thead><tr><th>Reading</th><th class="table__num">T</th><th class="table__num">RH</th>
                    <th class="table__num">Gas</th><th class="table__num">Days</th></tr></thead>
                <tbody>${rows}</tbody>
            </table>
        </div>`;
}

function render(data) {
    const band = data.new_batch_mae_days;
    const warnings = (data.warnings || [])
        .map((w) => `<li>${escapeHtml(w)}</li>`)
        .join("");
    const outside = Object.keys(data.out_of_envelope || {}).length > 0;

    result.innerHTML = `
        ${outside
            ? `<div class="banner banner--crit" style="margin-bottom:0.9rem">
                 <div class="banner__body small"><div class="banner__title">Outside the training range</div>
                 This estimate is an extrapolation and should not be trusted.</div></div>`
            : ""}
        <div class="stat" style="border:0;box-shadow:none;padding:0">
            <div class="stat__label">Predicted remaining shelf life</div>
            <div class="stat__value">${formatNumber(data.predicted_days, 2)}<small>days</small></div>
            <div class="stat__meta">
                ${band != null ? `about &plusmn;${formatNumber(band, 2)} days on a new batch &middot; ` : ""}
                reading #${data.reading_number} of session <span class="mono">${escapeHtml(data.session)}</span>
            </div>
        </div>
        ${warnings ? `<ul class="model-notes">${warnings}</ul>` : ""}
        <hr class="divider">
        <dl class="kv">
            <dt>Raw model output</dt><dd class="mono">${formatNumber(data.predicted_days_raw, 4)} days${data.clamped ? " (clamped)" : ""}</dd>
            <dt>Inference</dt><dd>${formatNumber(data.inference_ms, 2)} ms</dd>
            <dt>Image decoder</dt><dd>${escapeHtml(data.decoder)}</dd>
            <dt>Model</dt><dd class="tiny">${escapeHtml(data.model.name)} &middot; ${escapeHtml(data.model.version)}</dd>
        </dl>
        ${renderHistory()}
        <p class="tiny muted" style="margin-top:0.9rem">
            Inference only &mdash; nothing is saved and the model is not modified.
        </p>`;
}

form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const file = fileInput.files[0];
    if (!file) {
        toast("Choose a photo first.", "error");
        return;
    }

    const button = document.getElementById("pgSubmit");
    const label = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span>Running';

    try {
        const payload = new FormData();
        payload.append("image", file);
        for (const name of ["temperature", "humidity", "gas", "session", "minutes_since_previous"]) {
            payload.append(name, form.querySelector(`[name="${name}"]`).value);
        }
        const data = await api("/admin/api/predict", { method: "POST", formData: payload });
        if (data.reading_number === 1) history.length = 0;
        history.push({
            reading: data.reading_number,
            temperature: data.sensor.temperature,
            humidity: data.sensor.humidity,
            gas: data.sensor.gas,
            days: data.predicted_days,
        });
        render(data);
    } catch (error) {
        result.innerHTML = `<div class="banner banner--crit"><div class="banner__body">
            ${escapeHtml(error.message || "Inference failed.")}</div></div>`;
    } finally {
        button.disabled = false;
        button.innerHTML = label;
    }
});

document.getElementById("pgReset")?.addEventListener("click", async () => {
    try {
        await api("/admin/api/playground/reset", {
            method: "POST",
            body: { session: sessionInput.value || "playground" },
        });
        history.length = 0;
        result.innerHTML = '<p class="small muted">New session started. The next reading is a cold start.</p>';
        toast("Session cleared.");
    } catch (error) {
        toast(error.message || "Could not reset the session.", "error");
    }
});

sessionInput?.addEventListener("change", () => {
    history.length = 0;
});
