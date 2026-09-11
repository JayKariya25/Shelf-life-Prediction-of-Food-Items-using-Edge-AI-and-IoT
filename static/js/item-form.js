/* Shared "add item" modal wiring, used by both the dashboard and the items page. */
import { api, closeModal, toast } from "./core.js";

/**
 * @param {string} formId
 * @param {(item: object) => void} onCreated
 */
export function wireAddItemForm(formId, onCreated) {
    const form = document.getElementById(formId);
    if (!form) return;

    const submit = form.querySelector('button[type="submit"]');
    const originalLabel = submit ? submit.innerHTML : "";

    // Default "stored since" to now, in the browser's own timezone.
    const storedAt = form.querySelector('input[name="stored_at"]');
    if (storedAt && !storedAt.value) {
        const now = new Date();
        now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
        storedAt.value = now.toISOString().slice(0, 16);
    }

    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        if (submit) {
            submit.disabled = true;
            submit.innerHTML = '<span class="spinner"></span>Saving';
        }

        try {
            const data = new FormData(form);
            const label = String(data.get("label") || "").trim();
            if (!label) {
                toast("Give the item a name.", "error");
                return;
            }

            // Upload the photo first so the item row can reference it.
            let imagePath = null;
            const file = data.get("image");
            if (file && file.size > 0) {
                const upload = new FormData();
                upload.append("image", file);
                const result = await api("/api/v1/uploads", { method: "POST", formData: upload });
                imagePath = result.image_path;
            }

            const rawStoredAt = String(data.get("stored_at") || "").trim();
            const payload = {
                label,
                food_type_id: Number(data.get("food_type_id")),
                quantity: String(data.get("quantity") || "").trim(),
                storage: String(data.get("storage") || "room"),
                notes: String(data.get("notes") || "").trim(),
                image_path: imagePath,
            };
            if (rawStoredAt) {
                // datetime-local has no timezone; interpret it as local time.
                payload.stored_at = new Date(rawStoredAt).toISOString();
            }

            const created = await api("/api/v1/items", { method: "POST", body: payload });
            toast(`Now tracking ${created.item.label}.`);
            form.reset();
            closeModal(form.closest(".modal"));
            if (typeof onCreated === "function") onCreated(created.item);
        } catch (error) {
            toast(error.message || "Could not save the item.", "error");
        } finally {
            if (submit) {
                submit.disabled = false;
                submit.innerHTML = originalLabel;
            }
        }
    });
}
