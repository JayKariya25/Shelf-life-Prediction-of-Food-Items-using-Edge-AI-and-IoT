/* Developer console: reload the model artifacts from disk. */
import { api, toast } from "./core.js";

document.getElementById("reloadModelBtn")?.addEventListener("click", async (event) => {
    const button = event.currentTarget;
    const label = button.innerHTML;
    button.disabled = true;
    button.innerHTML = '<span class="spinner"></span>Reloading';
    try {
        const result = await api("/admin/api/reload-model", { method: "POST" });
        if (result.model.trained_model_available) {
            toast(`Loaded ${result.model.active_model}.`);
        } else {
            toast(result.model.trained_model_error || "No trained model found.", "warn", 7000);
        }
        setTimeout(() => window.location.reload(), 900);
    } catch (error) {
        toast(error.message || "Reload failed.", "error");
        button.disabled = false;
        button.innerHTML = label;
    }
});
