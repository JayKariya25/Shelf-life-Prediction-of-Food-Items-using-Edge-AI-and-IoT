/* Device page: register devices, reveal the token once, and remove devices. */
import { api, closeModal, openModal, toast } from "./core.js";

document.getElementById("addDeviceForm")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const submit = form.querySelector('button[type="submit"]');
    const label = submit.innerHTML;
    submit.disabled = true;
    submit.innerHTML = '<span class="spinner"></span>Creating';
    try {
        const data = new FormData(form);
        const result = await api("/api/v1/devices", {
            method: "POST",
            body: {
                name: String(data.get("name") || "").trim(),
                location: String(data.get("location") || "").trim(),
            },
        });
        // The plaintext token exists only in this response. Show it in a modal
        // rather than putting it in the URL, where it would land in history,
        // the referrer header and any proxy log.
        closeModal("addDeviceModal");
        const holder = document.getElementById("tokenValue");
        if (holder) holder.textContent = result.token;
        openModal("tokenModal");
    } catch (error) {
        toast(error.message || "Could not register the device.", "error");
    } finally {
        submit.disabled = false;
        submit.innerHTML = label;
    }
});

document.getElementById("copyTokenBtn")?.addEventListener("click", async () => {
    const token = document.getElementById("tokenValue")?.textContent || "";
    try {
        await navigator.clipboard.writeText(token);
        toast("Token copied to the clipboard.");
    } catch {
        // Clipboard access needs a secure context; plain HTTP on the Pi has none.
        toast("Copy blocked by the browser - select the token and copy it manually.", "warn");
    }
});

document.getElementById("tokenDoneBtn")?.addEventListener("click", () => {
    closeModal("tokenModal");
    window.location.reload();
});

document.querySelectorAll("[data-delete-device]").forEach((button) => {
    button.addEventListener("click", async () => {
        if (!window.confirm("Remove this device? Its token stops working immediately.")) return;
        button.disabled = true;
        try {
            await api(`/api/v1/devices/${button.dataset.deleteDevice}`, { method: "DELETE" });
            button.closest("tr")?.remove();
            toast("Device removed.");
        } catch (error) {
            toast(error.message || "Could not remove the device.", "error");
            button.disabled = false;
        }
    });
});
