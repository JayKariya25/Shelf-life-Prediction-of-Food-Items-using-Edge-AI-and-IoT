/* Accounts: change roles and enable/disable accounts. */
import { api, toast } from "./core.js";

document.querySelectorAll("[data-role-select]").forEach((select) => {
    const original = select.value;
    select.dataset.previous = original;
    select.addEventListener("change", async () => {
        const id = select.dataset.roleSelect;
        select.disabled = true;
        try {
            await api(`/admin/api/users/${id}/role`, {
                method: "PATCH",
                body: { role: select.value },
            });
            select.dataset.previous = select.value;
            toast(`Account ${id} is now a ${select.value}.`);
        } catch (error) {
            select.value = select.dataset.previous; // roll the control back
            toast(error.message || "Could not change the role.", "error", 6000);
        } finally {
            select.disabled = false;
        }
    });
});

document.querySelectorAll("[data-toggle-active]").forEach((button) => {
    button.addEventListener("click", async () => {
        const id = button.dataset.toggleActive;
        const next = button.dataset.active !== "1";
        button.disabled = true;
        try {
            await api(`/admin/api/users/${id}/active`, {
                method: "PATCH",
                body: { is_active: next },
            });
            window.location.reload();
        } catch (error) {
            toast(error.message || "Could not update the account.", "error", 6000);
            button.disabled = false;
        }
    });
});
