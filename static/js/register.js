/* Client-side hints for the registration form. The server re-validates
   everything: this only saves the user a round trip. */
import { toast } from "./core.js";

const form = document.getElementById("registerForm");
const password = document.getElementById("password");
const confirm = document.getElementById("confirm_password");
const bar = document.getElementById("strengthBar");
const text = document.getElementById("strengthText");

const WEAK = new Set([
    "password", "password1", "password123", "12345678", "123456789",
    "1234567890", "qwerty123", "abc12345", "iloveyou", "admin123",
    "welcome1", "letmein1", "11111111", "00000000", "passw0rd",
]);

function score(value) {
    if (!value) return { pct: 0, tone: "muted", label: "At least 8 characters." };
    if (WEAK.has(value.toLowerCase())) {
        return { pct: 15, tone: "crit", label: "Too common - pick something less guessable." };
    }
    if (value.length < 8) return { pct: 20, tone: "crit", label: "Too short - use 8 or more characters." };
    if (/^\d+$/.test(value)) return { pct: 30, tone: "crit", label: "Digits only is easy to guess." };

    let points = 0;
    if (value.length >= 12) points += 1;
    if (/[a-z]/.test(value) && /[A-Z]/.test(value)) points += 1;
    if (/\d/.test(value)) points += 1;
    if (/[^A-Za-z0-9]/.test(value)) points += 1;

    if (points <= 1) return { pct: 45, tone: "warn", label: "Acceptable - mixing case or symbols would help." };
    if (points === 2) return { pct: 70, tone: "warn", label: "Good." };
    return { pct: 100, tone: "ok", label: "Strong." };
}

function update() {
    const result = score(password.value);
    bar.style.width = `${result.pct}%`;
    bar.className = `meter__fill meter__fill--${result.tone}`;
    text.textContent = result.label;
}

password?.addEventListener("input", update);

confirm?.addEventListener("input", () => {
    confirm.setCustomValidity(
        confirm.value && confirm.value !== password.value ? "Passwords do not match." : ""
    );
});

form?.addEventListener("submit", (event) => {
    const email = form.querySelector("input[name='email']").value.trim();
    const mobile = form.querySelector("input[name='mobile']").value.trim();
    if (!email && !mobile) {
        event.preventDefault();
        toast("Enter an email address or a mobile number.", "error");
        return;
    }
    if (password.value !== confirm.value) {
        event.preventDefault();
        toast("Passwords do not match.", "error");
    }
});
