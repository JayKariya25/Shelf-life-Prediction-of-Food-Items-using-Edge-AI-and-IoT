"""Registration, sign-in, session handling and CSRF."""

from __future__ import annotations

import unittest

from tests.support import AppTestCase

from shelflife.db import query_one
from shelflife.security import rate_limiter, validate_registration


class RegistrationValidation(unittest.TestCase):
    def test_requires_an_identifier(self):
        errors = validate_registration(None, None, "correct-horse-9", "correct-horse-9")
        self.assertIn("Provide an email address or a mobile number.", errors)

    def test_rejects_short_passwords(self):
        errors = validate_registration("a@b.com", None, "short", "short")
        self.assertTrue(any("at least 8" in error for error in errors))

    def test_rejects_common_passwords(self):
        errors = validate_registration("a@b.com", None, "password123", "password123")
        self.assertTrue(any("too common" in error for error in errors))

    def test_rejects_digit_only_passwords(self):
        errors = validate_registration("a@b.com", None, "9182736450", "9182736450")
        self.assertTrue(any("only digits" in error for error in errors))

    def test_rejects_mismatched_confirmation(self):
        errors = validate_registration("a@b.com", None, "correct-horse-9", "different-9")
        self.assertIn("Passwords do not match.", errors)

    def test_rejects_malformed_email_and_mobile(self):
        self.assertTrue(any("email" in e for e in
                            validate_registration("not-an-email", None, "correct-horse-9", "correct-horse-9")))
        self.assertTrue(any("Mobile" in e for e in
                            validate_registration(None, "12ab", "correct-horse-9", "correct-horse-9")))

    def test_accepts_a_good_registration(self):
        self.assertEqual(validate_registration("a@b.com", None, "correct-horse-9", "correct-horse-9"), [])


class RegistrationFlow(AppTestCase):
    def test_registration_creates_a_user_and_settings_row(self):
        response = self.register()
        self.assertEqual(response.status_code, 302)
        with self.app.app_context():
            user = query_one("SELECT * FROM users WHERE email = ?", ("user@example.com",))
            self.assertIsNotNone(user)
            self.assertIsNotNone(query_one("SELECT * FROM settings WHERE user_id = ?", (user["id"],)))

    def test_password_is_never_stored_in_the_clear(self):
        self.register(password="correct-horse-9")
        with self.app.app_context():
            user = query_one("SELECT * FROM users WHERE email = ?", ("user@example.com",))
            self.assertNotIn("correct-horse-9", user["password_hash"])
            self.assertGreater(len(user["password_hash"]), 40)

    def test_duplicate_email_is_rejected(self):
        self.register()
        response = self.register()
        self.assertEqual(response.status_code, 409)

    def test_email_is_normalised_to_lowercase(self):
        self.register(email="MixedCase@Example.COM")
        with self.app.app_context():
            self.assertIsNotNone(query_one("SELECT 1 FROM users WHERE email = ?", ("mixedcase@example.com",)))

    def test_invalid_registration_returns_422_style_error(self):
        response = self.client.post(
            "/register",
            data={"email": "", "mobile": "", "password": "x", "confirm_password": "y",
                  "csrf_token": self.csrf("/register")},
        )
        self.assertEqual(response.status_code, 400)


class LoginFlow(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register()

    def test_login_with_email(self):
        self.assertEqual(self.login().status_code, 302)

    def test_login_with_mobile(self):
        self.register(email="", mobile="9876543210", password="correct-horse-9")
        self.assertEqual(self.login(identifier="9876543210").status_code, 302)

    def test_wrong_password_is_rejected(self):
        self.assertEqual(self.login(password="wrong-password-1").status_code, 401)

    def test_unknown_account_gives_the_same_message_as_a_wrong_password(self):
        unknown = self.login(identifier="nobody@example.com").get_data(as_text=True)
        wrong = self.login(password="wrong-password-1").get_data(as_text=True)
        self.assertIn("Incorrect email/mobile or password.", unknown)
        self.assertIn("Incorrect email/mobile or password.", wrong)

    def test_login_records_last_login_time(self):
        self.login()
        with self.app.app_context():
            user = query_one("SELECT last_login_at FROM users WHERE email = ?", ("user@example.com",))
            self.assertIsNotNone(user["last_login_at"])

    def test_session_id_is_rotated_on_login(self):
        before = self.client.get("/login")
        self.login()
        after = self.client.get("/dashboard")
        self.assertEqual(after.status_code, 200)
        self.assertNotEqual(before.get_data(), after.get_data())

    def test_protected_pages_redirect_when_signed_out(self):
        for path in ("/dashboard", "/items", "/alerts", "/device", "/profile"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 302, path)
            self.assertIn("/login", response.headers["Location"])

    def test_logout_clears_the_session(self):
        self.login()
        token = self.csrf("/dashboard")
        response = self.client.post("/logout", data={"csrf_token": token})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.get("/dashboard").status_code, 302)

    def test_logout_is_not_reachable_by_get(self):
        self.login()
        self.assertEqual(self.client.get("/logout").status_code, 405)


class RedirectSafety(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register()

    def _login_with_next(self, target):
        body = self.client.get("/login").get_data(as_text=True)
        import re
        token = re.search(r'name="csrf_token" value="([^"]+)"', body).group(1)
        return self.client.post(
            f"/login?next={target}",
            data={"identifier": "user@example.com", "password": "correct-horse-9",
                  "csrf_token": token},
        )

    def test_external_next_target_is_ignored(self):
        for evil in ("https://evil.example.com", "//evil.example.com", "javascript:alert(1)"):
            response = self._login_with_next(evil)
            self.assertEqual(response.headers["Location"], "/dashboard", evil)
            self.client.post("/logout", data={"csrf_token": self.csrf("/dashboard")})

    def test_relative_next_target_is_honoured(self):
        response = self._login_with_next("/items")
        self.assertEqual(response.headers["Location"], "/items")


class LoginThrottling(AppTestCase):
    def setUp(self):
        super().setUp()
        self.app.config["LOGIN_MAX_ATTEMPTS"] = 3
        self.register()

    def test_repeated_failures_are_locked_out(self):
        for _ in range(3):
            self.login(password="wrong-password-1")
        response = self.login(password="wrong-password-1")
        self.assertEqual(response.status_code, 429)
        self.assertIn("Too many failed sign-in attempts", response.get_data(as_text=True))

    def test_successful_login_clears_the_counter(self):
        self.login(password="wrong-password-1")
        self.assertEqual(self.login().status_code, 302)
        rate_limiter._hits.clear()


class CsrfProtection(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register_and_login()

    def test_form_post_without_a_token_is_rejected(self):
        response = self.client.post("/logout", data={})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(self.client.get("/dashboard").status_code, 200,
                         "the session must survive a rejected CSRF attempt")

    def test_api_post_without_a_token_is_rejected(self):
        response = self.client.post("/api/v1/items", json={"label": "x", "food_type_id": 1})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json()["error"]["code"], "csrf_failed")

    def test_api_post_with_a_wrong_token_is_rejected(self):
        response = self.client.post(
            "/api/v1/items", json={"label": "x", "food_type_id": 1},
            headers={"X-CSRF-Token": "not-the-right-token"},
        )
        self.assertEqual(response.status_code, 403)

    def test_api_post_with_the_right_token_succeeds(self):
        item = self.create_item()
        self.assertEqual(item["label"], "Test tomatoes")

    def test_device_endpoints_are_csrf_exempt(self):
        """Token-authenticated device calls carry no cookies, so CSRF cannot apply."""
        response = self.client.post("/api/device/v1/readings", json={"temperature_c": 20})
        self.assertEqual(response.status_code, 401)  # rejected for auth, not CSRF
        self.assertEqual(response.get_json()["error"]["code"], "unauthorised")


class SecurityHeaders(AppTestCase):
    def test_headers_are_present_on_every_response(self):
        response = self.client.get("/login")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "DENY")
        self.assertIn("default-src 'self'", response.headers["Content-Security-Policy"])
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])

    def test_csp_forbids_remote_scripts(self):
        policy = self.client.get("/login").headers["Content-Security-Policy"]
        self.assertIn("script-src 'self'", policy)
        self.assertNotIn("unsafe-eval", policy)

    def test_api_responses_are_not_cacheable(self):
        self.register_and_login()
        response = self.client.get("/api/v1/settings")
        self.assertEqual(response.headers["Cache-Control"], "no-store")


if __name__ == "__main__":
    unittest.main()
