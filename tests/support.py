"""Shared test fixtures."""

from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("SECRET_KEY", "test-secret-key-not-used-in-production")

from shelflife import create_app                                  # noqa: E402
from shelflife.db import execute, iso_now, query_one               # noqa: E402
from shelflife.security import rate_limiter                        # noqa: E402

CSRF_RE = re.compile(r'name="csrf_token" value="([^"]+)"')


class AppTestCase(unittest.TestCase):
    """Base case: a throwaway app backed by a temporary SQLite file."""

    sensor_source = "simulated"

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.app = create_app(
            "testing",
            DATABASE_PATH=str(root / "test.db"),
            UPLOAD_DIR=str(root / "uploads"),
            SENSOR_SOURCE=self.sensor_source,
        )
        os.makedirs(self.app.config["UPLOAD_DIR"], exist_ok=True)
        self.client = self.app.test_client()
        rate_limiter._hits.clear()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # --- helpers ---------------------------------------------------------
    def csrf(self, path: str = "/login") -> str:
        body = self.client.get(path).get_data(as_text=True)
        match = CSRF_RE.search(body)
        self.assertIsNotNone(match, f"No CSRF token found on {path}")
        return match.group(1)

    def register(self, email="user@example.com", password="correct-horse-9", mobile=""):
        return self.client.post(
            "/register",
            data={
                "email": email, "mobile": mobile, "password": password,
                "confirm_password": password, "csrf_token": self.csrf("/register"),
            },
            follow_redirects=False,
        )

    def login(self, identifier="user@example.com", password="correct-horse-9", client=None):
        client = client or self.client
        body = client.get("/login").get_data(as_text=True)
        token = CSRF_RE.search(body).group(1)
        return client.post(
            "/login",
            data={"identifier": identifier, "password": password, "csrf_token": token},
            follow_redirects=False,
        )

    def register_and_login(self, email="user@example.com", password="correct-horse-9",
                           role=None):
        self.register(email=email, password=password)
        if role:
            self.set_role(email, role)
        response = self.login(identifier=email, password=password)
        self.assertEqual(response.status_code, 302, "login should redirect on success")
        return response

    def register_other(self, email: str, password: str = "correct-horse-9"):
        """Register a second account without disturbing the current session.

        A signed-in client is redirected away from /register, so this uses a
        throwaway client of its own.
        """
        other = self.app.test_client()
        body = other.get("/register").get_data(as_text=True)
        token = CSRF_RE.search(body).group(1)
        response = other.post(
            "/register",
            data={"email": email, "mobile": "", "password": password,
                  "confirm_password": password, "csrf_token": token},
        )
        assert response.status_code == 302, response.get_data(as_text=True)[:400]
        return other

    def user_id_for(self, email: str) -> int:
        with self.app.app_context():
            row = query_one("SELECT id FROM users WHERE email = ?", (email,))
            assert row is not None, f"no account for {email}"
            return int(row["id"])

    def set_role(self, email: str, role: str) -> None:
        with self.app.app_context():
            execute("UPDATE users SET role = ? WHERE email = ?", (role, email))

    def register_and_login_admin(self, email="admin@example.com", password="correct-horse-9"):
        return self.register_and_login(email=email, password=password, role="admin")

    def logout(self, client=None):
        client = client or self.client
        return client.post("/logout", data={"csrf_token": self.csrf("/dashboard")})

    def api_token(self) -> str:
        """CSRF token usable for JSON API calls."""
        return self.csrf("/dashboard")

    def post_json(self, path, payload, client=None, method="POST"):
        client = client or self.client
        return client.open(
            path, method=method, json=payload,
            headers={"X-CSRF-Token": self.api_token()},
        )

    def create_item(self, label="Test tomatoes", food_key="tomato", **extra):
        with self.app.app_context():
            food = query_one("SELECT id FROM food_types WHERE key = ?", (food_key,))
            food_id = food["id"]
        payload = {"label": label, "food_type_id": food_id, "storage": "room"}
        payload.update(extra)
        response = self.post_json("/api/v1/items", payload)
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        return response.get_json()["data"]["item"]

    def make_device(self, name="Test Pi"):
        response = self.post_json("/api/v1/devices", {"name": name})
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        return response.get_json()["data"]["token"]
