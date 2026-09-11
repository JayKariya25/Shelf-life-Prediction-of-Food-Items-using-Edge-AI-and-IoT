"""The two interfaces: what each role can and cannot reach."""

from __future__ import annotations

import unittest

from tests.support import AppTestCase

from shelflife.db import query_one

# Every page and endpoint behind the developer console.
ADMIN_PAGES = ("/admin/", "/admin/model", "/admin/playground",
               "/admin/evaluation", "/admin/users", "/admin/system")
ADMIN_GET_APIS = ("/admin/api/test-samples",)
ADMIN_POST_APIS = ("/admin/api/predict", "/admin/api/evaluate",
                   "/admin/api/reload-model", "/admin/api/prune-readings")

USER_PAGES = ("/dashboard", "/items", "/alerts", "/device", "/profile")


class DefaultRole(AppTestCase):
    def test_new_accounts_are_ordinary_users(self):
        self.register()
        with self.app.app_context():
            self.assertEqual(query_one("SELECT role FROM users")["role"], "user")

    def test_migrated_accounts_default_to_user(self):
        """Existing prototype accounts must not silently gain admin rights."""
        self.register(email="legacy@example.com")
        with self.app.app_context():
            row = query_one("SELECT role FROM users WHERE email = ?", ("legacy@example.com",))
            self.assertEqual(row["role"], "user")


class AnonymousAccess(AppTestCase):
    def test_admin_pages_redirect_to_login(self):
        for path in ADMIN_PAGES:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 302, path)
            self.assertIn("/login", response.headers["Location"], path)

    def test_admin_apis_return_401_json(self):
        for path in ADMIN_GET_APIS:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 401, path)
            self.assertEqual(response.get_json()["error"]["code"], "unauthenticated")


class UserIsLockedOut(AppTestCase):
    """The central requirement: a plain user must not reach the developer side."""

    def setUp(self):
        super().setUp()
        self.register_and_login(email="plain@example.com")

    def test_user_can_use_the_monitoring_interface(self):
        for path in USER_PAGES:
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_every_admin_page_is_forbidden(self):
        for path in ADMIN_PAGES:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 403, path)

    def test_forbidden_page_explains_rather_than_redirecting(self):
        body = self.client.get("/admin/").get_data(as_text=True)
        self.assertIn("Administrators only", body)

    def test_every_admin_get_api_is_forbidden(self):
        for path in ADMIN_GET_APIS:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 403, path)
            self.assertEqual(response.get_json()["error"]["code"], "forbidden")

    def test_every_admin_post_api_is_forbidden(self):
        for path in ADMIN_POST_APIS:
            response = self.post_json(path, {})
            self.assertEqual(response.status_code, 403, path)

    def test_user_cannot_change_roles(self):
        with self.app.app_context():
            user_id = query_one("SELECT id FROM users")["id"]
        response = self.post_json(f"/admin/api/users/{user_id}/role",
                                  {"role": "admin"}, method="PATCH")
        self.assertEqual(response.status_code, 403)
        with self.app.app_context():
            self.assertEqual(query_one("SELECT role FROM users")["role"], "user")

    def test_user_cannot_read_dataset_images(self):
        response = self.client.get("/admin/api/test-samples/0/image")
        self.assertEqual(response.status_code, 403)

    def test_developer_navigation_is_absent(self):
        body = self.client.get("/dashboard").get_data(as_text=True)
        self.assertNotIn("/admin/", body)
        self.assertNotIn("Playground", body)
        self.assertNotIn("Developer", body)


class AdminSeesBothInterfaces(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register_and_login_admin()

    def test_admin_reaches_the_user_interface(self):
        for path in USER_PAGES:
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_admin_reaches_every_developer_page(self):
        for path in ADMIN_PAGES:
            self.assertEqual(self.client.get(path).status_code, 200, path)

    def test_developer_navigation_is_present(self):
        body = self.client.get("/dashboard").get_data(as_text=True)
        self.assertIn("Developer", body)
        self.assertIn("/admin/playground", body)
        self.assertIn("/admin/evaluation", body)

    def test_admin_badge_is_shown(self):
        self.assertIn(">Admin", self.client.get("/dashboard").get_data(as_text=True))

    def test_admin_can_still_track_items_like_a_user(self):
        item = self.create_item(label="Admin tomatoes")
        self.assertEqual(item["label"], "Admin tomatoes")


class RoleManagement(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register_and_login_admin()
        self.register_other("plain@example.com")
        self.plain_id = self.user_id_for("plain@example.com")
        self.admin_id = self.user_id_for("admin@example.com")

    def test_promoting_a_user(self):
        response = self.post_json(f"/admin/api/users/{self.plain_id}/role",
                                  {"role": "admin"}, method="PATCH")
        self.assertEqual(response.status_code, 200)
        with self.app.app_context():
            self.assertEqual(
                query_one("SELECT role FROM users WHERE id = ?", (self.plain_id,))["role"],
                "admin",
            )

    def test_promoted_user_gains_console_access(self):
        self.post_json(f"/admin/api/users/{self.plain_id}/role", {"role": "admin"}, method="PATCH")
        self.logout()
        self.login(identifier="plain@example.com")
        self.assertEqual(self.client.get("/admin/").status_code, 200)

    def test_demoted_admin_loses_console_access(self):
        self.post_json(f"/admin/api/users/{self.plain_id}/role", {"role": "admin"}, method="PATCH")
        self.post_json(f"/admin/api/users/{self.plain_id}/role", {"role": "user"}, method="PATCH")
        self.logout()
        self.login(identifier="plain@example.com")
        self.assertEqual(self.client.get("/admin/").status_code, 403)

    def test_invalid_role_is_rejected(self):
        response = self.post_json(f"/admin/api/users/{self.plain_id}/role",
                                  {"role": "superuser"}, method="PATCH")
        self.assertEqual(response.status_code, 422)

    def test_the_last_admin_cannot_be_demoted(self):
        """Otherwise the console becomes unreachable without a database edit."""
        response = self.post_json(f"/admin/api/users/{self.admin_id}/role",
                                  {"role": "user"}, method="PATCH")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"]["code"], "last_admin")

    def test_the_last_admin_cannot_be_deactivated(self):
        response = self.post_json(f"/admin/api/users/{self.admin_id}/active",
                                  {"is_active": False}, method="PATCH")
        self.assertEqual(response.status_code, 409)

    def test_an_admin_cannot_deactivate_themselves(self):
        self.post_json(f"/admin/api/users/{self.plain_id}/role", {"role": "admin"}, method="PATCH")
        response = self.post_json(f"/admin/api/users/{self.admin_id}/active",
                                  {"is_active": False}, method="PATCH")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.get_json()["error"]["code"], "self_disable")

    def test_deactivated_user_cannot_sign_in(self):
        self.post_json(f"/admin/api/users/{self.plain_id}/active",
                       {"is_active": False}, method="PATCH")
        self.logout()
        response = self.login(identifier="plain@example.com")
        self.assertEqual(response.status_code, 401)

    def test_role_changes_require_csrf(self):
        response = self.client.patch(f"/admin/api/users/{self.plain_id}/role",
                                     json={"role": "admin"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.get_json()["error"]["code"], "csrf_failed")


class AdminIsolationFromUserData(AppTestCase):
    """Admin rights are for the console, not a licence to read someone's items."""

    def test_admin_does_not_see_another_users_items_in_the_normal_ui(self):
        self.register_and_login(email="alice@example.com")
        alice_item = self.create_item(label="Alice tomatoes")
        self.logout()

        self.register_and_login_admin()
        self.assertEqual(self.client.get(f"/items/{alice_item['id']}").status_code, 404)
        items = self.client.get("/api/v1/items?status=all").get_json()["data"]["items"]
        self.assertEqual(items, [])


if __name__ == "__main__":
    unittest.main()
