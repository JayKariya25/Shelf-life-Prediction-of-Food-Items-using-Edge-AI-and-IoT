"""Page rendering: every route must render, in every state."""

from __future__ import annotations

import unittest

from tests.support import AppTestCase


class PagesRender(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register_and_login()

    def get(self, path):
        response = self.client.get(path)
        self.assertEqual(response.status_code, 200, f"{path} -> {response.status_code}")
        return response.get_data(as_text=True)

    def test_all_pages_render_when_empty(self):
        for path in ("/dashboard", "/items", "/alerts", "/device", "/profile"):
            body = self.get(path)
            self.assertIn("</html>", body)
            self.assertNotIn("None", body.split("<body>")[1][:200])

    def test_all_pages_render_with_data(self):
        item = self.create_item()
        for path in ("/dashboard", "/items", "/alerts", "/device", "/profile", f"/items/{item['id']}"):
            self.assertIn("</html>", self.get(path))

    def test_item_filters_render(self):
        self.create_item()
        for status in ("active", "consumed", "discarded", "all"):
            self.get(f"/items?status={status}")

    def test_unknown_item_filter_falls_back_to_active(self):
        self.get("/items?status=nonsense")

    def test_missing_item_page_renders_the_404_template(self):
        response = self.client.get("/items/999999")
        self.assertEqual(response.status_code, 404)
        self.assertIn("Nothing here", response.get_data(as_text=True))

    def test_dashboard_states_the_estimator_is_a_baseline(self):
        """The honesty banner must be visible while no trained model is loaded."""
        body = self.get("/dashboard")
        self.assertIn("not a trained model", body)
        self.assertIn("Q10 kinetic baseline", body)

    def test_simulated_data_is_labelled_on_the_dashboard(self):
        self.assertIn("Simulated data", self.get("/dashboard"))

    def test_no_page_loads_assets_from_a_cdn(self):
        """Offline operation on the Pi depends on this staying true."""
        for path in ("/dashboard", "/items", "/alerts", "/device", "/profile", "/login", "/register"):
            body = self.client.get(path).get_data(as_text=True)
            for needle in ("cdn.jsdelivr.net", "cdnjs.", "unpkg.com", "googleapis.com", "http://cdn"):
                self.assertNotIn(needle, body, f"{path} references {needle}")

    def test_every_page_carries_a_csrf_token(self):
        for path in ("/dashboard", "/items", "/alerts", "/device", "/profile"):
            self.assertIn('name="csrf-token"', self.get(path))

    def test_item_detail_shows_the_calculation_rationale(self):
        item = self.create_item()
        body = self.get(f"/items/{item['id']}")
        self.assertIn("How this was calculated", body)
        self.assertIn("Q10", body)

    def test_item_detail_says_the_image_was_not_used(self):
        item = self.create_item()
        self.assertIn("image not used", self.get(f"/items/{item['id']}"))


class AuthPagesRender(AppTestCase):
    def test_login_and_register_render_when_signed_out(self):
        for path in ("/login", "/register"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn("csrf_token", response.get_data(as_text=True))

    def test_root_redirects_to_login_when_signed_out(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login", response.headers["Location"])

    def test_root_redirects_to_dashboard_when_signed_in(self):
        self.register_and_login()
        response = self.client.get("/")
        self.assertEqual(response.headers["Location"], "/dashboard")

    def test_signed_in_users_are_bounced_off_the_login_page(self):
        self.register_and_login()
        self.assertEqual(self.client.get("/login").status_code, 302)
        self.assertEqual(self.client.get("/register").status_code, 302)


class StaticAssets(AppTestCase):
    def test_stylesheet_and_scripts_are_served(self):
        for asset in ("css/app.css", "js/core.js", "js/charts.js", "js/dashboard.js",
                      "js/theme-boot.js", "img/favicon.svg"):
            response = self.client.get(f"/static/{asset}")
            self.assertEqual(response.status_code, 200, asset)
            self.assertGreater(len(response.get_data()), 50, asset)

    def test_every_script_referenced_by_a_template_exists(self):
        import re
        from pathlib import Path

        templates = Path(self.app.root_path).parent / "templates"
        referenced = set()
        for template in templates.rglob("*.html"):
            for match in re.finditer(
                r"url_for\('static',\s*filename='([^']+)'\)", template.read_text(encoding="utf-8")
            ):
                referenced.add(match.group(1))
        for asset in sorted(referenced):
            if "uploads/" in asset:
                continue  # runtime-generated
            self.assertEqual(
                self.client.get(f"/static/{asset}").status_code, 200, f"missing static asset: {asset}"
            )


class XssHardening(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register_and_login()

    def test_item_labels_are_escaped_in_html(self):
        payload = '<script>alert("xss")</script>'
        item = self.create_item(label=payload)
        body = self.client.get("/items").get_data(as_text=True)
        self.assertNotIn("<script>alert", body)
        self.assertIn("&lt;script&gt;", body)
        detail = self.client.get(f"/items/{item['id']}").get_data(as_text=True)
        self.assertNotIn("<script>alert", detail)

    def test_notes_are_escaped(self):
        """The payload may appear as inert text, but never as a live tag."""
        item = self.create_item(notes='"><img src=x onerror=alert(1)>')
        body = self.client.get(f"/items/{item['id']}").get_data(as_text=True)
        self.assertNotIn("<img src=x", body)
        self.assertIn("&lt;img src=x", body)

    def test_json_payload_blocks_cannot_break_out_of_the_script_tag(self):
        """tojson must escape '<' so a label cannot close the data block."""
        self.create_item(label="</script><script>alert(1)</script>")
        body = self.client.get("/dashboard").get_data(as_text=True)
        self.assertNotIn("</script><script>alert", body)
        payload = body.split('id="snapshotData" type="application/json">')[1].split("</script>")[0]
        self.assertNotIn("<", payload)
        self.assertIn("\\u003c", payload)


if __name__ == "__main__":
    unittest.main()


class HiddenAttribute(AppTestCase):
    def test_stylesheet_forces_the_hidden_attribute_to_win(self):
        """Components set `display`, which would otherwise override [hidden]."""
        css = self.client.get("/static/css/app.css").get_data(as_text=True)
        self.assertIn("[hidden] { display: none !important; }", css)

    def test_alert_badge_is_hidden_when_there_is_nothing_to_show(self):
        self.register_and_login()
        body = self.client.get("/device").get_data(as_text=True)
        self.assertIn("hidden", body.split('id="alertBadge"')[1][:40])

    def test_alert_badge_count_renders_on_every_page_without_javascript(self):
        from shelflife import services

        self.register_and_login()
        item = self.create_item()
        with self.app.app_context():
            services.raise_alert(1, item["id"], "critical", "Test alert", "Body text")
        for path in ("/dashboard", "/items", "/alerts", "/device", "/profile"):
            body = self.client.get(path).get_data(as_text=True)
            badge = body.split('id="alertBadge"')[1][:60]
            self.assertNotIn("hidden", badge, path)
            self.assertIn(">1<", badge, path)


class ContentSecurityPolicy(AppTestCase):
    def test_blob_previews_are_permitted(self):
        """The admin playground previews a picked file via URL.createObjectURL."""
        policy = self.client.get("/login").headers["Content-Security-Policy"]
        self.assertIn("img-src 'self' data: blob:", policy)

    def test_policy_still_forbids_remote_scripts_and_framing(self):
        policy = self.client.get("/login").headers["Content-Security-Policy"]
        self.assertIn("script-src 'self'", policy)
        self.assertIn("frame-ancestors 'none'", policy)
        self.assertNotIn("blob:", policy.split("script-src")[1].split(";")[0])


class JinjaDictAttributeTrap(unittest.TestCase):
    """`{{ mapping.items }}` renders a bound method, not the value.

    Jinja tries getattr before getitem, so any dict key that collides with a
    dict method silently renders as "<built-in method ...>". This scans the
    templates so the mistake cannot reach a page again.
    """

    COLLIDING = ("items", "keys", "values", "copy", "pop", "clear", "update", "get")

    def test_no_template_reads_a_colliding_key_as_an_attribute(self):
        import re
        from pathlib import Path

        templates = Path(__file__).resolve().parent.parent / "templates"
        pattern = re.compile(
            r"\.(?P<name>" + "|".join(COLLIDING_NAMES := JinjaDictAttributeTrap.COLLIDING) + r")\b(?P<after>\s*\()?"
        )
        offenders = []
        for template in templates.rglob("*.html"):
            for number, line in enumerate(template.read_text(encoding="utf-8").splitlines(), 1):
                if "{{" not in line and "{%" not in line:
                    continue
                for match in pattern.finditer(line):
                    if match.group("after"):
                        continue  # an explicit call such as .items() is fine
                    before = line[: match.start()]
                    # url_for('pages.items') and similar string literals are fine.
                    if before.rstrip().endswith(("'", '"')) or "'" in before[-30:] or '"' in before[-30:]:
                        continue
                    offenders.append(f"{template.name}:{number}: {line.strip()}")
        self.assertEqual(offenders, [], "use mapping['key'] instead:\n" + "\n".join(offenders))


class ThemePicker(AppTestCase):
    def setUp(self):
        super().setUp()
        self.register_and_login()

    def test_picker_offers_all_three_appearance_options(self):
        body = self.client.get("/dashboard").get_data(as_text=True)
        self.assertIn('id="themeSelect"', body)
        for value in ("system", "light", "dark"):
            self.assertIn(f'value="{value}"', body)

    def test_system_is_the_first_option(self):
        body = self.client.get("/dashboard").get_data(as_text=True)
        options = body.split('id="themeSelect"')[1].split("</select>")[0]
        self.assertLess(options.index('value="system"'), options.index('value="light"'))

    def test_the_old_toggle_button_is_gone(self):
        body = self.client.get("/dashboard").get_data(as_text=True)
        self.assertNotIn("data-theme-toggle", body)

    def test_the_picker_has_an_accessible_label(self):
        body = self.client.get("/dashboard").get_data(as_text=True)
        self.assertIn('for="themeSelect"', body)


class NoPressureInTheInterface(AppTestCase):
    """Pressure is collected if hardware sends it, but never displayed."""

    def setUp(self):
        super().setUp()
        self.register_and_login()

    def test_no_page_mentions_pressure(self):
        self.create_item()
        for path in ("/dashboard", "/items", "/alerts", "/device", "/profile"):
            body = self.client.get(path).get_data(as_text=True).lower()
            self.assertNotIn("pressure", body, path)
            self.assertNotIn("hpa", body, path)

    def test_the_item_page_does_not_mention_pressure(self):
        item = self.create_item()
        body = self.client.get(f"/items/{item['id']}").get_data(as_text=True).lower()
        self.assertNotIn("pressure", body)

    def test_the_dashboard_script_does_not_render_pressure(self):
        script = self.client.get("/static/js/dashboard.js").get_data(as_text=True)
        self.assertNotIn("pressure", script.lower())
