"""Create or reset a local development account. Never run this in production."""

from __future__ import annotations

import sys

from werkzeug.security import generate_password_hash

sys.path.insert(0, ".")
from shelflife import create_app                      # noqa: E402
from shelflife.db import execute, iso_now, query_one   # noqa: E402


def main(email: str, password: str) -> None:
    app = create_app()
    with app.app_context():
        existing = query_one("SELECT id FROM users WHERE email = ?", (email,))
        if existing:
            execute(
                "UPDATE users SET password_hash = ?, is_active = 1 WHERE id = ?",
                (generate_password_hash(password), existing["id"]),
            )
            user_id = existing["id"]
            print(f"Reset password for existing user id={user_id} ({email})")
        else:
            cursor = execute(
                "INSERT INTO users (email, full_name, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (email, "Dev User", generate_password_hash(password), iso_now()),
            )
            user_id = int(cursor.lastrowid)
            print(f"Created user id={user_id} ({email})")
        if not query_one("SELECT 1 FROM settings WHERE user_id = ?", (user_id,)):
            execute(
                "INSERT INTO settings (user_id, email_notifications, sms_notifications, alert_days_before)"
                " VALUES (?, 1, 0, 2)",
                (user_id,),
            )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dev@example.com",
         sys.argv[2] if len(sys.argv) > 2 else "devpassword123")
