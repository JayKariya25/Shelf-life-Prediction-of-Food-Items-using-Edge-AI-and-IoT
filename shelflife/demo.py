"""Seed a small, realistic set of tracked items for demos and screenshots.

Everything created here is ordinary application data: real item rows, run
through the real estimator. Nothing is faked into the predictions table.
"""

from __future__ import annotations

from datetime import timedelta

from .db import execute, iso_now, query_one, utcnow
from .services import get_item, list_items, run_prediction

# The project covers banana and tomato only. Ages are chosen so a demo shows the
# full range of states rather than a wall of red: something healthy, something to
# use soon, and something already past its estimate.
DEMO_ITEMS = (
    # (food key, label, quantity, storage, hours ago it was stored)
    ("tomato", "Market tomatoes", "6 pieces", "counter", 36),
    ("tomato", "Salad tomatoes", "4 pieces", "fridge", 120),
    ("banana", "Bananas", "1 bunch", "counter", 60),
    ("banana", "Ripe bananas", "3 pieces", "counter", 132),
)


def seed_demo_items(email: str) -> int:
    """Create the demo items for ``email``. Existing labels are skipped."""
    user = query_one("SELECT id FROM users WHERE email = ?", (email,))
    if user is None:
        raise SystemExit(f"No account found for {email}. Register it first.")
    user_id = user["id"]

    existing = {item["label"] for item in list_items(user_id, "all")}
    now = utcnow()
    created = 0

    for food_key, label, quantity, storage, hours_ago in DEMO_ITEMS:
        if label in existing:
            continue
        food = query_one("SELECT id FROM food_types WHERE key = ?", (food_key,))
        if food is None:
            continue
        stored_at = (now - timedelta(hours=hours_ago)).replace(microsecond=0).isoformat()
        timestamp = iso_now()
        cursor = execute(
            """
            INSERT INTO items (
                user_id, food_type_id, label, quantity, storage, stored_at,
                status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'active', ?, ?)
            """,
            (user_id, food["id"], label, quantity, storage, stored_at, timestamp, timestamp),
        )
        item = get_item(user_id, int(cursor.lastrowid))
        if item:
            run_prediction(user_id, item)
        created += 1

    return created
