"""SQLite access layer, schema definition and forward-only migrations.

The database is deliberately plain SQLite: it is the only store that runs
comfortably on a Raspberry Pi 5 with no extra services, and it keeps the whole
capstone reproducible from a single file.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator

from flask import current_app, g

SCHEMA_VERSION = 4

# --- schema -----------------------------------------------------------------
# Tables are created in dependency order. Everything is IF NOT EXISTS so that
# init_db() is idempotent and safe to run against an existing database.
SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS users (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        email          TEXT UNIQUE,
        mobile         TEXT UNIQUE,
        full_name      TEXT,
        password_hash  TEXT NOT NULL,
        created_at     TEXT NOT NULL,
        last_login_at  TEXT,
        is_active      INTEGER NOT NULL DEFAULT 1,
        role           TEXT NOT NULL DEFAULT 'user'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS settings (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id               INTEGER UNIQUE NOT NULL,
        email_notifications   INTEGER NOT NULL DEFAULT 0,
        sms_notifications     INTEGER NOT NULL DEFAULT 0,
        alert_days_before     INTEGER NOT NULL DEFAULT 2,
        latest_image          TEXT,
        theme                 TEXT NOT NULL DEFAULT 'system',
        temperature_unit      TEXT NOT NULL DEFAULT 'C',
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS devices (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       INTEGER NOT NULL,
        name          TEXT NOT NULL,
        location      TEXT,
        token_hash    TEXT NOT NULL UNIQUE,
        token_prefix  TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        last_seen_at  TEXT,
        firmware      TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS food_types (
        id                    INTEGER PRIMARY KEY AUTOINCREMENT,
        key                   TEXT NOT NULL UNIQUE,
        name                  TEXT NOT NULL,
        category              TEXT NOT NULL,
        emoji                 TEXT NOT NULL DEFAULT '',
        ref_shelf_life_hours  REAL NOT NULL,
        ref_temperature_c     REAL NOT NULL,
        q10                   REAL NOT NULL,
        ideal_temp_min_c      REAL,
        ideal_temp_max_c      REAL,
        ideal_humidity_min    REAL,
        ideal_humidity_max    REAL,
        notes                 TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS items (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id        INTEGER NOT NULL,
        food_type_id   INTEGER NOT NULL,
        device_id      INTEGER,
        label          TEXT NOT NULL,
        quantity       TEXT,
        storage        TEXT NOT NULL DEFAULT 'room',
        stored_at      TEXT NOT NULL,
        status         TEXT NOT NULL DEFAULT 'active',
        closed_at      TEXT,
        image_path     TEXT,
        notes          TEXT,
        created_at     TEXT NOT NULL,
        updated_at     TEXT NOT NULL,
        FOREIGN KEY (user_id)      REFERENCES users (id)      ON DELETE CASCADE,
        FOREIGN KEY (food_type_id) REFERENCES food_types (id),
        FOREIGN KEY (device_id)    REFERENCES devices (id)    ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS readings (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id        INTEGER NOT NULL,
        device_id      INTEGER,
        item_id        INTEGER,
        recorded_at    TEXT NOT NULL,
        temperature_c  REAL,
        humidity_pct   REAL,
        pressure_hpa   REAL,
        gas_ppm        REAL,
        source         TEXT NOT NULL DEFAULT 'simulated',
        FOREIGN KEY (user_id)   REFERENCES users (id)   ON DELETE CASCADE,
        FOREIGN KEY (device_id) REFERENCES devices (id) ON DELETE SET NULL,
        FOREIGN KEY (item_id)   REFERENCES items (id)   ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id              INTEGER NOT NULL,
        item_id              INTEGER,
        created_at           TEXT NOT NULL,
        freshness_class      TEXT NOT NULL,
        freshness_confidence REAL,
        remaining_hours      REAL,
        remaining_hours_low  REAL,
        remaining_hours_high REAL,
        model_name           TEXT NOT NULL,
        model_version        TEXT NOT NULL,
        model_kind           TEXT NOT NULL,
        inference_ms         REAL,
        temperature_c        REAL,
        humidity_pct         REAL,
        gas_ppm              REAL,
        image_path           TEXT,
        rationale            TEXT,
        fallback_reason      TEXT,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
        FOREIGN KEY (item_id) REFERENCES items (id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id         INTEGER NOT NULL,
        item_id         INTEGER,
        prediction_id   INTEGER,
        created_at      TEXT NOT NULL,
        level           TEXT NOT NULL,
        title           TEXT NOT NULL,
        message         TEXT NOT NULL,
        acknowledged_at TEXT,
        FOREIGN KEY (user_id)       REFERENCES users (id)       ON DELETE CASCADE,
        FOREIGN KEY (item_id)       REFERENCES items (id)       ON DELETE CASCADE,
        FOREIGN KEY (prediction_id) REFERENCES predictions (id) ON DELETE SET NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
)

# Indexes are applied *after* column migration, because an index may reference a
# column that only exists once the migration has added it.
INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX IF NOT EXISTS idx_readings_user_time  ON readings (user_id, recorded_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_readings_item_time  ON readings (item_id, recorded_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_item    ON predictions (item_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_user    ON predictions (user_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_user_time    ON alerts (user_id, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS idx_items_user_status   ON items (user_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_devices_user        ON devices (user_id)",
    "CREATE INDEX IF NOT EXISTS idx_users_role          ON users (role)",
)

# --- reference data ----------------------------------------------------------
# Reference shelf life and Q10 coefficients are order-of-magnitude values taken
# from standard postharvest storage guidance. They parameterise the *baseline*
# kinetic estimator only; a trained model, once available, overrides them.
# The project scope is banana and tomato only, matching the trained model's
# training set. Reference shelf life and Q10 coefficients are order-of-magnitude
# values from standard postharvest storage guidance; they parameterise the
# *baseline* kinetic estimator only.
SUPPORTED_FOOD_KEYS = ("banana", "tomato")

FOOD_TYPE_SEED: tuple[dict[str, Any], ...] = (
    dict(key="banana", name="Banana", category="fruit", emoji="\N{BANANA}",
         ref_shelf_life_hours=144, ref_temperature_c=20, q10=2.8,
         ideal_temp_min_c=13, ideal_temp_max_c=15, ideal_humidity_min=85, ideal_humidity_max=95,
         notes="High ethylene producer; blackens under refrigeration."),
    dict(key="tomato", name="Tomato", category="vegetable", emoji="\N{TOMATO}",
         ref_shelf_life_hours=168, ref_temperature_c=20, q10=2.4,
         ideal_temp_min_c=10, ideal_temp_max_c=15, ideal_humidity_min=85, ideal_humidity_max=95,
         notes="Chilling injury below ~10 C; ripens fast above 25 C."),
)


# --- helpers ----------------------------------------------------------------
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    """Timestamps are stored as ISO-8601 UTC strings, second resolution."""
    return utcnow().replace(microsecond=0).isoformat()


def parse_ts(value: str | None) -> datetime | None:
    """Parse a stored timestamp back into an aware datetime, tolerantly."""
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def connect(database_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(database_path, timeout=15, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 15000")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def get_db() -> sqlite3.Connection:
    """Per-request connection, closed by :func:`close_db`."""
    if "db" not in g:
        g.db = connect(current_app.config["DATABASE_PATH"])
    return g.db


def close_db(exc: BaseException | None = None) -> None:
    conn = g.pop("db", None)
    if conn is not None:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection | None = None) -> Iterator[sqlite3.Connection]:
    """Explicit transaction wrapper (connections run in autocommit mode)."""
    own = conn is None
    conn = conn or get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
    finally:
        if own:
            pass  # request teardown owns the lifetime


def query_all(sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
    return get_db().execute(sql, tuple(params)).fetchall()


def query_one(sql: str, params: Iterable[Any] = ()) -> sqlite3.Row | None:
    return get_db().execute(sql, tuple(params)).fetchone()


def execute(sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
    return get_db().execute(sql, tuple(params))


# --- schema management -------------------------------------------------------
def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _migrate_legacy_columns(conn: sqlite3.Connection) -> list[str]:
    """Add columns introduced after the first prototype.

    The original prototype shipped ``users`` and ``settings`` only. Existing
    rows are preserved; new columns are appended with sensible defaults.
    """
    applied: list[str] = []
    additions = {
        "users": {
            "full_name": "ALTER TABLE users ADD COLUMN full_name TEXT",
            "last_login_at": "ALTER TABLE users ADD COLUMN last_login_at TEXT",
            "is_active": "ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1",
            # 'user' sees only the monitoring UI; 'admin' additionally sees the
            # developer console. Existing accounts stay ordinary users.
            "role": "ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'",
        },
        "predictions": {
            "fallback_reason": "ALTER TABLE predictions ADD COLUMN fallback_reason TEXT",
        },
        "settings": {
            "theme": "ALTER TABLE settings ADD COLUMN theme TEXT NOT NULL DEFAULT 'system'",
            "temperature_unit": "ALTER TABLE settings ADD COLUMN temperature_unit TEXT NOT NULL DEFAULT 'C'",
        },
    }
    for table, columns in additions.items():
        if not _table_exists(conn, table):
            continue
        existing = _table_columns(conn, table)
        for column, ddl in columns.items():
            if column not in existing:
                conn.execute(ddl)
                applied.append(f"{table}.{column}")
    return applied


def _seed_food_types(conn: sqlite3.Connection) -> int:
    """Insert reference profiles that are not present yet (never overwrites)."""
    inserted = 0
    for profile in FOOD_TYPE_SEED:
        existing = conn.execute(
            "SELECT 1 FROM food_types WHERE key = ?", (profile["key"],)
        ).fetchone()
        if existing:
            continue
        conn.execute(
            """
            INSERT INTO food_types (
                key, name, category, emoji, ref_shelf_life_hours, ref_temperature_c,
                q10, ideal_temp_min_c, ideal_temp_max_c, ideal_humidity_min,
                ideal_humidity_max, notes
            ) VALUES (
                :key, :name, :category, :emoji, :ref_shelf_life_hours, :ref_temperature_c,
                :q10, :ideal_temp_min_c, :ideal_temp_max_c, :ideal_humidity_min,
                :ideal_humidity_max, :notes
            )
            """,
            profile,
        )
        inserted += 1
    return inserted


def _prune_food_types(conn: sqlite3.Connection) -> list[str]:
    """Remove produce profiles outside the project scope.

    Only rows that no item references are removed, so narrowing the scope can
    never silently delete a user's tracked item. Anything still in use is left
    alone and reported, for a human to decide about.
    """
    removed: list[str] = []
    rows = conn.execute(
        """
        SELECT f.id, f.key, (SELECT COUNT(*) FROM items i WHERE i.food_type_id = f.id) AS uses
        FROM food_types f
        """
    ).fetchall()
    for row in rows:
        if row["key"] in SUPPORTED_FOOD_KEYS or row["uses"]:
            continue
        conn.execute("DELETE FROM food_types WHERE id = ?", (row["id"],))
        removed.append(row["key"])
    return removed


def food_types_in_use_outside_scope(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Out-of-scope profiles that still have items attached."""
    placeholders = ",".join("?" for _ in SUPPORTED_FOOD_KEYS)
    return conn.execute(
        f"""
        SELECT f.key, f.name, COUNT(i.id) AS uses
        FROM food_types f JOIN items i ON i.food_type_id = f.id
        WHERE f.key NOT IN ({placeholders})
        GROUP BY f.id ORDER BY f.key
        """,
        SUPPORTED_FOOD_KEYS,
    ).fetchall()


def _backfill_settings(conn: sqlite3.Connection) -> int:
    """Guarantee every user owns exactly one settings row."""
    cursor = conn.execute(
        """
        INSERT INTO settings (user_id, email_notifications, sms_notifications, alert_days_before)
        SELECT u.id, 0, 0, 2 FROM users u
        WHERE NOT EXISTS (SELECT 1 FROM settings s WHERE s.user_id = u.id)
        """
    )
    return cursor.rowcount or 0


def init_db(database_path: str) -> dict[str, Any]:
    """Create or upgrade the schema in place. Safe to call on every boot."""
    conn = connect(database_path)
    report: dict[str, Any] = {"created": False, "migrated_columns": [], "seeded_food_types": 0}
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            fresh = not _table_exists(conn, "users")
            for statement in SCHEMA_STATEMENTS:
                conn.execute(statement)
            report["created"] = fresh
            report["migrated_columns"] = _migrate_legacy_columns(conn)
            for statement in INDEX_STATEMENTS:
                conn.execute(statement)
            report["seeded_food_types"] = _seed_food_types(conn)
            report["pruned_food_types"] = _prune_food_types(conn)
            report["backfilled_settings"] = _backfill_settings(conn)
            conn.execute(
                "INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(SCHEMA_VERSION),),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    return report


def prune_readings(conn: sqlite3.Connection, retention_days: int) -> int:
    """Drop readings older than the retention window (edge storage is small)."""
    if retention_days <= 0:
        return 0
    cutoff = (utcnow() - __import__("datetime").timedelta(days=retention_days)).replace(microsecond=0).isoformat()
    cursor = conn.execute("DELETE FROM readings WHERE recorded_at < ?", (cutoff,))
    return cursor.rowcount or 0
