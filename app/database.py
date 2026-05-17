import sqlite3
import uuid
import secrets
from contextlib import contextmanager
from .config import settings


def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS apps (
                app_id        TEXT PRIMARY KEY,
                name          TEXT NOT NULL,
                api_key       TEXT UNIQUE NOT NULL,
                chroma_prefix TEXT UNIQUE NOT NULL,
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


@contextmanager
def get_db():
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_app(name: str) -> dict:
    app_id = str(uuid.uuid4())
    api_key = f"ses-{secrets.token_urlsafe(32)}"
    # 8 hex chars — unique per app, used to namespace ChromaDB collections
    chroma_prefix = secrets.token_hex(4)

    with get_db() as conn:
        conn.execute(
            "INSERT INTO apps (app_id, name, api_key, chroma_prefix) VALUES (?, ?, ?, ?)",
            (app_id, name, api_key, chroma_prefix),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM apps WHERE app_id = ?", (app_id,)
        ).fetchone()
        return dict(row)


def get_app_by_api_key(api_key: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM apps WHERE api_key = ?", (api_key,)
        ).fetchone()
        return dict(row) if row else None


def list_apps() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT app_id, name, created_at FROM apps ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]
