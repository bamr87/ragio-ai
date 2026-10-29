"""SQLite catalog: the pipeline's source of truth and the 'parent' docstore.

Qdrant only holds vectors + light metadata. The full text of every window and
segment lives here, so parent/neighbor expansion never depends on the vector DB.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id            TEXT PRIMARY KEY,      -- stable short id derived from the feed GUID
    guid          TEXT UNIQUE,
    title         TEXT NOT NULL,
    published     TEXT,                  -- ISO date
    description   TEXT,
    page_url      TEXT,
    audio_url     TEXT,
    duration_sec  INTEGER,
    audio_path    TEXT,
    status        TEXT NOT NULL DEFAULT 'catalogued',  -- catalogued|downloaded|transcribed|chunked|indexed|failed
    error         TEXT,
    summary       TEXT,                  -- LLM-generated (phase 2)
    updated_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS segments (   -- raw Whisper segments (fine-grained timestamps)
    episode_id TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    start      REAL NOT NULL,
    "end"      REAL NOT NULL,
    text       TEXT NOT NULL,
    PRIMARY KEY (episode_id, idx)
);
CREATE TABLE IF NOT EXISTS windows (    -- child nodes: merged ~200-word timestamp windows
    episode_id TEXT NOT NULL,
    idx        INTEGER NOT NULL,
    start      REAL NOT NULL,
    "end"      REAL NOT NULL,
    text       TEXT NOT NULL,
    PRIMARY KEY (episode_id, idx)
);
CREATE INDEX IF NOT EXISTS windows_ep ON windows(episode_id, idx);
"""


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def set_status(conn: sqlite3.Connection, episode_id: str, status: str, error: str | None = None) -> None:
    conn.execute(
        "UPDATE episodes SET status=?, error=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (status, error, episode_id),
    )


def episodes_with_status(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM episodes WHERE status=? ORDER BY published", (status,)
    ).fetchall()


def get_episode(conn: sqlite3.Connection, episode_id: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()


def get_windows(conn: sqlite3.Connection, episode_id: str, lo: int, hi: int) -> list[sqlite3.Row]:
    """Inclusive index range [lo, hi] — used for neighbor expansion around a hit."""
    return conn.execute(
        'SELECT episode_id, idx, start, "end", text FROM windows '
        "WHERE episode_id=? AND idx BETWEEN ? AND ? ORDER BY idx",
        (episode_id, lo, hi),
    ).fetchall()


def episode_text(conn: sqlite3.Connection, episode_id: str) -> str:
    rows = conn.execute(
        "SELECT text FROM windows WHERE episode_id=? ORDER BY idx", (episode_id,)
    ).fetchall()
    return "\n".join(r["text"] for r in rows)
