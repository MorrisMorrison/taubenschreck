from __future__ import annotations

import sqlite3
from datetime import datetime


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            event_type TEXT NOT NULL,
            reason TEXT NOT NULL,
            snapshot_path TEXT
        )
        """
    )
    con.commit()


def insert_event(con, ts: str, event_type: str, reason: str, snapshot_path: str | None) -> int:
    cur = con.execute(
        "INSERT INTO events (ts, event_type, reason, snapshot_path) VALUES (?, ?, ?, ?)",
        (ts, event_type, reason, snapshot_path),
    )
    con.commit()
    return int(cur.lastrowid)


def list_events(con, limit: int = 50) -> list[dict]:
    rows = con.execute(
        "SELECT id, ts, event_type, reason, snapshot_path FROM events ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def stats(con, now: datetime) -> dict:
    total = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    day = now.strftime("%Y-%m-%d")
    today = con.execute(
        "SELECT COUNT(*) FROM events WHERE substr(ts, 1, 10) = ?", (day,)
    ).fetchone()[0]
    last = con.execute("SELECT ts FROM events ORDER BY id DESC LIMIT 1").fetchone()
    return {"total": int(total), "today": int(today), "last_ts": last[0] if last else None}
