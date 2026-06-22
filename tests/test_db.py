from datetime import datetime

import numpy as np

from taubenschreck.backend import db
from taubenschreck.backend.storage import save_snapshot


def _con(tmp_path):
    con = db.connect(str(tmp_path / "t.db"))
    db.init_db(con)
    return con


def test_insert_and_list_newest_first(tmp_path):
    con = _con(tmp_path)
    db.insert_event(con, "2026-06-22T12:00:00", "fire", "fire", "a.jpg")
    db.insert_event(con, "2026-06-22T12:05:00", "fire", "fire", "b.jpg")
    rows = db.list_events(con, limit=10)
    assert [r["snapshot_path"] for r in rows] == ["b.jpg", "a.jpg"]


def test_stats_counts_today_and_total(tmp_path):
    con = _con(tmp_path)
    now = datetime(2026, 6, 22, 12, 0, 0)
    db.insert_event(con, "2026-06-21T12:00:00", "fire", "fire", None)  # yesterday
    db.insert_event(con, "2026-06-22T08:00:00", "fire", "fire", None)  # today
    db.insert_event(con, "2026-06-22T09:00:00", "fire", "fire", None)  # today
    s = db.stats(con, now)
    assert s["total"] == 3
    assert s["today"] == 2
    assert s["last_ts"] == "2026-06-22T09:00:00"


def test_save_snapshot_writes_jpeg(tmp_path):
    frame = np.full((48, 64, 3), 128, dtype=np.uint8)
    ts = datetime(2026, 6, 22, 12, 0, 0, 123456)
    path = save_snapshot(tmp_path, ts, frame)
    assert path.endswith(".jpg")
    assert (tmp_path / path.split("/")[-1]).exists()
