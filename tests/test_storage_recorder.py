from datetime import datetime

import numpy as np

from taubenschreck.backend import db
from taubenschreck.backend.recorder import Recorder
from taubenschreck.core.types import Event, EventType


class _SpyNotifier:
    def __init__(self):
        self.calls = []

    def notify_fire(self, event, snapshot_path=None):
        self.calls.append((event, snapshot_path))


def _recorder(tmp_path, notifier=None):
    con = db.connect(str(tmp_path / "t.db"))
    db.init_db(con)
    return con, Recorder(con, str(tmp_path / "snaps"), notifier=notifier)


def test_record_saves_snapshot_and_row_and_notifies(tmp_path):
    notifier = _SpyNotifier()
    con, rec = _recorder(tmp_path, notifier)
    frame = np.full((48, 64, 3), 100, dtype=np.uint8)
    event = Event(datetime(2026, 6, 22, 12, 0, 0), EventType.FIRE, "fire")
    rec.record(event, frame, [])
    rows = db.list_events(con)
    assert len(rows) == 1
    assert rows[0]["snapshot_path"].endswith(".jpg")
    assert len(notifier.calls) == 1


def test_record_without_frame_stores_null_snapshot(tmp_path):
    con, rec = _recorder(tmp_path)
    event = Event(datetime(2026, 6, 22, 12, 0, 0), EventType.FIRE, "manual_test")
    rec.record(event, None, [])
    rows = db.list_events(con)
    assert rows[0]["snapshot_path"] is None
