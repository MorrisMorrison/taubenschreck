from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from taubenschreck.backend import db
from taubenschreck.backend.storage import save_snapshot
from taubenschreck.core.types import Detection, Event


@dataclass
class Recorder:
    con: object                 # sqlite3.Connection
    snapshot_dir: str
    notifier: object | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, event: Event, frame: np.ndarray | None, detections: list[Detection]) -> int:
        snapshot_path = None
        if frame is not None:
            snapshot_path = save_snapshot(self.snapshot_dir, event.timestamp, frame)
        with self._lock:
            row_id = db.insert_event(
                self.con,
                event.timestamp.isoformat(),
                event.event_type.value,
                event.reason,
                snapshot_path,
            )
        if self.notifier is not None:
            self.notifier.notify_fire(event, snapshot_path)
        return row_id
