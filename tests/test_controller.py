from datetime import datetime

import numpy as np

from taubenschreck.backend import db
from taubenschreck.backend.controller import Controller
from taubenschreck.backend.recorder import Recorder
from taubenschreck.core.config import AppConfig, SafetyConfig
from taubenschreck.core.state import SafetyState
from taubenschreck.detector.model import FakeDetector
from taubenschreck.detector.pipeline import Pipeline
from taubenschreck.detector.sprayer.mock import MockPump


def _build(tmp_path):
    con = db.connect(str(tmp_path / "t.db"))
    db.init_db(con)
    recorder = Recorder(con, str(tmp_path / "snaps"))
    sprayer = MockPump()
    cfg = AppConfig(safety=SafetyConfig(persistence_frames=1, burst_seconds=1.5))
    pipe = Pipeline(
        source=None,
        detector=FakeDetector([[]]),
        sprayer=sprayer,
        config=cfg,
        state=SafetyState(),
        clock=lambda: datetime(2026, 6, 22, 12, 0, 0),
    )
    ctrl = Controller(pipe, recorder, sprayer, cfg.safety)
    return con, ctrl, sprayer


def test_arm_disarm_toggles_state(tmp_path):
    _, ctrl, _ = _build(tmp_path)
    assert ctrl.is_armed() is False    # boot disarmed
    ctrl.arm()
    assert ctrl.is_armed() is True
    ctrl.disarm()
    assert ctrl.is_armed() is False


def test_test_fire_sprays_and_records_manual_event(tmp_path):
    con, ctrl, sprayer = _build(tmp_path)
    ctrl.test_fire()
    assert sprayer.fires == [1.5]
    rows = db.list_events(con)
    assert len(rows) == 1
    assert rows[0]["reason"] == "manual_test"
    assert rows[0]["snapshot_path"] is None
