from datetime import datetime

from fastapi.testclient import TestClient

from taubenschreck.backend import db
from taubenschreck.backend.app import create_app
from taubenschreck.backend.controller import Controller
from taubenschreck.backend.recorder import Recorder
from taubenschreck.core.config import AppConfig, SafetyConfig
from taubenschreck.core.state import SafetyState
from taubenschreck.detector.model import FakeDetector
from taubenschreck.detector.pipeline import Pipeline
from taubenschreck.detector.sprayer.mock import MockPump


def _client(tmp_path):
    con = db.connect(str(tmp_path / "t.db"))
    db.init_db(con)
    recorder = Recorder(con, str(tmp_path / "snaps"))
    sprayer = MockPump()
    cfg = AppConfig(safety=SafetyConfig(persistence_frames=1))
    pipe = Pipeline(source=None, detector=FakeDetector([[]]), sprayer=sprayer,
                    config=cfg, state=SafetyState(), clock=lambda: datetime(2026, 6, 22, 12, 0, 0))
    ctrl = Controller(pipe, recorder, sprayer, cfg.safety)
    app = create_app(ctrl, con, "taubenschreck/dashboard/static", str(tmp_path / "snaps"))
    return TestClient(app), con


def test_arm_disarm_endpoints(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/api/state").json() == {"armed": False}
    assert client.post("/api/arm").json() == {"armed": True}
    assert client.get("/api/state").json() == {"armed": True}
    assert client.post("/api/disarm").json() == {"armed": False}


def test_test_fire_endpoint_records(tmp_path):
    client, con = _client(tmp_path)
    assert client.post("/api/test-fire").json() == {"ok": True}
    assert db.list_events(con)[0]["reason"] == "manual_test"


def test_events_and_stats_endpoints(tmp_path):
    client, con = _client(tmp_path)
    db.insert_event(con, "2026-06-22T08:00:00", "fire", "fire", None)
    events = client.get("/api/events").json()["events"]
    assert len(events) == 1
    stats = client.get("/api/stats").json()
    assert stats["total"] == 1


def test_root_serves_dashboard(tmp_path):
    client, _ = _client(tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Taubenschreck" in resp.text
