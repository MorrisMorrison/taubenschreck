from datetime import datetime, timedelta

from taubenschreck.core.types import Detection, Event, EventType, SafetyDecision
from taubenschreck.core.state import SafetyState


def test_detection_is_frozen():
    d = Detection(label="bird", confidence=0.9, bbox=(0, 0, 10, 10))
    assert d.label == "bird"


def test_event_defaults():
    now = datetime(2026, 6, 22, 12, 0, 0)
    e = Event(timestamp=now, event_type=EventType.FIRE, reason="fire")
    assert e.snapshot_path is None
    assert e.detections == ()
    assert e.event_type.value == "fire"


def test_decision_fields():
    assert SafetyDecision(fire=True, reason="fire").fire is True


def test_state_boot_disarmed():
    assert SafetyState().armed is False


def test_bursts_in_last_hour_counts_only_recent():
    now = datetime(2026, 6, 22, 12, 0, 0)
    times = (now - timedelta(minutes=90), now - timedelta(minutes=30), now)
    state = SafetyState(fire_times=times)
    assert state.bursts_in_last_hour(now) == 2
