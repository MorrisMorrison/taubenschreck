from datetime import datetime, time, timedelta

from taubenschreck.core.config import SafetyConfig
from taubenschreck.core.state import SafetyState
from taubenschreck.core.types import Detection
from taubenschreck.detector.safety import evaluate

CFG = SafetyConfig(persistence_frames=2, cooldown_seconds=10.0, max_bursts_per_hour=3)
NOON = datetime(2026, 6, 22, 12, 0, 0)  # inside default 07:00–21:00 window


def pigeon(conf=0.9):
    return Detection("bird", conf, (0, 0, 10, 10))


def person(conf=0.9):
    return Detection("person", conf, (0, 0, 10, 10))


def armed(**kw):
    return SafetyState(armed=True, **kw)


def test_fires_when_all_conditions_met():
    state = armed(consecutive_pigeon_frames=1)  # this frame makes it 2 == persistence
    decision, new = evaluate([pigeon()], NOON, state, CFG)
    assert decision.fire is True
    assert decision.reason == "fire"
    assert new.last_fire_at == NOON
    assert new.consecutive_pigeon_frames == 0
    assert NOON in new.fire_times


def test_person_suppresses_even_with_pigeon():
    state = armed(consecutive_pigeon_frames=5)
    decision, _ = evaluate([pigeon(), person()], NOON, state, CFG)
    assert decision.fire is False
    assert decision.reason == "person_present"


def test_person_below_threshold_does_not_suppress():
    state = armed(consecutive_pigeon_frames=1)
    weak_person = Detection("person", 0.1, (0, 0, 5, 5))  # < person_min_confidence 0.3
    decision, _ = evaluate([pigeon(), weak_person], NOON, state, CFG)
    assert decision.fire is True


def test_disarmed_suppresses():
    decision, _ = evaluate([pigeon()], NOON, SafetyState(consecutive_pigeon_frames=5), CFG)
    assert decision.reason == "disarmed"


def test_outside_window_suppresses():
    three_am = datetime(2026, 6, 22, 3, 0, 0)
    state = armed(consecutive_pigeon_frames=5)
    decision, _ = evaluate([pigeon()], three_am, state, CFG)
    assert decision.reason == "outside_window"


def test_no_pigeon_suppresses_and_resets_streak():
    state = armed(consecutive_pigeon_frames=5)
    decision, new = evaluate([], NOON, state, CFG)
    assert decision.reason == "no_pigeon"
    assert new.consecutive_pigeon_frames == 0


def test_below_persistence_suppresses_and_increments():
    state = armed(consecutive_pigeon_frames=0)  # becomes 1 < persistence 2
    decision, new = evaluate([pigeon()], NOON, state, CFG)
    assert decision.reason == "below_persistence"
    assert new.consecutive_pigeon_frames == 1


def test_cooldown_suppresses():
    state = armed(consecutive_pigeon_frames=1, last_fire_at=NOON - timedelta(seconds=5))
    decision, _ = evaluate([pigeon()], NOON, state, CFG)
    assert decision.reason == "cooldown"


def test_cooldown_elapsed_allows_fire():
    state = armed(consecutive_pigeon_frames=1, last_fire_at=NOON - timedelta(seconds=11))
    decision, _ = evaluate([pigeon()], NOON, state, CFG)
    assert decision.fire is True


def test_rate_limit_suppresses():
    recent = tuple(NOON - timedelta(minutes=m) for m in (1, 2, 3))  # 3 == max
    state = armed(consecutive_pigeon_frames=1, last_fire_at=NOON - timedelta(seconds=30), fire_times=recent)
    decision, _ = evaluate([pigeon()], NOON, state, CFG)
    assert decision.reason == "rate_limited"


def test_window_crossing_midnight():
    cfg = SafetyConfig(persistence_frames=1, active_start=time(22, 0), active_end=time(6, 0))
    one_am = datetime(2026, 6, 22, 1, 0, 0)
    decision, _ = evaluate([pigeon()], one_am, SafetyState(armed=True), cfg)
    assert decision.fire is True


def test_fire_prunes_old_fire_times():
    old = NOON - timedelta(minutes=90)
    state = armed(consecutive_pigeon_frames=1, last_fire_at=NOON - timedelta(seconds=30), fire_times=(old,))
    _, new = evaluate([pigeon()], NOON, state, CFG)
    assert old not in new.fire_times
    assert NOON in new.fire_times
