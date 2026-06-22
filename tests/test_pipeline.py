import threading
from datetime import datetime

import numpy as np

from taubenschreck.core.config import AppConfig, SafetyConfig
from taubenschreck.core.state import SafetyState
from taubenschreck.core.types import Detection, Event
from taubenschreck.detector.model import FakeDetector
from taubenschreck.detector.pipeline import Pipeline
from taubenschreck.detector.sprayer.mock import MockPump


def _frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def _pipeline(scripted, state, events):
    cfg = AppConfig(safety=SafetyConfig(persistence_frames=2, cooldown_seconds=0.0))
    return Pipeline(
        source=None,
        detector=FakeDetector(scripted),
        sprayer=MockPump(),
        config=cfg,
        state=state,
        on_event=lambda e, f, d: events.append(e),
        clock=lambda: datetime(2026, 6, 22, 12, 0, 0),
    )


def test_process_fires_after_persistence_and_emits_event():
    pigeon = [Detection("bird", 0.9, (0, 0, 10, 10))]
    events: list[Event] = []
    pipe = _pipeline([pigeon, pigeon], SafetyState(armed=True), events)
    assert pipe.process(_frame()) is None       # frame 1: streak 1 < 2
    fired = pipe.process(_frame())               # frame 2: streak 2 -> fire
    assert fired is not None and fired.event_type.value == "fire"
    assert len(events) == 1
    assert pipe.sprayer.fires == [pipe.config.safety.burst_seconds]


def test_process_suppresses_when_person_present():
    both = [Detection("bird", 0.9, (0, 0, 10, 10)), Detection("person", 0.9, (0, 0, 5, 5))]
    events: list[Event] = []
    pipe = _pipeline([both, both, both], SafetyState(armed=True, consecutive_pigeon_frames=5), events)
    assert pipe.process(_frame()) is None
    assert pipe.sprayer.fires == []
    assert events == []


def test_run_stops_on_should_stop():
    pigeon = [Detection("bird", 0.9, (0, 0, 10, 10))]

    class _Source:
        def frames(self):
            while True:
                yield _frame()

        def close(self):
            pass

    cfg = AppConfig(safety=SafetyConfig(persistence_frames=1, cooldown_seconds=0.0))
    stop = threading.Event()
    seen = {"n": 0}

    def on_event(e, f, d):
        seen["n"] += 1
        if seen["n"] >= 2:
            stop.set()

    pipe = Pipeline(
        source=_Source(),
        detector=FakeDetector([pigeon]),
        sprayer=MockPump(),
        config=cfg,
        state=SafetyState(armed=True),
        on_event=on_event,
        clock=lambda: datetime(2026, 6, 22, 12, 0, 0),
        should_stop=stop.is_set,
    )
    pipe.run()  # must terminate
    assert seen["n"] >= 2
