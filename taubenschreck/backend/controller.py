from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime

from taubenschreck.backend.recorder import Recorder
from taubenschreck.core.config import SafetyConfig
from taubenschreck.core.types import Event, EventType
from taubenschreck.detector.pipeline import Pipeline
from taubenschreck.detector.sprayer.base import Sprayer


class Controller:
    def __init__(self, pipeline: Pipeline, recorder: Recorder, sprayer: Sprayer, safety_config: SafetyConfig):
        self._pipeline = pipeline
        self._recorder = recorder
        self._sprayer = sprayer
        self._cfg = safety_config
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # route pipeline fire events into the recorder
        self._pipeline.on_event = lambda e, f, d: self._recorder.record(e, f, d)
        self._pipeline.should_stop = self._stop.is_set

    def is_armed(self) -> bool:
        return self._pipeline.state.armed

    def arm(self) -> None:
        with self._pipeline.lock:
            self._pipeline.state = replace(self._pipeline.state, armed=True)

    def disarm(self) -> None:
        with self._pipeline.lock:
            self._pipeline.state = replace(self._pipeline.state, armed=False)

    def test_fire(self) -> None:
        self._sprayer.fire(self._cfg.burst_seconds)
        event = Event(datetime.now(), EventType.FIRE, "manual_test")
        self._recorder.record(event, None, [])

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._pipeline.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
