from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np

from taubenschreck.core.config import AppConfig
from taubenschreck.core.state import SafetyState
from taubenschreck.core.types import Detection, Event, EventType
from taubenschreck.detector.model import Detector
from taubenschreck.detector.safety import evaluate
from taubenschreck.detector.sprayer.base import Sprayer


def _noop_event(event: Event, frame: np.ndarray, detections: list[Detection]) -> None:
    return None


@dataclass
class Pipeline:
    source: object                      # FrameSource (duck-typed; None in unit tests)
    detector: Detector
    sprayer: Sprayer
    config: AppConfig
    state: SafetyState
    on_event: Callable[[Event, np.ndarray, list[Detection]], None] = _noop_event
    clock: Callable[[], datetime] = datetime.now
    lock: threading.Lock = field(default_factory=threading.Lock)
    should_stop: Callable[[], bool] = lambda: False

    def process(self, frame: np.ndarray) -> Event | None:
        detections = self.detector.detect(frame)
        now = self.clock()
        with self.lock:
            decision, self.state = evaluate(detections, now, self.state, self.config.safety)
        if not decision.fire:
            return None
        self.sprayer.fire(self.config.safety.burst_seconds)
        event = Event(now, EventType.FIRE, decision.reason, None, tuple(detections))
        self.on_event(event, frame, detections)
        return event

    def run(self) -> None:
        for frame in self.source.frames():
            if self.should_stop():
                break
            self.process(frame)
        self.source.close()
