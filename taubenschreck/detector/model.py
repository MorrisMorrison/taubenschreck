from __future__ import annotations

from typing import Protocol

import numpy as np

from taubenschreck.core.types import Detection


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class FakeDetector:
    """Deterministic detector for tests: returns scripted detections per call."""

    def __init__(self, scripted: list[list[Detection]]):
        if not scripted:
            raise ValueError("scripted must be non-empty")
        self._scripted = scripted
        self._i = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        idx = min(self._i, len(self._scripted) - 1)
        self._i += 1
        return self._scripted[idx]


class YoloDetector:
    """Wraps ultralytics YOLO. Heavy import is deferred to construction."""

    def __init__(self, weights: str = "yolov8n.pt", min_confidence: float = 0.25):
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._min_conf = min_confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self._model(frame, verbose=False)[0]
        out: list[Detection] = []
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < self._min_conf:
                continue
            label = self._model.names[int(box.cls[0])]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            out.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return out
