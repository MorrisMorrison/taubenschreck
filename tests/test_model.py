import numpy as np

from taubenschreck.core.types import Detection
from taubenschreck.detector.model import FakeDetector, YoloDetector


def test_fake_detector_returns_scripted_in_order():
    a = [Detection("bird", 0.9, (0, 0, 1, 1))]
    b = []
    det = FakeDetector([a, b])
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert det.detect(frame) == a
    assert det.detect(frame) == b


def test_fake_detector_repeats_last_when_exhausted():
    last = [Detection("person", 0.8, (0, 0, 1, 1))]
    det = FakeDetector([last])
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    assert det.detect(frame) == last
    assert det.detect(frame) == last  # repeats


def test_yolo_detector_box_mapping(monkeypatch):
    """YoloDetector maps ultralytics box tensors to Detection objects and filters by confidence."""

    class _FakeBox:
        def __init__(self, conf, cls, xyxy):
            self.conf = [conf]
            self.cls = [cls]
            self.xyxy = [xyxy]

    class _FakeResult:
        def __init__(self, boxes):
            self.boxes = boxes

    class _FakeModel:
        names = {0: "bird", 1: "person"}

        def __call__(self, frame, verbose=False):
            above = _FakeBox(0.85, 0, (10, 20, 50, 60))   # bird, above threshold
            below = _FakeBox(0.10, 1, (0, 0, 5, 5))       # person, below threshold
            return [_FakeResult([above, below])]

    det = YoloDetector.__new__(YoloDetector)
    det._model = _FakeModel()
    det._min_conf = 0.25

    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    results = det.detect(frame)

    assert len(results) == 1
    assert results[0] == Detection(label="bird", confidence=0.85, bbox=(10, 20, 50, 60))
