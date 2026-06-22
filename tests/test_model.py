import numpy as np

from taubenschreck.core.types import Detection
from taubenschreck.detector.model import FakeDetector


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
