from __future__ import annotations

from typing import Callable, Iterator

import cv2
import numpy as np


class WebcamSource:
    def __init__(self, index: int = 0, should_stop: Callable[[], bool] = lambda: False):
        self._index = index
        self._should_stop = should_stop
        self._cap: cv2.VideoCapture | None = None

    def frames(self) -> Iterator[np.ndarray]:
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open webcam index {self._index}")
        try:
            while not self._should_stop():
                ok, frame = self._cap.read()
                if not ok:
                    break
                yield frame
        finally:
            self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
