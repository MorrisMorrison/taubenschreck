from __future__ import annotations

from typing import Iterator

import cv2
import numpy as np


class VideoFileSource:
    def __init__(self, path: str):
        self._path = path
        self._cap: cv2.VideoCapture | None = None

    def frames(self) -> Iterator[np.ndarray]:
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video: {self._path}")
        try:
            while True:
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
