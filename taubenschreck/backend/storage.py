from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def save_snapshot(snapshot_dir: str | Path, ts: datetime, frame: np.ndarray) -> str:
    directory = Path(snapshot_dir)
    directory.mkdir(parents=True, exist_ok=True)
    name = ts.strftime("%Y%m%d-%H%M%S-%f") + ".jpg"
    out = directory / name
    if not cv2.imwrite(str(out), frame):
        raise RuntimeError(f"failed to write snapshot: {out}")
    return f"{directory.name}/{name}"
