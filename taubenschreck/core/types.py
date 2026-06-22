from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2


class EventType(str, Enum):
    FIRE = "fire"
    SUPPRESS = "suppress"


@dataclass(frozen=True)
class Event:
    timestamp: datetime
    event_type: EventType
    reason: str
    snapshot_path: str | None = None
    detections: tuple[Detection, ...] = ()


@dataclass(frozen=True)
class SafetyDecision:
    fire: bool
    reason: str
