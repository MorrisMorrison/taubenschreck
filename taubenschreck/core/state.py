from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class SafetyState:
    armed: bool = False
    consecutive_pigeon_frames: int = 0
    last_fire_at: datetime | None = None
    fire_times: tuple[datetime, ...] = ()

    def bursts_in_last_hour(self, now: datetime) -> int:
        cutoff = now - timedelta(hours=1)
        return sum(1 for t in self.fire_times if t >= cutoff)
