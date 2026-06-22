from __future__ import annotations

from typing import Protocol


class Sprayer(Protocol):
    def fire(self, duration_seconds: float) -> None: ...
    def close(self) -> None: ...
