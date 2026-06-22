from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MockPump:
    fires: list[float] = field(default_factory=list)

    def fire(self, duration_seconds: float) -> None:
        self.fires.append(duration_seconds)
        logger.info("MOCK FIRE for %.2fs", duration_seconds)

    def close(self) -> None:
        logger.debug("MockPump closed")
