from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time, timedelta

from taubenschreck.core.config import SafetyConfig
from taubenschreck.core.state import SafetyState
from taubenschreck.core.types import Detection, SafetyDecision


def _within_window(now_t: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= now_t <= end
    return now_t >= start or now_t <= end  # window crosses midnight


def evaluate(
    detections: list[Detection],
    now: datetime,
    state: SafetyState,
    config: SafetyConfig,
) -> tuple[SafetyDecision, SafetyState]:
    person_present = any(
        d.label == config.person_label and d.confidence >= config.person_min_confidence
        for d in detections
    )
    pigeon_present = any(
        d.label == config.pigeon_label and d.confidence >= config.pigeon_min_confidence
        for d in detections
    )

    streak = state.consecutive_pigeon_frames + 1 if pigeon_present else 0
    state = replace(state, consecutive_pigeon_frames=streak)

    def suppress(reason: str) -> tuple[SafetyDecision, SafetyState]:
        return SafetyDecision(fire=False, reason=reason), state

    if person_present:
        return suppress("person_present")
    if not state.armed:
        return suppress("disarmed")
    if not _within_window(now.time(), config.active_start, config.active_end):
        return suppress("outside_window")
    if not pigeon_present:
        return suppress("no_pigeon")
    if streak < config.persistence_frames:
        return suppress("below_persistence")
    if state.last_fire_at is not None and (now - state.last_fire_at).total_seconds() < config.cooldown_seconds:
        return suppress("cooldown")
    if state.bursts_in_last_hour(now) >= config.max_bursts_per_hour:
        return suppress("rate_limited")

    cutoff = now - timedelta(hours=1)
    fire_times = tuple(t for t in state.fire_times if t >= cutoff) + (now,)
    fired = replace(state, last_fire_at=now, fire_times=fire_times, consecutive_pigeon_frames=0)
    return SafetyDecision(fire=True, reason="fire"), fired
