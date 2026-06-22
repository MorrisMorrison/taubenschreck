from __future__ import annotations

import tomllib
from dataclasses import dataclass, fields, replace
from datetime import time
from pathlib import Path


@dataclass(frozen=True)
class SafetyConfig:
    pigeon_label: str = "bird"
    person_label: str = "person"
    pigeon_min_confidence: float = 0.5
    person_min_confidence: float = 0.3
    persistence_frames: int = 3
    active_start: time = time(7, 0)
    active_end: time = time(21, 0)
    burst_seconds: float = 1.5
    cooldown_seconds: float = 10.0
    max_bursts_per_hour: int = 30


@dataclass(frozen=True)
class AppConfig:
    safety: SafetyConfig = SafetyConfig()
    frame_source: str = "webcam"          # "webcam" | "videofile"
    video_path: str | None = None
    webcam_index: int = 0
    sprayer: str = "mock"                 # "mock" | "gpio" (gpio = Phase 2)
    model_weights: str = "yolov8n.pt"
    ntfy_url: str | None = None
    snapshot_dir: str = "snapshots"
    db_path: str = "taubenschreck.db"
    host: str = "127.0.0.1"
    port: int = 8000


def _parse_time(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


def _filter(cls, data: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def load_config(path: str | Path) -> AppConfig:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    safety_raw = data.pop("safety", {})
    time_keys = {"active_start", "active_end"}
    safety_filtered = {k: v for k, v in _filter(SafetyConfig, safety_raw).items() if k not in time_keys}
    safety = SafetyConfig(**safety_filtered)
    if "active_start" in safety_raw:
        safety = replace(safety, active_start=_parse_time(safety_raw["active_start"]))
    if "active_end" in safety_raw:
        safety = replace(safety, active_end=_parse_time(safety_raw["active_end"]))
    return AppConfig(safety=safety, **_filter(AppConfig, data))
