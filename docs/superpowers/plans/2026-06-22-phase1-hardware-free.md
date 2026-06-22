# Taubenschreck — Phase 1 (Hardware-Free) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete Taubenschreck software stack so it runs end-to-end on a PC with a webcam and a mock pump — detecting pigeons, applying the safety gate, recording fire events with snapshots, pushing ntfy notifications, and serving a control dashboard — with zero Raspberry Pi hardware.

**Architecture:** Four decoupled units behind interfaces. A `Pipeline` pulls frames from a `FrameSource`, runs a `Detector`, applies a pure `evaluate()` safety gate, and fires a `Sprayer`. A `Controller` runs that loop in a background thread and exposes arm/disarm/test-fire. A `FastAPI` app records events to SQLite + snapshot files, pushes ntfy notifications via a `Recorder`, and serves a static dashboard. Swapping `Webcam`→`PiCamera` and `MockPump`→`GpioPump` in Phase 2 is a config change, not a logic change.

**Tech Stack:** Python 3.11+ · ultralytics (YOLOv8n) + OpenCV · FastAPI + Uvicorn · SQLite (stdlib) · tomllib (stdlib) · httpx (ntfy) · pytest.

## Global Constraints

- **Python:** `>=3.11` (uses stdlib `tomllib`). Dev machine has 3.12.
- **Public repo:** no infra hostnames/IPs/secrets in code or tests. The ntfy URL is supplied only via local config (`config/config.toml`, gitignored); never hardcode it.
- **License/headers:** MIT (repo root `LICENSE`). No per-file headers required.
- **Deterrence only:** the system never does anything but log/fire a brief water burst. No escalation logic in Phase 1.
- **Boot state = DISARMED.** Every fresh `SafetyState` starts `armed=False`. Arming is an explicit action.
- **Humans always win:** any `person` detection at or above `person_min_confidence` hard-suppresses firing.
- **Package layout:** the spec's logical units (`detector/`, `backend/`, `notifier/`, `dashboard/`) are realized as subpackages under an importable `taubenschreck/` package, with shared domain types in `taubenschreck/core/`. Tests live in top-level `tests/`.
- **Every task ends green** (`pytest` passes) and is committed.

---

## File Structure

```
taubenschreck/
├─ pyproject.toml                      # package + deps + pytest config
├─ .gitignore                          # venv, *.db, snapshots/, config/config.toml
├─ config/
│  ├─ config.example.toml              # committed template (no secrets)
│  └─ config.toml                      # local, gitignored
├─ taubenschreck/
│  ├─ __init__.py
│  ├─ core/
│  │  ├─ __init__.py
│  │  ├─ types.py                      # Detection, SafetyDecision, Event, EventType
│  │  ├─ state.py                      # SafetyState (frozen)
│  │  └─ config.py                     # SafetyConfig, AppConfig, load_config()
│  ├─ detector/
│  │  ├─ __init__.py
│  │  ├─ safety.py                     # evaluate() — pure safety gate
│  │  ├─ model.py                      # Detector protocol, YoloDetector, FakeDetector
│  │  ├─ pipeline.py                   # Pipeline
│  │  ├─ sources/
│  │  │  ├─ __init__.py
│  │  │  ├─ base.py                    # FrameSource protocol
│  │  │  ├─ videofile.py               # VideoFileSource
│  │  │  └─ webcam.py                  # WebcamSource
│  │  └─ sprayer/
│  │     ├─ __init__.py
│  │     ├─ base.py                    # Sprayer protocol
│  │     └─ mock.py                    # MockPump
│  ├─ backend/
│  │  ├─ __init__.py
│  │  ├─ db.py                         # SQLite event store
│  │  ├─ storage.py                    # snapshot save
│  │  ├─ recorder.py                   # Recorder (db + storage + notifier)
│  │  ├─ controller.py                 # Controller (threaded loop, arm/disarm/test-fire)
│  │  └─ app.py                        # FastAPI factory: read API + control + static
│  ├─ notifier/
│  │  ├─ __init__.py
│  │  └─ ntfy.py                       # NtfyNotifier
│  ├─ dashboard/
│  │  └─ static/{index.html,app.js,style.css}
│  └─ main.py                          # entrypoint: build from config, run uvicorn
└─ tests/
   ├─ test_core_types.py
   ├─ test_config.py
   ├─ test_safety.py
   ├─ test_sprayer.py
   ├─ test_sources.py
   ├─ test_model.py
   ├─ test_pipeline.py
   ├─ test_db.py
   ├─ test_storage_recorder.py
   ├─ test_notifier.py
   ├─ test_controller.py
   └─ test_app.py
```

---

### Task 1: Project scaffolding & tooling

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `taubenschreck/__init__.py`, `taubenschreck/core/__init__.py`, `taubenschreck/detector/__init__.py`, `taubenschreck/detector/sources/__init__.py`, `taubenschreck/detector/sprayer/__init__.py`, `taubenschreck/backend/__init__.py`, `taubenschreck/notifier/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "taubenschreck"
version = "0.1.0"
description = "Vision-triggered water sentry that humanely keeps pigeons off a balcony."
requires-python = ">=3.11"
dependencies = [
    "numpy>=1.26",
    "opencv-python>=4.9",
    "ultralytics>=8.1",
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "httpx>=0.27",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.4"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["taubenschreck*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# venv & build
.venv/
*.egg-info/
__pycache__/
# runtime artifacts (never commit)
*.db
snapshots/
config/config.toml
# model weights download
*.pt
```

- [ ] **Step 3: Create the empty package files**

Create each `__init__.py` listed above with a single line:

```python
"""Taubenschreck package."""
```

- [ ] **Step 4: Write a smoke test** in `tests/test_smoke.py`

```python
import taubenschreck


def test_package_imports():
    assert taubenschreck is not None
```

- [ ] **Step 5: Create venv, install, run the smoke test**

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest tests/test_smoke.py -v
```
Expected: `1 passed`. (First install pulls torch/ultralytics — can take a few minutes.)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore taubenschreck/ tests/test_smoke.py
git commit -m "chore: scaffold taubenschreck package and tooling"
```

---

### Task 2: Core domain types

**Files:**
- Create: `taubenschreck/core/types.py`, `taubenschreck/core/state.py`
- Test: `tests/test_core_types.py`

**Interfaces:**
- Produces:
  - `Detection(label: str, confidence: float, bbox: tuple[int,int,int,int])` — frozen.
  - `EventType` str-enum: `FIRE="fire"`, `SUPPRESS="suppress"`.
  - `Event(timestamp: datetime, event_type: EventType, reason: str, snapshot_path: str|None=None, detections: tuple[Detection,...]=())` — frozen.
  - `SafetyDecision(fire: bool, reason: str)` — frozen.
  - `SafetyState(armed=False, consecutive_pigeon_frames=0, last_fire_at: datetime|None=None, fire_times: tuple[datetime,...]=())` — frozen, with `bursts_in_last_hour(now) -> int`.

- [ ] **Step 1: Write the failing test** in `tests/test_core_types.py`

```python
from datetime import datetime, timedelta

from taubenschreck.core.types import Detection, Event, EventType, SafetyDecision
from taubenschreck.core.state import SafetyState


def test_detection_is_frozen():
    d = Detection(label="bird", confidence=0.9, bbox=(0, 0, 10, 10))
    assert d.label == "bird"


def test_event_defaults():
    now = datetime(2026, 6, 22, 12, 0, 0)
    e = Event(timestamp=now, event_type=EventType.FIRE, reason="fire")
    assert e.snapshot_path is None
    assert e.detections == ()
    assert e.event_type.value == "fire"


def test_decision_fields():
    assert SafetyDecision(fire=True, reason="fire").fire is True


def test_state_boot_disarmed():
    assert SafetyState().armed is False


def test_bursts_in_last_hour_counts_only_recent():
    now = datetime(2026, 6, 22, 12, 0, 0)
    times = (now - timedelta(minutes=90), now - timedelta(minutes=30), now)
    state = SafetyState(fire_times=times)
    assert state.bursts_in_last_hour(now) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_core_types.py -v`
Expected: FAIL (`ModuleNotFoundError: taubenschreck.core.types`).

- [ ] **Step 3: Implement `taubenschreck/core/types.py`**

```python
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
```

- [ ] **Step 4: Implement `taubenschreck/core/state.py`**

```python
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
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_core_types.py -v`
Expected: `5 passed`.

- [ ] **Step 6: Commit**

```bash
git add taubenschreck/core/types.py taubenschreck/core/state.py tests/test_core_types.py
git commit -m "feat(core): add domain types and safety state"
```

---

### Task 3: Configuration

**Files:**
- Create: `taubenschreck/core/config.py`, `config/config.example.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `SafetyConfig(pigeon_label="bird", person_label="person", pigeon_min_confidence=0.5, person_min_confidence=0.3, persistence_frames=3, active_start: time=time(7,0), active_end: time=time(21,0), burst_seconds=1.5, cooldown_seconds=10.0, max_bursts_per_hour=30)` — frozen.
  - `AppConfig(safety: SafetyConfig, frame_source="webcam", video_path: str|None=None, webcam_index=0, sprayer="mock", model_weights="yolov8n.pt", ntfy_url: str|None=None, snapshot_dir="snapshots", db_path="taubenschreck.db", host="127.0.0.1", port=8000)` — frozen.
  - `load_config(path) -> AppConfig`.

- [ ] **Step 1: Write the failing test** in `tests/test_config.py`

```python
from datetime import time

from taubenschreck.core.config import AppConfig, SafetyConfig, load_config


def test_defaults():
    cfg = SafetyConfig()
    assert cfg.pigeon_label == "bird"
    assert cfg.active_start == time(7, 0)
    assert cfg.persistence_frames == 3


def test_load_config_parses_times_and_nested(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text(
        """
frame_source = "videofile"
video_path = "clip.mp4"
ntfy_url = "http://example.test/topic"

[safety]
persistence_frames = 5
active_start = "06:30"
active_end = "22:15"
cooldown_seconds = 8.0
""",
        encoding="utf-8",
    )
    cfg = load_config(p)
    assert isinstance(cfg, AppConfig)
    assert cfg.frame_source == "videofile"
    assert cfg.video_path == "clip.mp4"
    assert cfg.ntfy_url == "http://example.test/topic"
    assert cfg.safety.persistence_frames == 5
    assert cfg.safety.active_start == time(6, 30)
    assert cfg.safety.active_end == time(22, 15)
    assert cfg.safety.cooldown_seconds == 8.0


def test_load_config_uses_defaults_for_missing(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text("frame_source = \"webcam\"\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.safety.persistence_frames == 3
    assert cfg.webcam_index == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `taubenschreck/core/config.py`**

```python
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
    safety = SafetyConfig(**_filter(SafetyConfig, safety_raw))
    if "active_start" in safety_raw:
        safety = replace(safety, active_start=_parse_time(safety_raw["active_start"]))
    if "active_end" in safety_raw:
        safety = replace(safety, active_end=_parse_time(safety_raw["active_end"]))
    return AppConfig(safety=safety, **_filter(AppConfig, data))
```

- [ ] **Step 4: Write `config/config.example.toml`** (committed template — no secrets)

```toml
# Copy to config/config.toml and edit. config.toml is gitignored.
frame_source = "webcam"    # "webcam" | "videofile"
# video_path = "samples/pigeons.mp4"
webcam_index = 0
sprayer = "mock"           # Phase 1 is always "mock"
model_weights = "yolov8n.pt"

# Self-hosted ntfy topic URL. Leave unset to disable push.
# ntfy_url = "https://ntfy.example.test/taubenschreck"

snapshot_dir = "snapshots"
db_path = "taubenschreck.db"
host = "127.0.0.1"
port = 8000

[safety]
pigeon_min_confidence = 0.5
person_min_confidence = 0.3
persistence_frames = 3
active_start = "07:00"
active_end = "21:00"
burst_seconds = 1.5
cooldown_seconds = 10.0
max_bursts_per_hour = 30
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_config.py -v`
Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add taubenschreck/core/config.py config/config.example.toml tests/test_config.py
git commit -m "feat(core): add config model and TOML loader"
```

---

### Task 4: Safety gate (the critical unit)

**Files:**
- Create: `taubenschreck/detector/safety.py`
- Test: `tests/test_safety.py`

**Interfaces:**
- Consumes: `Detection`, `SafetyDecision` (Task 2), `SafetyState` (Task 2), `SafetyConfig` (Task 3).
- Produces: `evaluate(detections: list[Detection], now: datetime, state: SafetyState, config: SafetyConfig) -> tuple[SafetyDecision, SafetyState]`. Pure: no I/O, no mutation of inputs; returns a new state. Suppress-reason precedence: `person_present` > `disarmed` > `outside_window` > `no_pigeon` > `below_persistence` > `cooldown` > `rate_limited`. On fire: returns `reason="fire"`, sets `last_fire_at=now`, appends `now` to `fire_times` (pruned to last hour), resets `consecutive_pigeon_frames=0`.

- [ ] **Step 1: Write the failing tests** in `tests/test_safety.py` (exhaustive — one per branch)

```python
from datetime import datetime, time, timedelta

from taubenschreck.core.config import SafetyConfig
from taubenschreck.core.state import SafetyState
from taubenschreck.core.types import Detection
from taubenschreck.detector.safety import evaluate

CFG = SafetyConfig(persistence_frames=2, cooldown_seconds=10.0, max_bursts_per_hour=3)
NOON = datetime(2026, 6, 22, 12, 0, 0)  # inside default 07:00–21:00 window


def pigeon(conf=0.9):
    return Detection("bird", conf, (0, 0, 10, 10))


def person(conf=0.9):
    return Detection("person", conf, (0, 0, 10, 10))


def armed(**kw):
    return SafetyState(armed=True, **kw)


def test_fires_when_all_conditions_met():
    state = armed(consecutive_pigeon_frames=1)  # this frame makes it 2 == persistence
    decision, new = evaluate([pigeon()], NOON, state, CFG)
    assert decision.fire is True
    assert decision.reason == "fire"
    assert new.last_fire_at == NOON
    assert new.consecutive_pigeon_frames == 0
    assert NOON in new.fire_times


def test_person_suppresses_even_with_pigeon():
    state = armed(consecutive_pigeon_frames=5)
    decision, _ = evaluate([pigeon(), person()], NOON, state, CFG)
    assert decision.fire is False
    assert decision.reason == "person_present"


def test_person_below_threshold_does_not_suppress():
    state = armed(consecutive_pigeon_frames=1)
    weak_person = Detection("person", 0.1, (0, 0, 5, 5))  # < person_min_confidence 0.3
    decision, _ = evaluate([pigeon(), weak_person], NOON, state, CFG)
    assert decision.fire is True


def test_disarmed_suppresses():
    decision, _ = evaluate([pigeon()], NOON, SafetyState(consecutive_pigeon_frames=5), CFG)
    assert decision.reason == "disarmed"


def test_outside_window_suppresses():
    three_am = datetime(2026, 6, 22, 3, 0, 0)
    state = armed(consecutive_pigeon_frames=5)
    decision, _ = evaluate([pigeon()], three_am, state, CFG)
    assert decision.reason == "outside_window"


def test_no_pigeon_suppresses_and_resets_streak():
    state = armed(consecutive_pigeon_frames=5)
    decision, new = evaluate([], NOON, state, CFG)
    assert decision.reason == "no_pigeon"
    assert new.consecutive_pigeon_frames == 0


def test_below_persistence_suppresses_and_increments():
    state = armed(consecutive_pigeon_frames=0)  # becomes 1 < persistence 2
    decision, new = evaluate([pigeon()], NOON, state, CFG)
    assert decision.reason == "below_persistence"
    assert new.consecutive_pigeon_frames == 1


def test_cooldown_suppresses():
    state = armed(consecutive_pigeon_frames=1, last_fire_at=NOON - timedelta(seconds=5))
    decision, _ = evaluate([pigeon()], NOON, state, CFG)
    assert decision.reason == "cooldown"


def test_cooldown_elapsed_allows_fire():
    state = armed(consecutive_pigeon_frames=1, last_fire_at=NOON - timedelta(seconds=11))
    decision, _ = evaluate([pigeon()], NOON, state, CFG)
    assert decision.fire is True


def test_rate_limit_suppresses():
    recent = tuple(NOON - timedelta(minutes=m) for m in (1, 2, 3))  # 3 == max
    state = armed(consecutive_pigeon_frames=1, last_fire_at=NOON - timedelta(seconds=30), fire_times=recent)
    decision, _ = evaluate([pigeon()], NOON, state, CFG)
    assert decision.reason == "rate_limited"


def test_window_crossing_midnight():
    cfg = SafetyConfig(persistence_frames=1, active_start=time(22, 0), active_end=time(6, 0))
    one_am = datetime(2026, 6, 22, 1, 0, 0)
    decision, _ = evaluate([pigeon()], one_am, SafetyState(armed=True), cfg)
    assert decision.fire is True


def test_fire_prunes_old_fire_times():
    old = NOON - timedelta(minutes=90)
    state = armed(consecutive_pigeon_frames=1, last_fire_at=NOON - timedelta(seconds=30), fire_times=(old,))
    _, new = evaluate([pigeon()], NOON, state, CFG)
    assert old not in new.fire_times
    assert NOON in new.fire_times
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_safety.py -v`
Expected: FAIL (`ModuleNotFoundError: taubenschreck.detector.safety`).

- [ ] **Step 3: Implement `taubenschreck/detector/safety.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_safety.py -v`
Expected: `12 passed`.

- [ ] **Step 5: Commit**

```bash
git add taubenschreck/detector/safety.py tests/test_safety.py
git commit -m "feat(detector): add pure safety-gate evaluate() with exhaustive tests"
```

---

### Task 5: Sprayer interface + MockPump

**Files:**
- Create: `taubenschreck/detector/sprayer/base.py`, `taubenschreck/detector/sprayer/mock.py`
- Test: `tests/test_sprayer.py`

**Interfaces:**
- Produces:
  - `Sprayer` Protocol: `fire(duration_seconds: float) -> None`, `close() -> None`.
  - `MockPump` with `fires: list[float]` recording each burst duration; logs at INFO.

- [ ] **Step 1: Write the failing test** in `tests/test_sprayer.py`

```python
import logging

from taubenschreck.detector.sprayer.mock import MockPump


def test_mock_records_fire_durations():
    pump = MockPump()
    pump.fire(1.5)
    pump.fire(2.0)
    assert pump.fires == [1.5, 2.0]


def test_mock_logs_on_fire(caplog):
    pump = MockPump()
    with caplog.at_level(logging.INFO):
        pump.fire(1.0)
    assert any("FIRE" in r.message for r in caplog.records)


def test_mock_close_is_safe():
    MockPump().close()  # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_sprayer.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `taubenschreck/detector/sprayer/base.py`**

```python
from __future__ import annotations

from typing import Protocol


class Sprayer(Protocol):
    def fire(self, duration_seconds: float) -> None: ...
    def close(self) -> None: ...
```

- [ ] **Step 4: Implement `taubenschreck/detector/sprayer/mock.py`**

```python
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
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_sprayer.py -v`
Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add taubenschreck/detector/sprayer/ tests/test_sprayer.py
git commit -m "feat(detector): add Sprayer protocol and MockPump"
```

---

### Task 6: FrameSource interface + VideoFile + Webcam

**Files:**
- Create: `taubenschreck/detector/sources/base.py`, `taubenschreck/detector/sources/videofile.py`, `taubenschreck/detector/sources/webcam.py`
- Test: `tests/test_sources.py`

**Interfaces:**
- Produces:
  - `FrameSource` Protocol: `frames() -> Iterator[np.ndarray]`, `close() -> None`.
  - `VideoFileSource(path: str)` — yields frames until the file ends.
  - `WebcamSource(index: int = 0)` — yields frames from a camera until stopped (manual-only; no automated test).

- [ ] **Step 1: Write the failing test** in `tests/test_sources.py`

```python
import cv2
import numpy as np

from taubenschreck.detector.sources.videofile import VideoFileSource


def _write_video(path, n_frames=5, w=64, h=48):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (w, h))
    for i in range(n_frames):
        frame = np.full((h, w, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_videofile_yields_all_frames(tmp_path):
    vid = tmp_path / "clip.avi"
    _write_video(vid, n_frames=5)
    src = VideoFileSource(str(vid))
    frames = list(src.frames())
    src.close()
    assert len(frames) == 5
    assert frames[0].shape == (48, 64, 3)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_sources.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `taubenschreck/detector/sources/base.py`**

```python
from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np


class FrameSource(Protocol):
    def frames(self) -> Iterator[np.ndarray]: ...
    def close(self) -> None: ...
```

- [ ] **Step 4: Implement `taubenschreck/detector/sources/videofile.py`**

```python
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
```

- [ ] **Step 5: Implement `taubenschreck/detector/sources/webcam.py`**

```python
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
```

- [ ] **Step 6: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_sources.py -v`
Expected: `1 passed`.

- [ ] **Step 7: Manual webcam smoke (optional, needs a camera)**

```bash
.venv/Scripts/python -c "from taubenschreck.detector.sources.webcam import WebcamSource; import itertools; s=WebcamSource(0); fs=list(itertools.islice(s.frames(),3)); s.close(); print('captured', len(fs), 'frames', fs[0].shape)"
```
Expected: prints `captured 3 frames (H, W, 3)`.

- [ ] **Step 8: Commit**

```bash
git add taubenschreck/detector/sources/ tests/test_sources.py
git commit -m "feat(detector): add FrameSource protocol with VideoFile and Webcam"
```

---

### Task 7: Detector model wrapper + FakeDetector

**Files:**
- Create: `taubenschreck/detector/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `Detection` (Task 2).
- Produces:
  - `Detector` Protocol: `detect(frame: np.ndarray) -> list[Detection]`.
  - `YoloDetector(weights="yolov8n.pt", min_confidence=0.25)` — lazy-imports ultralytics in `__init__`; maps YOLO boxes to `Detection`.
  - `FakeDetector(scripted: list[list[Detection]])` — returns the next scripted list per `detect()` call, repeating the last once exhausted. Used by all downstream tests.

- [ ] **Step 1: Write the failing test** in `tests/test_model.py`

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_model.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `taubenschreck/detector/model.py`**

```python
from __future__ import annotations

from typing import Protocol

import numpy as np

from taubenschreck.core.types import Detection


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class FakeDetector:
    """Deterministic detector for tests: returns scripted detections per call."""

    def __init__(self, scripted: list[list[Detection]]):
        if not scripted:
            raise ValueError("scripted must be non-empty")
        self._scripted = scripted
        self._i = 0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        idx = min(self._i, len(self._scripted) - 1)
        self._i += 1
        return self._scripted[idx]


class YoloDetector:
    """Wraps ultralytics YOLO. Heavy import is deferred to construction."""

    def __init__(self, weights: str = "yolov8n.pt", min_confidence: float = 0.25):
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._min_conf = min_confidence

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self._model(frame, verbose=False)[0]
        out: list[Detection] = []
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < self._min_conf:
                continue
            label = self._model.names[int(box.cls[0])]
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            out.append(Detection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return out
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_model.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Manual YOLO smoke (optional, downloads weights ~6 MB on first run)**

```bash
.venv/Scripts/python -c "import numpy as np; from taubenschreck.detector.model import YoloDetector; d=YoloDetector(); print(d.detect(np.zeros((480,640,3),dtype=np.uint8)))"
```
Expected: prints `[]` (no objects in a black frame) without error.

- [ ] **Step 6: Commit**

```bash
git add taubenschreck/detector/model.py tests/test_model.py
git commit -m "feat(detector): add Detector protocol, YoloDetector, and FakeDetector"
```

---

### Task 8: Pipeline

**Files:**
- Create: `taubenschreck/detector/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `FrameSource`, `Detector`, `Sprayer`, `evaluate()`, `AppConfig`, `SafetyState`, `Event`, `EventType`, `Detection`.
- Produces:
  - `Pipeline(source, detector, sprayer, config: AppConfig, state: SafetyState, on_event: Callable[[Event, np.ndarray, list[Detection]], None]=..., clock: Callable[[], datetime]=datetime.now, lock: threading.Lock=..., should_stop: Callable[[], bool]=lambda: False)`.
  - `Pipeline.process(frame) -> Event | None` — runs detect → evaluate → (fire + emit). Returns the `Event` on fire else `None`. Thread-safe via `lock`.
  - `Pipeline.run() -> None` — iterates `source.frames()` calling `process`, stops when `should_stop()` is true.

- [ ] **Step 1: Write the failing test** in `tests/test_pipeline.py`

```python
import threading
from datetime import datetime

import numpy as np

from taubenschreck.core.config import AppConfig, SafetyConfig
from taubenschreck.core.state import SafetyState
from taubenschreck.core.types import Detection, Event
from taubenschreck.detector.model import FakeDetector
from taubenschreck.detector.pipeline import Pipeline
from taubenschreck.detector.sprayer.mock import MockPump


def _frame():
    return np.zeros((48, 64, 3), dtype=np.uint8)


def _pipeline(scripted, state, events):
    cfg = AppConfig(safety=SafetyConfig(persistence_frames=2, cooldown_seconds=0.0))
    return Pipeline(
        source=None,
        detector=FakeDetector(scripted),
        sprayer=MockPump(),
        config=cfg,
        state=state,
        on_event=lambda e, f, d: events.append(e),
        clock=lambda: datetime(2026, 6, 22, 12, 0, 0),
    )


def test_process_fires_after_persistence_and_emits_event():
    pigeon = [Detection("bird", 0.9, (0, 0, 10, 10))]
    events: list[Event] = []
    pipe = _pipeline([pigeon, pigeon], SafetyState(armed=True), events)
    assert pipe.process(_frame()) is None       # frame 1: streak 1 < 2
    fired = pipe.process(_frame())               # frame 2: streak 2 -> fire
    assert fired is not None and fired.event_type.value == "fire"
    assert len(events) == 1
    assert pipe.sprayer.fires == [pipe.config.safety.burst_seconds]


def test_process_suppresses_when_person_present():
    both = [Detection("bird", 0.9, (0, 0, 10, 10)), Detection("person", 0.9, (0, 0, 5, 5))]
    events: list[Event] = []
    pipe = _pipeline([both, both, both], SafetyState(armed=True, consecutive_pigeon_frames=5), events)
    assert pipe.process(_frame()) is None
    assert pipe.sprayer.fires == []
    assert events == []


def test_run_stops_on_should_stop():
    pigeon = [Detection("bird", 0.9, (0, 0, 10, 10))]

    class _Source:
        def frames(self):
            while True:
                yield _frame()

        def close(self):
            pass

    cfg = AppConfig(safety=SafetyConfig(persistence_frames=1, cooldown_seconds=0.0))
    stop = threading.Event()
    seen = {"n": 0}

    def on_event(e, f, d):
        seen["n"] += 1
        if seen["n"] >= 2:
            stop.set()

    pipe = Pipeline(
        source=_Source(),
        detector=FakeDetector([pigeon]),
        sprayer=MockPump(),
        config=cfg,
        state=SafetyState(armed=True),
        on_event=on_event,
        clock=lambda: datetime(2026, 6, 22, 12, 0, 0),
        should_stop=stop.is_set,
    )
    pipe.run()  # must terminate
    assert seen["n"] >= 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `taubenschreck/detector/pipeline.py`**

```python
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np

from taubenschreck.core.config import AppConfig
from taubenschreck.core.state import SafetyState
from taubenschreck.core.types import Detection, Event, EventType
from taubenschreck.detector.model import Detector
from taubenschreck.detector.safety import evaluate
from taubenschreck.detector.sprayer.base import Sprayer


def _noop_event(event: Event, frame: np.ndarray, detections: list[Detection]) -> None:
    return None


@dataclass
class Pipeline:
    source: object                      # FrameSource (duck-typed; None in unit tests)
    detector: Detector
    sprayer: Sprayer
    config: AppConfig
    state: SafetyState
    on_event: Callable[[Event, np.ndarray, list[Detection]], None] = _noop_event
    clock: Callable[[], datetime] = datetime.now
    lock: threading.Lock = field(default_factory=threading.Lock)
    should_stop: Callable[[], bool] = lambda: False

    def process(self, frame: np.ndarray) -> Event | None:
        detections = self.detector.detect(frame)
        now = self.clock()
        with self.lock:
            decision, self.state = evaluate(detections, now, self.state, self.config.safety)
        if not decision.fire:
            return None
        self.sprayer.fire(self.config.safety.burst_seconds)
        event = Event(now, EventType.FIRE, decision.reason, None, tuple(detections))
        self.on_event(event, frame, detections)
        return event

    def run(self) -> None:
        for frame in self.source.frames():
            if self.should_stop():
                break
            self.process(frame)
        self.source.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add taubenschreck/detector/pipeline.py tests/test_pipeline.py
git commit -m "feat(detector): add Pipeline wiring source->detector->safety->sprayer"
```

---

### Task 9: Persistence — SQLite DB + snapshot storage

**Files:**
- Create: `taubenschreck/backend/db.py`, `taubenschreck/backend/storage.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Produces (`db.py`):
  - `connect(db_path: str) -> sqlite3.Connection` (row factory = `sqlite3.Row`, `check_same_thread=False`).
  - `init_db(con) -> None` — creates `events(id INTEGER PK, ts TEXT, event_type TEXT, reason TEXT, snapshot_path TEXT)`.
  - `insert_event(con, ts: str, event_type: str, reason: str, snapshot_path: str|None) -> int` (returns row id).
  - `list_events(con, limit: int = 50) -> list[dict]` (newest first).
  - `stats(con, now: datetime) -> dict` with keys `total`, `today`, `last_ts`.
- Produces (`storage.py`):
  - `save_snapshot(snapshot_dir: str|Path, ts: datetime, frame: np.ndarray) -> str` — writes a JPEG named `YYYYMMDD-HHMMSS-ffffff.jpg`, returns the relative path string.

- [ ] **Step 1: Write the failing test** in `tests/test_db.py`

```python
from datetime import datetime

import numpy as np

from taubenschreck.backend import db
from taubenschreck.backend.storage import save_snapshot


def _con(tmp_path):
    con = db.connect(str(tmp_path / "t.db"))
    db.init_db(con)
    return con


def test_insert_and_list_newest_first(tmp_path):
    con = _con(tmp_path)
    db.insert_event(con, "2026-06-22T12:00:00", "fire", "fire", "a.jpg")
    db.insert_event(con, "2026-06-22T12:05:00", "fire", "fire", "b.jpg")
    rows = db.list_events(con, limit=10)
    assert [r["snapshot_path"] for r in rows] == ["b.jpg", "a.jpg"]


def test_stats_counts_today_and_total(tmp_path):
    con = _con(tmp_path)
    now = datetime(2026, 6, 22, 12, 0, 0)
    db.insert_event(con, "2026-06-21T12:00:00", "fire", "fire", None)  # yesterday
    db.insert_event(con, "2026-06-22T08:00:00", "fire", "fire", None)  # today
    db.insert_event(con, "2026-06-22T09:00:00", "fire", "fire", None)  # today
    s = db.stats(con, now)
    assert s["total"] == 3
    assert s["today"] == 2
    assert s["last_ts"] == "2026-06-22T09:00:00"


def test_save_snapshot_writes_jpeg(tmp_path):
    frame = np.full((48, 64, 3), 128, dtype=np.uint8)
    ts = datetime(2026, 6, 22, 12, 0, 0, 123456)
    path = save_snapshot(tmp_path, ts, frame)
    assert path.endswith(".jpg")
    assert (tmp_path / path.split("/")[-1]).exists()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_db.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `taubenschreck/backend/db.py`**

```python
from __future__ import annotations

import sqlite3
from datetime import datetime


def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            event_type TEXT NOT NULL,
            reason TEXT NOT NULL,
            snapshot_path TEXT
        )
        """
    )
    con.commit()


def insert_event(con, ts: str, event_type: str, reason: str, snapshot_path: str | None) -> int:
    cur = con.execute(
        "INSERT INTO events (ts, event_type, reason, snapshot_path) VALUES (?, ?, ?, ?)",
        (ts, event_type, reason, snapshot_path),
    )
    con.commit()
    return int(cur.lastrowid)


def list_events(con, limit: int = 50) -> list[dict]:
    rows = con.execute(
        "SELECT id, ts, event_type, reason, snapshot_path FROM events ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def stats(con, now: datetime) -> dict:
    total = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    day = now.strftime("%Y-%m-%d")
    today = con.execute(
        "SELECT COUNT(*) FROM events WHERE substr(ts, 1, 10) = ?", (day,)
    ).fetchone()[0]
    last = con.execute("SELECT ts FROM events ORDER BY id DESC LIMIT 1").fetchone()
    return {"total": int(total), "today": int(today), "last_ts": last[0] if last else None}
```

- [ ] **Step 4: Implement `taubenschreck/backend/storage.py`**

```python
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
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_db.py -v`
Expected: `3 passed`.

- [ ] **Step 6: Commit**

```bash
git add taubenschreck/backend/db.py taubenschreck/backend/storage.py tests/test_db.py
git commit -m "feat(backend): add SQLite event store and snapshot storage"
```

---

### Task 10: ntfy notifier

**Files:**
- Create: `taubenschreck/notifier/ntfy.py`
- Test: `tests/test_notifier.py`

**Interfaces:**
- Consumes: `Event` (Task 2).
- Produces:
  - `Transport = Callable[[str, bytes, dict[str, str]], None]` (url, body, headers).
  - `NtfyNotifier(url: str, transport: Transport = <httpx POST>)` with `notify_fire(event: Event, snapshot_path: str|None=None) -> None`.

- [ ] **Step 1: Write the failing test** in `tests/test_notifier.py`

```python
from datetime import datetime

from taubenschreck.core.types import Event, EventType
from taubenschreck.notifier.ntfy import NtfyNotifier


def test_notify_fire_posts_message():
    calls = []

    def fake_transport(url, body, headers):
        calls.append((url, body, headers))

    n = NtfyNotifier("http://ntfy.test/taubenschreck", transport=fake_transport)
    n.notify_fire(Event(datetime(2026, 6, 22, 12, 0, 0), EventType.FIRE, "fire"))

    assert len(calls) == 1
    url, body, headers = calls[0]
    assert url == "http://ntfy.test/taubenschreck"
    assert b"12:00:00" in body
    assert headers["Title"] == "Taubenschreck"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_notifier.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `taubenschreck/notifier/ntfy.py`**

```python
from __future__ import annotations

from typing import Callable

import httpx

from taubenschreck.core.types import Event

Transport = Callable[[str, bytes, dict], None]


def _http_post(url: str, body: bytes, headers: dict) -> None:
    httpx.post(url, content=body, headers=headers, timeout=5.0)


class NtfyNotifier:
    def __init__(self, url: str, transport: Transport = _http_post):
        self._url = url
        self._transport = transport

    def notify_fire(self, event: Event, snapshot_path: str | None = None) -> None:
        msg = f"Pigeon repelled at {event.timestamp:%H:%M:%S} ({event.reason})"
        headers = {"Title": "Taubenschreck", "Tags": "bird"}
        self._transport(self._url, msg.encode("utf-8"), headers)
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_notifier.py -v`
Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add taubenschreck/notifier/ntfy.py tests/test_notifier.py
git commit -m "feat(notifier): add ntfy push notifier with injectable transport"
```

---

### Task 11: Recorder (event sink: db + snapshot + notify)

**Files:**
- Create: `taubenschreck/backend/recorder.py`
- Test: `tests/test_storage_recorder.py`

**Interfaces:**
- Consumes: `db.insert_event`, `save_snapshot` (Task 9), `NtfyNotifier` (Task 10), `Event`, `Detection`.
- Produces:
  - `Recorder(con, snapshot_dir: str, notifier=None)` with `record(event: Event, frame: np.ndarray|None, detections: list[Detection]) -> int`. Saves a snapshot when `frame is not None`, inserts the event row (returns its id), and calls `notifier.notify_fire` when a notifier is set. Thread-safe DB writes via an internal lock.

- [ ] **Step 1: Write the failing test** in `tests/test_storage_recorder.py`

```python
from datetime import datetime

import numpy as np

from taubenschreck.backend import db
from taubenschreck.backend.recorder import Recorder
from taubenschreck.core.types import Event, EventType


class _SpyNotifier:
    def __init__(self):
        self.calls = []

    def notify_fire(self, event, snapshot_path=None):
        self.calls.append((event, snapshot_path))


def _recorder(tmp_path, notifier=None):
    con = db.connect(str(tmp_path / "t.db"))
    db.init_db(con)
    return con, Recorder(con, str(tmp_path / "snaps"), notifier=notifier)


def test_record_saves_snapshot_and_row_and_notifies(tmp_path):
    notifier = _SpyNotifier()
    con, rec = _recorder(tmp_path, notifier)
    frame = np.full((48, 64, 3), 100, dtype=np.uint8)
    event = Event(datetime(2026, 6, 22, 12, 0, 0), EventType.FIRE, "fire")
    rec.record(event, frame, [])
    rows = db.list_events(con)
    assert len(rows) == 1
    assert rows[0]["snapshot_path"].endswith(".jpg")
    assert len(notifier.calls) == 1


def test_record_without_frame_stores_null_snapshot(tmp_path):
    con, rec = _recorder(tmp_path)
    event = Event(datetime(2026, 6, 22, 12, 0, 0), EventType.FIRE, "manual_test")
    rec.record(event, None, [])
    rows = db.list_events(con)
    assert rows[0]["snapshot_path"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_storage_recorder.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `taubenschreck/backend/recorder.py`**

```python
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from taubenschreck.backend import db
from taubenschreck.backend.storage import save_snapshot
from taubenschreck.core.types import Detection, Event


@dataclass
class Recorder:
    con: object                 # sqlite3.Connection
    snapshot_dir: str
    notifier: object | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, event: Event, frame: np.ndarray | None, detections: list[Detection]) -> int:
        snapshot_path = None
        if frame is not None:
            snapshot_path = save_snapshot(self.snapshot_dir, event.timestamp, frame)
        with self._lock:
            row_id = db.insert_event(
                self.con,
                event.timestamp.isoformat(),
                event.event_type.value,
                event.reason,
                snapshot_path,
            )
        if self.notifier is not None:
            self.notifier.notify_fire(event, snapshot_path)
        return row_id
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_storage_recorder.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add taubenschreck/backend/recorder.py tests/test_storage_recorder.py
git commit -m "feat(backend): add Recorder sink (snapshot + db + notify)"
```

---

### Task 12: Controller (threaded loop + arm/disarm/test-fire)

**Files:**
- Create: `taubenschreck/backend/controller.py`
- Test: `tests/test_controller.py`

**Interfaces:**
- Consumes: `Pipeline` (Task 8), `Recorder` (Task 11), `Sprayer` (Task 5), `SafetyConfig`, `SafetyState`, `Event`, `EventType`.
- Produces:
  - `Controller(pipeline: Pipeline, recorder: Recorder, sprayer: Sprayer, safety_config: SafetyConfig)`.
  - `arm()`, `disarm()`, `is_armed() -> bool` — flip `pipeline.state.armed` under `pipeline.lock`.
  - `test_fire() -> None` — fires the sprayer once and records an `Event(..., reason="manual_test")` with no snapshot (bypasses the safety gate by design; the operator pressed the button).
  - `start()` / `stop()` — run `pipeline.run()` in a background daemon thread; `stop()` signals `should_stop` and joins.

- [ ] **Step 1: Write the failing test** in `tests/test_controller.py`

```python
from datetime import datetime

import numpy as np

from taubenschreck.backend import db
from taubenschreck.backend.controller import Controller
from taubenschreck.backend.recorder import Recorder
from taubenschreck.core.config import AppConfig, SafetyConfig
from taubenschreck.core.state import SafetyState
from taubenschreck.detector.model import FakeDetector
from taubenschreck.detector.pipeline import Pipeline
from taubenschreck.detector.sprayer.mock import MockPump


def _build(tmp_path):
    con = db.connect(str(tmp_path / "t.db"))
    db.init_db(con)
    recorder = Recorder(con, str(tmp_path / "snaps"))
    sprayer = MockPump()
    cfg = AppConfig(safety=SafetyConfig(persistence_frames=1, burst_seconds=1.5))
    pipe = Pipeline(
        source=None,
        detector=FakeDetector([[]]),
        sprayer=sprayer,
        config=cfg,
        state=SafetyState(),
        clock=lambda: datetime(2026, 6, 22, 12, 0, 0),
    )
    ctrl = Controller(pipe, recorder, sprayer, cfg.safety)
    return con, ctrl, sprayer


def test_arm_disarm_toggles_state(tmp_path):
    _, ctrl, _ = _build(tmp_path)
    assert ctrl.is_armed() is False    # boot disarmed
    ctrl.arm()
    assert ctrl.is_armed() is True
    ctrl.disarm()
    assert ctrl.is_armed() is False


def test_test_fire_sprays_and_records_manual_event(tmp_path):
    con, ctrl, sprayer = _build(tmp_path)
    ctrl.test_fire()
    assert sprayer.fires == [1.5]
    rows = db.list_events(con)
    assert len(rows) == 1
    assert rows[0]["reason"] == "manual_test"
    assert rows[0]["snapshot_path"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_controller.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement `taubenschreck/backend/controller.py`**

```python
from __future__ import annotations

import threading
from dataclasses import replace
from datetime import datetime

from taubenschreck.backend.recorder import Recorder
from taubenschreck.core.config import SafetyConfig
from taubenschreck.core.types import Event, EventType
from taubenschreck.detector.pipeline import Pipeline
from taubenschreck.detector.sprayer.base import Sprayer


class Controller:
    def __init__(self, pipeline: Pipeline, recorder: Recorder, sprayer: Sprayer, safety_config: SafetyConfig):
        self._pipeline = pipeline
        self._recorder = recorder
        self._sprayer = sprayer
        self._cfg = safety_config
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # route pipeline fire events into the recorder
        self._pipeline.on_event = lambda e, f, d: self._recorder.record(e, f, d)
        self._pipeline.should_stop = self._stop.is_set

    def is_armed(self) -> bool:
        return self._pipeline.state.armed

    def arm(self) -> None:
        with self._pipeline.lock:
            self._pipeline.state = replace(self._pipeline.state, armed=True)

    def disarm(self) -> None:
        with self._pipeline.lock:
            self._pipeline.state = replace(self._pipeline.state, armed=False)

    def test_fire(self) -> None:
        self._sprayer.fire(self._cfg.burst_seconds)
        event = Event(datetime.now(), EventType.FIRE, "manual_test")
        self._recorder.record(event, None, [])

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._pipeline.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_controller.py -v`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add taubenschreck/backend/controller.py tests/test_controller.py
git commit -m "feat(backend): add Controller (threaded loop, arm/disarm/test-fire)"
```

---

### Task 13: FastAPI app (read API + control + static)

**Files:**
- Create: `taubenschreck/backend/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `Controller` (Task 12), `db.list_events`/`db.stats` (Task 9).
- Produces: `create_app(controller: Controller, con, static_dir: str, snapshot_dir: str) -> FastAPI` with routes:
  - `GET /api/state` → `{"armed": bool}`
  - `POST /api/arm` → `{"armed": true}`; `POST /api/disarm` → `{"armed": false}`
  - `POST /api/test-fire` → `{"ok": true}`
  - `GET /api/events?limit=50` → `{"events": [...]}`
  - `GET /api/stats` → `{"total", "today", "last_ts"}`
  - `GET /snapshots/{name}` → image file (mounted static)
  - `GET /` → dashboard `index.html`

- [ ] **Step 1: Write the failing test** in `tests/test_app.py`

```python
from datetime import datetime

from fastapi.testclient import TestClient

from taubenschreck.backend import db
from taubenschreck.backend.app import create_app
from taubenschreck.backend.controller import Controller
from taubenschreck.backend.recorder import Recorder
from taubenschreck.core.config import AppConfig, SafetyConfig
from taubenschreck.core.state import SafetyState
from taubenschreck.detector.model import FakeDetector
from taubenschreck.detector.pipeline import Pipeline
from taubenschreck.detector.sprayer.mock import MockPump


def _client(tmp_path):
    con = db.connect(str(tmp_path / "t.db"))
    db.init_db(con)
    recorder = Recorder(con, str(tmp_path / "snaps"))
    sprayer = MockPump()
    cfg = AppConfig(safety=SafetyConfig(persistence_frames=1))
    pipe = Pipeline(source=None, detector=FakeDetector([[]]), sprayer=sprayer,
                    config=cfg, state=SafetyState(), clock=lambda: datetime(2026, 6, 22, 12, 0, 0))
    ctrl = Controller(pipe, recorder, sprayer, cfg.safety)
    app = create_app(ctrl, con, "taubenschreck/dashboard/static", str(tmp_path / "snaps"))
    return TestClient(app), con


def test_arm_disarm_endpoints(tmp_path):
    client, _ = _client(tmp_path)
    assert client.get("/api/state").json() == {"armed": False}
    assert client.post("/api/arm").json() == {"armed": True}
    assert client.get("/api/state").json() == {"armed": True}
    assert client.post("/api/disarm").json() == {"armed": False}


def test_test_fire_endpoint_records(tmp_path):
    client, con = _client(tmp_path)
    assert client.post("/api/test-fire").json() == {"ok": True}
    assert db.list_events(con)[0]["reason"] == "manual_test"


def test_events_and_stats_endpoints(tmp_path):
    client, con = _client(tmp_path)
    db.insert_event(con, "2026-06-22T08:00:00", "fire", "fire", None)
    events = client.get("/api/events").json()["events"]
    assert len(events) == 1
    stats = client.get("/api/stats").json()
    assert stats["total"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_app.py -v`
Expected: FAIL (`ModuleNotFoundError`). (The static dir need not exist yet for the API tests; it is mounted in Task 14.)

- [ ] **Step 3: Implement `taubenschreck/backend/app.py`**

```python
from __future__ import annotations

import os
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from taubenschreck.backend import db
from taubenschreck.backend.controller import Controller


def create_app(controller: Controller, con, static_dir: str, snapshot_dir: str) -> FastAPI:
    app = FastAPI(title="Taubenschreck")

    @app.get("/api/state")
    def get_state():
        return {"armed": controller.is_armed()}

    @app.post("/api/arm")
    def arm():
        controller.arm()
        return {"armed": controller.is_armed()}

    @app.post("/api/disarm")
    def disarm():
        controller.disarm()
        return {"armed": controller.is_armed()}

    @app.post("/api/test-fire")
    def test_fire():
        controller.test_fire()
        return {"ok": True}

    @app.get("/api/events")
    def events(limit: int = 50):
        return {"events": db.list_events(con, limit)}

    @app.get("/api/stats")
    def stats():
        return db.stats(con, datetime.now())

    os.makedirs(snapshot_dir, exist_ok=True)
    app.mount("/snapshots", StaticFiles(directory=snapshot_dir), name="snapshots")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_app.py -v`
Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add taubenschreck/backend/app.py tests/test_app.py
git commit -m "feat(backend): add FastAPI app (state/arm/disarm/test-fire/events/stats)"
```

---

### Task 14: Dashboard UI

**Files:**
- Create: `taubenschreck/dashboard/static/index.html`, `taubenschreck/dashboard/static/style.css`, `taubenschreck/dashboard/static/app.js`
- Test: extend `tests/test_app.py` with a root-route smoke test.

**Interfaces:**
- Consumes: the JSON API from Task 13. No new Python interfaces.

- [ ] **Step 1: Write the failing smoke test** — append to `tests/test_app.py`

```python
def test_root_serves_dashboard(tmp_path):
    client, _ = _client(tmp_path)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Taubenschreck" in resp.text
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_app.py::test_root_serves_dashboard -v`
Expected: FAIL (404 / file not found — `index.html` does not exist yet).

- [ ] **Step 3: Implement `taubenschreck/dashboard/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Taubenschreck</title>
  <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <header>
    <h1>🐦 Taubenschreck</h1>
    <div id="armed-badge" class="badge">…</div>
  </header>

  <section class="controls">
    <button id="arm-btn">Arm</button>
    <button id="disarm-btn">Disarm</button>
    <button id="test-fire-btn" class="danger">Test fire</button>
  </section>

  <section class="stats">
    <div>Today: <span id="stat-today">–</span></div>
    <div>Total: <span id="stat-total">–</span></div>
    <div>Last: <span id="stat-last">–</span></div>
  </section>

  <h2>Recent events</h2>
  <ul id="events"></ul>

  <script src="/static/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Implement `taubenschreck/dashboard/static/style.css`**

```css
:root { font-family: system-ui, sans-serif; }
body { margin: 0; padding: 1.5rem; max-width: 720px; margin-inline: auto; color: #1a1a1a; }
header { display: flex; align-items: center; justify-content: space-between; }
.badge { padding: .3rem .7rem; border-radius: 999px; font-weight: 600; background: #ddd; }
.badge.armed { background: #d33; color: #fff; }
.badge.disarmed { background: #2a2; color: #fff; }
.controls button { font-size: 1rem; padding: .5rem 1rem; margin-right: .5rem; cursor: pointer; }
button.danger { background: #d33; color: #fff; border: none; border-radius: 6px; }
.stats { display: flex; gap: 1.5rem; margin: 1rem 0; }
#events { list-style: none; padding: 0; }
#events li { display: flex; gap: .8rem; align-items: center; padding: .4rem 0; border-bottom: 1px solid #eee; }
#events img { width: 96px; height: 72px; object-fit: cover; border-radius: 4px; background: #eee; }
```

- [ ] **Step 5: Implement `taubenschreck/dashboard/static/app.js`**

```javascript
async function getJSON(url) { return (await fetch(url)).json(); }
async function post(url) { return (await fetch(url, { method: "POST" })).json(); }

function renderArmed(armed) {
  const b = document.getElementById("armed-badge");
  b.textContent = armed ? "ARMED" : "DISARMED";
  b.className = "badge " + (armed ? "armed" : "disarmed");
}

async function refreshState() { renderArmed((await getJSON("/api/state")).armed); }

async function refreshStats() {
  const s = await getJSON("/api/stats");
  document.getElementById("stat-today").textContent = s.today;
  document.getElementById("stat-total").textContent = s.total;
  document.getElementById("stat-last").textContent = s.last_ts || "–";
}

async function refreshEvents() {
  const { events } = await getJSON("/api/events?limit=25");
  const ul = document.getElementById("events");
  ul.innerHTML = "";
  for (const e of events) {
    const li = document.createElement("li");
    const img = e.snapshot_path
      ? `<img src="/snapshots/${e.snapshot_path.split("/").pop()}" alt="snapshot" />`
      : `<img alt="no snapshot" />`;
    li.innerHTML = `${img}<span>${e.ts} — <b>${e.reason}</b></span>`;
    ul.appendChild(li);
  }
}

async function refreshAll() { await Promise.all([refreshState(), refreshStats(), refreshEvents()]); }

document.getElementById("arm-btn").onclick = async () => { await post("/api/arm"); refreshState(); };
document.getElementById("disarm-btn").onclick = async () => { await post("/api/disarm"); refreshState(); };
document.getElementById("test-fire-btn").onclick = async () => { await post("/api/test-fire"); refreshAll(); };

refreshAll();
setInterval(refreshAll, 3000);
```

- [ ] **Step 6: Run to verify pass**

Run: `.venv/Scripts/python -m pytest tests/test_app.py -v`
Expected: `4 passed`.

- [ ] **Step 7: Commit**

```bash
git add taubenschreck/dashboard/static/ tests/test_app.py
git commit -m "feat(dashboard): add control + stats + events UI"
```

---

### Task 15: Entrypoint, quickstart docs, end-to-end run

**Files:**
- Create: `taubenschreck/main.py`, `README.md` (replace the stub), `docs/quickstart.md`
- Modify: none
- Test: full-suite run + manual end-to-end.

**Interfaces:**
- Consumes: everything above.
- Produces: `build_app(config_path: str) -> tuple[FastAPI, Controller]` and a `main()` that loads config, builds the source/detector/sprayer, starts the controller, and runs uvicorn.

- [ ] **Step 1: Implement `taubenschreck/main.py`**

```python
from __future__ import annotations

import sys

import uvicorn

from taubenschreck.backend import db
from taubenschreck.backend.app import create_app
from taubenschreck.backend.controller import Controller
from taubenschreck.backend.recorder import Recorder
from taubenschreck.core.config import AppConfig, load_config
from taubenschreck.core.state import SafetyState
from taubenschreck.detector.model import YoloDetector
from taubenschreck.detector.pipeline import Pipeline
from taubenschreck.detector.sources.videofile import VideoFileSource
from taubenschreck.detector.sources.webcam import WebcamSource
from taubenschreck.detector.sprayer.mock import MockPump
from taubenschreck.notifier.ntfy import NtfyNotifier

STATIC_DIR = "taubenschreck/dashboard/static"


def _build_source(cfg: AppConfig, should_stop):
    if cfg.frame_source == "videofile":
        if not cfg.video_path:
            raise ValueError("frame_source=videofile requires video_path")
        return VideoFileSource(cfg.video_path)
    return WebcamSource(cfg.webcam_index, should_stop=should_stop)


def build_app(config_path: str):
    cfg = load_config(config_path)
    con = db.connect(cfg.db_path)
    db.init_db(con)
    notifier = NtfyNotifier(cfg.ntfy_url) if cfg.ntfy_url else None
    recorder = Recorder(con, cfg.snapshot_dir, notifier=notifier)
    sprayer = MockPump()  # Phase 1: always mock
    detector = YoloDetector(cfg.model_weights, min_confidence=0.25)
    pipeline = Pipeline(source=None, detector=detector, sprayer=sprayer,
                        config=cfg, state=SafetyState())
    controller = Controller(pipeline, recorder, sprayer, cfg.safety)
    # Build the source AFTER the controller wires pipeline.should_stop, since the
    # source captures that callable at construction.
    pipeline.source = _build_source(cfg, pipeline.should_stop)
    app = create_app(controller, con, STATIC_DIR, cfg.snapshot_dir)
    return app, controller, cfg


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.toml"
    app, controller, cfg = build_app(config_path)
    controller.start()
    try:
        uvicorn.run(app, host=cfg.host, port=cfg.port)
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `README.md`** (replace the stub)

````markdown
# 🐦 Taubenschreck

A vision-triggered water sentry that humanely keeps pigeons off a balcony.
A camera + a small neural net detect pigeons and fire a brief burst of water
at a fixed spray zone, so the birds learn to avoid the spot. **Deterrence, not harm.**

Built in public. Phase 1 runs entirely on a PC — webcam + a *mock* pump — with
no Raspberry Pi hardware. Real camera + pump come in Phase 2.

## Quickstart (Phase 1, no hardware)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows; use .venv/bin on Linux/macOS
cp config/config.example.toml config/config.toml  # then edit
.venv/Scripts/python -m taubenschreck.main config/config.toml
```

Open http://127.0.0.1:8000 — arm the system, point your webcam at a pigeon
photo on your phone, and watch the event log + (optional) ntfy push fire.

See [docs/quickstart.md](docs/quickstart.md) for details and
[the design spec](docs/superpowers/specs/2026-06-22-taubenschreck-design.md).

## Tests

```bash
.venv/Scripts/python -m pytest -v
```

## License

MIT
````

- [ ] **Step 3: Write `docs/quickstart.md`**

```markdown
# Quickstart — Phase 1 (hardware-free)

## Configure
Copy `config/config.example.toml` to `config/config.toml` and set:
- `frame_source` — `"webcam"` (live) or `"videofile"` + `video_path` (a saved clip).
- `ntfy_url` — your self-hosted ntfy topic URL, or leave unset to disable push.
- `[safety]` — tune `active_start`/`active_end`, `persistence_frames`, `cooldown_seconds`.

## Run
```bash
.venv/Scripts/python -m taubenschreck.main config/config.toml
```
The dashboard is at `http://<host>:<port>` (default `127.0.0.1:8000`).

## What "firing" means in Phase 1
The sprayer is a `MockPump`: it logs `MOCK FIRE for 1.50s` and records an event
+ snapshot. Nothing physical happens yet. This is the "it *would* have fired"
system. In Phase 2 we swap `MockPump` → `GpioPump` and `WebcamSource` →
`PiCamera` via config — no logic changes.

## Safety reminder (for Phase 2)
Boot state is always DISARMED. The dashboard Arm/Disarm + the hardware kill
switch are independent. Aim nozzles downward into your own balcony only.
```

- [ ] **Step 4: Run the full test suite**

Run: `.venv/Scripts/python -m pytest -v`
Expected: all tests pass (`test_smoke, test_core_types, test_config, test_safety, test_sprayer, test_sources, test_model, test_pipeline, test_db, test_notifier, test_storage_recorder, test_controller, test_app`).

- [ ] **Step 5: Manual end-to-end run**

```bash
cp config/config.example.toml config/config.toml
.venv/Scripts/python -m taubenschreck.main config/config.toml
```
Then in a browser at `http://127.0.0.1:8000`:
1. Badge shows **DISARMED**. Click **Arm** → badge turns red **ARMED**.
2. Click **Test fire** → an event appears with reason `manual_test`; terminal logs `MOCK FIRE`.
3. Hold a pigeon photo in front of the webcam for ~1s → a `fire` event with a snapshot thumbnail appears; if `ntfy_url` is set, a push arrives.
4. Step into frame yourself → no firing (person suppression).

- [ ] **Step 6: Commit**

```bash
git add taubenschreck/main.py README.md docs/quickstart.md
git commit -m "feat: add entrypoint, quickstart docs, and end-to-end wiring"
```

---

## Self-Review

**1. Spec coverage**

| Spec item | Task(s) |
|---|---|
| Edge detector service (capture→detect→safety→spray) | 6, 7, 8 |
| `FrameSource` (Webcam/VideoFile) abstraction | 6 |
| `Sprayer` (MockPump) abstraction | 5 |
| Pure safety gate + exhaustive tests | 4 |
| Person suppression / armed / window / cooldown / rate-limit / persistence | 4 |
| Boot state DISARMED | 2, 4, 12 |
| Event backend + SQLite + snapshot storage | 9, 11 |
| Dashboard: stats, recent events+thumbnails, armed state, arm/disarm, test-fire | 13, 14 |
| ntfy notifier (self-hosted, config-only URL) | 10, 11 |
| Config (active hours, thresholds, cooldown, caps, source, sprayer, notifier) | 3 |
| Runs on PC with webcam + mock pump, no hardware | 15 |
| Pulse limiter (Layer 2) / kill switch (Layer 3) | Phase 2 (hardware) — out of scope here; noted in quickstart |

Gaps: none for Phase 1. Layers 2–3 of the safety model are hardware-bound and belong to Phase 2; `burst_seconds` is already the single capped burst length the pulse limiter will enforce.

**2. Placeholder scan:** No TBD/TODO/"add error handling" placeholders; every code step shows complete code.

**3. Type consistency:** `Detection`, `Event`/`EventType`, `SafetyDecision`, `SafetyState`, `SafetyConfig`/`AppConfig`, `evaluate()`, `Pipeline.process/run`, `Recorder.record`, `Controller.arm/disarm/is_armed/test_fire/start/stop`, and `create_app(controller, con, static_dir, snapshot_dir)` are used with identical signatures across all tasks.
```
