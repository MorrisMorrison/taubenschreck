# Taubenschreck — Design Spec

**Status:** Draft (approved in brainstorming) · **Date:** 2026-06-22

> *Taubenschreck* (German: "pigeon fright") is a vision-triggered water sentry that
> humanely keeps pigeons off a balcony. A camera and a small neural net on a Raspberry
> Pi detect pigeons and fire a short burst of water at a fixed spray zone, so the birds
> learn to avoid the area. The goal is **deterrence, not harm.**

---

## 1. Goals & non-goals

### Goals
- Detect pigeons on the balcony from a fixed camera and fire a brief water burst to shoo them.
- Run the full detect → decide → fire loop **self-contained on the edge device**; firing never depends on the network.
- Be **safe by design**: never spray a person, never flood the balcony, never run at the wrong time.
- Be **developable before hardware exists** — the whole system runs on a PC with a webcam and a mock pump.
- Give the owner visibility: push notification on fire, a saved snapshot per event, and a small stats dashboard.

### Non-goals (YAGNI for v1)
- No pan/tilt aiming — a **fixed spray zone** covering the favorite landing spots is enough. (Possible v2.)
- No harm to birds — water deterrence only.
- No cat/dog/other-animal handling — the camera sees only the balcony (top floor); pigeons are effectively the only subject. (Person detection is still required, for the owner.)
- No off-grid power — mains is available on the balcony.
- No cloud services — everything self-hosted.

---

## 2. Hardware

### Compute & vision
- **Raspberry Pi 5 (4 GB)** running the whole loop. *Alternative:* a Pi 4 + a **Coral USB accelerator** for inference. (A spare Pi 3 is usable only for early software bring-up — too slow for the real-time net.)
- **Camera:** Pi Camera Module 3 (autofocus, decent low light) or a USB webcam, in its **own weatherproof housing** (the viewpoint has a clear line of sight but is exposed).
- Stable 5 V / 5 A USB-C PSU.

### Water subsystem
- **12 V diaphragm pump** (booster type, ~40–60 psi, with pressure switch).
- **Sealed refillable reservoir**, food-grade tubing, **1–2 fixed nozzles** aimed at the landing spots.
- Separate **12 V PSU** for the pump.
- *Fail-safe property:* a pump only pushes water while actively powered. There is no failure mode that floods the balcony from a stuck-open valve (unlike a mains solenoid).

### Control & safety electronics
- **Relay module or logic-level MOSFET** to switch the pump from a Pi GPIO pin (with flyback diode).
- **Physical kill switch** — inline toggle that physically cuts pump power; plus a GPIO sense line so software can report the armed state.
- Status LED (armed / disarmed) — optional.

### Enclosure & siting
- IP-rated enclosure for the Pi + electronics; cable glands; weatherproof camera housing.
- Pump + reservoir on the balcony floor; mains outlet on the balcony powers both PSUs.
- **Siting rule:** nozzles aim downward into the owner's own balcony, never over the railing toward the street or neighbors.

Rough cost: ~€120–180 (Pi 5 vs Pi 4 + Coral) plus ~€25–40 for pump and plumbing.

---

## 3. Architecture

Four decoupled units with well-defined interfaces, so each can be understood, built, and tested independently — and so the whole system runs on a PC before any hardware exists.

```
Camera → frames → YOLO → safety gate → [FIRE] → pump + snapshot + event → backend → {dashboard, push}
```

### Unit 1 — Detector service (edge, Python)
The hot loop. Hardware-agnostic via two interfaces:
- `FrameSource` → `PiCamera`, `Webcam`, `VideoFile`
- `Sprayer` → `GpioPump` (real relay), `MockPump` (logs only)

Pipeline: grab frame → YOLO inference → **safety gate** → if `FIRE`: pulse sprayer, save snapshot, emit event.

### Unit 2 — Safety gate (pure logic, no I/O)
A pure function `decide(detections, now, state) → FIRE | SUPPRESS(reason)`. The most-tested unit in the project. See §4.

### Unit 3 — Event backend + storage
Receives events over HTTP, stores them in SQLite and snapshots on disk, serves the dashboard API. May run on the edge device or on a separate always-on server — it is **not** in the safety-critical path.

### Unit 4 — Dashboard + notifier
- Web dashboard: pigeon counts over time, recent events with snapshot thumbnails, current armed state, and **manual arm/disarm + test-fire** controls.
- Notifier: push on fire (**ntfy**, self-hosted) + optional daily "N pigeons repelled" summary.

**Reliability split:** Units 1+2 are self-contained and keep working if the network or the dashboard server is down. Units 3+4 are "nice to have up" but never block firing.

---

## 4. Safety architecture (defense in depth)

Three independent layers plus operational guards. No single failure can soak a person or flood the balcony.

### Layer 1 — Software safety gate (pure logic)
Returns `FIRE` only if **all** hold:
- **No `person` in frame** — any person detection (even low-ish confidence) → hard suppress. Humans always win over pigeons.
- **Armed** — software arm state is on.
- **Inside active window** — current time within configured hours.
- **Cooldown elapsed** and **under max-bursts/hour**.
- **Persistence threshold** — a pigeon seen across **N consecutive frames** before firing, killing single-frame false positives.

Pure inputs → decision, so every rule is exhaustively unit-tested.

### Layer 2 — Pulse limiter (firmware-level watchdog)
The GPIO-toggling code physically cannot energize the pump longer than a hard cap (e.g. 2 s) per activation, enforced by a timed pulse / watchdog — even if the main loop hangs mid-spray. A stuck "on" self-terminates.

### Layer 3 — Hardware kill switch (fail-safe)
An inline switch that physically cuts pump power. No software, GPIO, or firmware can override it. A GPIO sense line only *reports* the state.

### Operational guards
- **Boot state = DISARMED.** Power loss / reboot never returns firing on its own.
- **Aim is fixed and downward** into the owner's balcony only (siting rule, documented).

---

## 5. Software design

**Stack:** Python 3.11+ · `ultralytics` (YOLOv8n) + OpenCV · `picamera2` / `gpiozero` on the Pi · `FastAPI` + SQLite for the backend/dashboard · `ntfy` for push (self-hosted instance).

**Detection model:** start with the pretrained YOLOv8n COCO model, using the **`bird`** class as the pigeon proxy and **`person`** for suppression. Collect snapshots in the field, then optionally fine-tune a pigeon-specific model (Phase 3) for fewer false positives and earlier detection.

### Repo layout
```
taubenschreck/
├─ detector/        # edge service: capture → detect → safety gate → spray
│  ├─ sources/      # FrameSource: PiCamera, Webcam, VideoFile
│  ├─ sprayer/      # Sprayer: GpioPump, MockPump
│  ├─ safety.py     # pure safety-gate logic
│  └─ pipeline.py   # wires it together, emits events
├─ backend/         # FastAPI: event ingest, SQLite, snapshot storage, API
├─ dashboard/       # web UI: stats, recent events + thumbnails, arm/disarm, test-fire
├─ notifier/        # push on fire + daily summary
├─ config/          # active hours, thresholds, cooldown, caps (one file)
├─ tests/           # safety-gate unit tests + pipeline integration (mock pump)
└─ docs/            # spec, wiring diagram, BOM, setup guide
```

**Key design move:** `FrameSource` and `Sprayer` are interfaces. The whole system runs with no hardware via `Webcam` / `VideoFile` + `MockPump` (which logs `FIRE` and saves the snapshot). Swapping to `PiCamera` + `GpioPump` is a configuration change — no logic changes.

### Configuration (single file)
Active-hours window, detection confidence threshold, persistence frame count `N`, burst duration, cooldown seconds, max bursts/hour, frame source, sprayer backend, notifier target.

---

## 6. Testing strategy

- **Safety gate:** exhaustive unit tests — the one piece that must never misfire. Cover person-present, disarmed, outside window, in-cooldown, over-cap, and below-persistence cases.
- **Pipeline:** integration test with `MockPump` over recorded video, asserting the fire/suppress decision sequence.
- **Hardware drivers (`PiCamera`, `GpioPump`):** manual bring-up checklist in `docs/`, including a kill-switch and pulse-limiter verification step.

---

## 7. Development phases

- **Phase 1 — now, hardware-free.** Build the detector pipeline, safety gate (+ full unit tests), backend, dashboard, and notifier. Test against a webcam and pigeon videos with the mock pump. End state: a working "it would have fired" system with snapshots, push, and dashboard.
- **Phase 2 — hardware arrives.** Implement `PiCamera` + `GpioPump`; wire the relay, kill switch, and pulse limiter; weatherproof and mount; field-tune thresholds and active hours; collect real snapshots.
- **Phase 3 — optional.** Fine-tune a pigeon-specific YOLO model from collected snapshots.

---

## 8. Open questions / future (v2+)

- Pan/tilt aiming that targets the detected bird.
- Pigeon-specific fine-tuned model (Phase 3 may graduate into this).
- Escalation logic (louder/longer response for repeat offenders).
- Multi-camera coverage if one view isn't enough.
