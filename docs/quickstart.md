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
