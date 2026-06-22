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
