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
