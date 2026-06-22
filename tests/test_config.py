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
