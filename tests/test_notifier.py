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
