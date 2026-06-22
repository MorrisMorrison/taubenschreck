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
