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
