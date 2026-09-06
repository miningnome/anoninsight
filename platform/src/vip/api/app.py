"""FastAPI application factory for the Video Intelligence Platform."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from vip.cameras.factory import create_camera_source
from vip.core.config import Settings
from vip.pipeline import CameraWorker

from .registry import CameraRegistry
from .routes.cameras import router as cameras_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    registry = CameraRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        for camera_config in settings.cameras:
            source = create_camera_source(camera_config)
            worker = CameraWorker(
                source,
                reconnect_delay_seconds=camera_config.reconnect_delay_seconds,
                reconnect_max_delay_seconds=camera_config.reconnect_max_delay_seconds,
            )
            registry.add(camera_config, worker)
            worker.start()
        yield
        registry.stop_all()

    app = FastAPI(title="Video Intelligence Platform", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.cameras = registry
    app.include_router(cameras_router, prefix="/api")
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
