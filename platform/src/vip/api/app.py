"""FastAPI application factory for the Video Intelligence Platform."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from vip.cameras.factory import create_camera_source
from vip.core.config import Settings
from vip.identity import EnrollmentService
from vip.pipeline import CameraWorker, FrameProcessor
from vip.storage import Database, Repository
from vip.vision import SerializedVisionEngine
from vip.vision.face import create_face_engine

from .registry import CameraRegistry
from .routes.cameras import router as cameras_router
from .routes.persons import router as persons_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    registry = CameraRegistry()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # One face engine, shared and serialized across every camera and the
        # enrollment path, so VRAM/RAM usage does not grow with the number
        # of cameras.
        face_engine = SerializedVisionEngine(create_face_engine(settings.vision))

        database = Database(settings.storage.database_path)
        database.connect()
        repository = Repository(database)
        app.state.database = database
        app.state.repository = repository
        app.state.enrollment = EnrollmentService(
            repository,
            face_engine,
            min_detection_score=settings.vision.min_score,
            max_upload_bytes=settings.storage.max_upload_bytes,
            store_crops=settings.storage.store_face_crops,
        )

        for camera_config in settings.cameras:
            source = create_camera_source(camera_config)
            worker = CameraWorker(
                source,
                reconnect_delay_seconds=camera_config.reconnect_delay_seconds,
                reconnect_max_delay_seconds=camera_config.reconnect_max_delay_seconds,
            )
            processor = FrameProcessor(
                worker, [face_engine], analysis_fps=camera_config.analysis_fps
            )
            registry.add(camera_config, worker, processor)
            worker.start()
            processor.start()
        yield
        registry.stop_all()
        database.close()

    app = FastAPI(title="Video Intelligence Platform", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.cameras = registry
    app.include_router(cameras_router, prefix="/api")
    app.include_router(persons_router, prefix="/api")
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
