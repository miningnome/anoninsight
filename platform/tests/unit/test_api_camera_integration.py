"""End-to-end wiring test: a real (tiny, generated) video file as camera
source, the mock face engine, and the full FastAPI app - covers the same
path used manually against a real RTSP camera, without needing one.

The infinite MJPEG stream itself is not consumed here: FastAPI's
TestClient buffers a streamed ASGI response before handing back control,
which deadlocks against a StreamingResponse whose generator never ends (by
design - a live camera stream has no natural end). That path is validated
by manually running the app against a real video/RTSP source (see
platform/README.md) rather than in this suite; see test_hud.py for a unit
test of the frame-annotation logic used by the stream endpoint.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vip.api.app import create_app
from vip.core.config import CameraConfig, Settings, VisionConfig


@pytest.fixture
def sample_video(tmp_path) -> str:
    cv2 = pytest.importorskip("cv2")
    path = str(tmp_path / "sample.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 32))
    for i in range(10):
        writer.write(np.full((32, 32, 3), i * 20 % 255, dtype=np.uint8))
    writer.release()
    return path


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_file_camera_streams_through_the_full_app(sample_video):
    settings = Settings(
        vision=VisionConfig(engine="mock"),
        cameras=[
            CameraConfig(
                id="camera-test",
                name="Local test file",
                source_type="file",
                url=sample_video,
                loop=True,
                display_fps=30.0,
                analysis_fps=30.0,
            )
        ],
    )

    with TestClient(create_app(settings)) as client:
        assert _wait_until(
            lambda: client.get("/api/cameras/camera-test").json()["status"] == "running"
        )

        detail = client.get("/api/cameras/camera-test").json()
        assert detail["faces_detected"] == 0  # mock engine stubs no faces
