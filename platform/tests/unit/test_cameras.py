from __future__ import annotations

import numpy as np
import pytest

from vip.cameras.base import CameraSourceError
from vip.cameras.factory import create_camera_source
from vip.cameras.rtsp import RTSPCamera
from vip.cameras.usb import USBCamera
from vip.cameras.video_file import VideoFileCamera
from vip.core.config import CameraConfig


class _FakeCapture:
    """Mimics the subset of cv2.VideoCapture used by OpenCVCameraSource,
    so RTSPCamera/USBCamera/VideoFileCamera are testable without any real
    camera hardware or video file on disk."""

    def __init__(self, frames, opened: bool = True) -> None:
        self._frames = list(frames)
        self._opened = opened
        self._index = 0

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame

    def get(self, prop_id: int) -> float:
        return 0.0

    def release(self) -> None:
        self._opened = False


def _blank_frame() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_rtsp_camera_reads_sequential_frames():
    capture = _FakeCapture([_blank_frame(), _blank_frame()])
    camera = RTSPCamera("cam-1", "rtsp://host/stream", capture_factory=lambda: capture)

    camera.open()
    assert camera.is_open

    first = camera.read()
    second = camera.read()

    assert first.camera_id == "cam-1"
    assert first.sequence == 1
    assert second.sequence == 2


def test_rtsp_camera_open_failure_raises():
    capture = _FakeCapture([], opened=False)
    camera = RTSPCamera("cam-1", "rtsp://host/stream", capture_factory=lambda: capture)

    with pytest.raises(CameraSourceError):
        camera.open()
    assert not camera.is_open


def test_rtsp_camera_read_before_open_raises():
    camera = RTSPCamera("cam-1", "rtsp://host/stream", capture_factory=lambda: _FakeCapture([]))
    with pytest.raises(CameraSourceError):
        camera.read()


def test_rtsp_camera_read_failure_raises_and_source_can_be_reopened():
    capture = _FakeCapture([])  # no frames queued -> immediate read failure
    camera = RTSPCamera("cam-1", "rtsp://host/stream", capture_factory=lambda: capture)
    camera.open()

    with pytest.raises(CameraSourceError):
        camera.read()

    # A reconnect loop closes and reopens the same CameraSource instance -
    # that must keep working with a fresh capture behind the same factory.
    camera.close()
    assert not camera.is_open
    replacement = _FakeCapture([_blank_frame()])
    camera2 = RTSPCamera("cam-1", "rtsp://host/stream", capture_factory=lambda: replacement)
    camera2.open()
    assert camera2.read() is not None


def test_redacted_url_strips_credentials_and_query():
    camera = RTSPCamera(
        "cam-1",
        "rtsp://admin:secret@192.168.1.50:554/stream1?token=abc",
        capture_factory=lambda: _FakeCapture([]),
    )
    redacted = camera.redacted_url()
    assert "secret" not in redacted
    assert "admin" not in redacted
    assert "token" not in redacted
    assert redacted == "rtsp://192.168.1.50:554/stream1"


def test_video_file_camera_uses_injected_capture_factory():
    capture = _FakeCapture([_blank_frame()])
    camera = VideoFileCamera("cam-file", "unused.mp4", capture_factory=lambda: capture)
    camera.open()
    assert camera.read() is not None


def test_usb_camera_uses_injected_capture_factory():
    capture = _FakeCapture([_blank_frame()])
    camera = USBCamera("cam-usb", 0, capture_factory=lambda: capture)
    camera.open()
    assert camera.read() is not None


def test_create_camera_source_dispatches_by_source_type():
    rtsp = create_camera_source(
        CameraConfig(id="c1", name="C1", source_type="rtsp", url="rtsp://host/s")
    )
    usb = create_camera_source(CameraConfig(id="c2", name="C2", source_type="usb", url="0"))
    file_ = create_camera_source(
        CameraConfig(id="c3", name="C3", source_type="file", url="clip.mp4")
    )

    assert isinstance(rtsp, RTSPCamera)
    assert isinstance(usb, USBCamera)
    assert isinstance(file_, VideoFileCamera)
