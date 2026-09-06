"""Shared open/read/close logic for CameraSource implementations backed by
an OpenCV-compatible capture object (RTSP, USB, local video files).

Subclasses only provide a `capture_factory` that builds a fresh capture for
their specific transport. Injecting that factory - rather than hardcoding
`cv2.VideoCapture(...)` in every subclass - is what makes RTSPCamera,
USBCamera and VideoFileCamera testable without any real camera hardware:
tests pass in a fake capture object that mimics this same structural
interface.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Protocol

from .base import CameraSource, CameraSourceError, Frame


class VideoCapture(Protocol):
    """Structural type matching the subset of cv2.VideoCapture used here."""

    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, Any]: ...
    def release(self) -> None: ...
    def get(self, prop_id: int) -> float: ...


CaptureFactory = Callable[[], "VideoCapture | None"]


class OpenCVCameraSource(CameraSource):
    def __init__(self, camera_id: str, capture_factory: CaptureFactory) -> None:
        super().__init__(camera_id)
        self._capture_factory = capture_factory
        self._capture: VideoCapture | None = None
        self._sequence = 0

    def open(self) -> None:
        self.close()
        capture = self._capture_factory()
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            raise CameraSourceError(f"camera {self.camera_id!r}: unable to open source")
        self._capture = capture

    def read(self) -> Frame | None:
        if self._capture is None:
            raise CameraSourceError(f"camera {self.camera_id!r}: read() called before open()")
        ok, image = self._capture.read()
        if not ok or image is None:
            raise CameraSourceError(f"camera {self.camera_id!r}: failed to read frame")
        self._sequence += 1
        return Frame(
            camera_id=self.camera_id,
            image=image,
            timestamp=datetime.now(timezone.utc),
            sequence=self._sequence,
        )

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()
