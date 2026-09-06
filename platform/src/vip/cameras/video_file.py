"""VideoFileCamera: replays a local video file as if it were a live camera.

Mainly for development and tests without real camera hardware - the same
CameraSource interface as RTSPCamera means the rest of the pipeline cannot
tell the difference.
"""

from __future__ import annotations

import time
from typing import Any

from ._opencv import CaptureFactory, OpenCVCameraSource

_DEFAULT_FPS = 25.0


class _LoopingFileCapture:
    """Wraps a cv2.VideoCapture over a file so read() restarts from the
    beginning once the file is exhausted, when `loop` is True.

    Reads are paced to the file's own FPS. A local file has no natural
    pacing the way network I/O paces a real RTSP read(), so without this a
    CameraWorker's reader loop would spin as fast as the disk/codec allow,
    starving every other thread of the GIL - this makes a video file behave
    like the camera it is meant to stand in for.
    """

    def __init__(self, path: str, loop: bool) -> None:
        import cv2

        self._cv2 = cv2
        self._loop = loop
        self._capture = cv2.VideoCapture(path)
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._frame_interval = 1.0 / fps if fps and fps > 0 else 1.0 / _DEFAULT_FPS
        self._next_read_at = time.monotonic()

    def isOpened(self) -> bool:
        return self._capture.isOpened()

    def read(self) -> tuple[bool, Any]:
        now = time.monotonic()
        if self._next_read_at > now:
            time.sleep(self._next_read_at - now)
        self._next_read_at = max(self._next_read_at, now) + self._frame_interval

        ok, image = self._capture.read()
        if not ok and self._loop:
            self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._capture.read()
        return ok, image

    def get(self, prop_id: int) -> float:
        return self._capture.get(prop_id)

    def release(self) -> None:
        self._capture.release()


class VideoFileCamera(OpenCVCameraSource):
    def __init__(
        self,
        camera_id: str,
        path: str,
        *,
        loop: bool = True,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self.path = path
        factory = capture_factory or (lambda: _LoopingFileCapture(path, loop))
        super().__init__(camera_id, factory)
