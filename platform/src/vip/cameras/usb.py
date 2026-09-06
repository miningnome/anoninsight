"""USBCamera: a locally attached USB/UVC webcam, addressed by device index."""

from __future__ import annotations

from ._opencv import CaptureFactory, OpenCVCameraSource


class USBCamera(OpenCVCameraSource):
    def __init__(
        self,
        camera_id: str,
        device_index: int = 0,
        *,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self.device_index = device_index
        factory = capture_factory or (lambda: _open_device(device_index))
        super().__init__(camera_id, factory)


def _open_device(device_index: int):
    import cv2

    return cv2.VideoCapture(device_index)
