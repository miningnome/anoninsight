"""RTSPCamera: the reference CameraSource for conventional IP cameras."""

from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from ._opencv import CaptureFactory, OpenCVCameraSource


def _default_rtsp_capture_factory(
    url: str, open_timeout_seconds: float, read_timeout_seconds: float
):
    import cv2

    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_seconds * 1000)
    capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_seconds * 1000)
    return capture


class RTSPCamera(OpenCVCameraSource):
    """CameraSource for standard RTSP/RTSPS IP cameras, decoded via FFmpeg
    through OpenCV.

    `capture_factory` can be overridden to inject a fake capture in tests,
    or to plug an alternative decoding backend (e.g. GStreamer) later
    without changing anything above this class.
    """

    def __init__(
        self,
        camera_id: str,
        url: str,
        *,
        open_timeout_seconds: float = 5.0,
        read_timeout_seconds: float = 5.0,
        capture_factory: CaptureFactory | None = None,
    ) -> None:
        self.url = url
        factory = capture_factory or (
            lambda: _default_rtsp_capture_factory(
                url, open_timeout_seconds, read_timeout_seconds
            )
        )
        super().__init__(camera_id, factory)

    def redacted_url(self) -> str:
        """URL safe to log or return over the API.

        Strips credentials and query string so RTSP passwords embedded in
        the URL never leak into logs or API responses.
        """
        parts = urlsplit(self.url)
        netloc = parts.hostname or ""
        if parts.port:
            netloc = f"{netloc}:{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
