"""Builds a CameraSource from a CameraConfig without the caller ever
importing RTSPCamera/USBCamera/VideoFileCamera directly - this is the one
place that knows the mapping from `source_type` to an implementation."""

from __future__ import annotations

from vip.core.config import CameraConfig

from .base import CameraSource
from .rtsp import RTSPCamera
from .usb import USBCamera
from .video_file import VideoFileCamera


def create_camera_source(config: CameraConfig) -> CameraSource:
    if config.source_type == "rtsp":
        return RTSPCamera(
            config.id,
            config.url,
            open_timeout_seconds=config.open_timeout_seconds,
            read_timeout_seconds=config.read_timeout_seconds,
        )
    if config.source_type == "usb":
        return USBCamera(config.id, int(config.url))
    if config.source_type == "file":
        return VideoFileCamera(config.id, config.url, loop=config.loop)
    raise ValueError(f"unsupported camera source_type: {config.source_type!r}")
