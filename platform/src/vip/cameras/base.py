"""Camera abstraction.

`CameraSource` decouples the rest of the platform from any specific camera
vendor or transport (RTSP, USB, local video files and, eventually, ONVIF).
Nothing above this module should ever import `cv2` or know that RTSP
exists; it only depends on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True)
class Frame:
    """One decoded video frame."""

    camera_id: str
    image: np.ndarray  # BGR, HxWx3, uint8
    timestamp: datetime
    sequence: int


class CameraSourceError(RuntimeError):
    """A CameraSource could not open, or lost, its underlying source.

    Callers (typically a reconnect loop) are expected to catch this, close
    the source, wait according to their own backoff policy, and try open()
    again - the CameraSource itself does not retry internally.
    """


class CameraSource(ABC):
    """A source of decoded video frames.

    Implementations must be safe to call `open()` after `close()`, since a
    reconnect loop will do exactly that after a transient failure.
    """

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id

    @abstractmethod
    def open(self) -> None:
        """Open the underlying source. Raises CameraSourceError on failure."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or None if none was available this call.

        Raises CameraSourceError if the source has become unusable and must
        be closed and reopened.
        """

    @abstractmethod
    def close(self) -> None:
        """Release underlying resources. Safe to call multiple times."""

    @property
    @abstractmethod
    def is_open(self) -> bool: ...
