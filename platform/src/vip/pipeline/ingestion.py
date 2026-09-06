"""CameraWorker owns the lifecycle of one CameraSource: continuous reading
in a background thread, reconnection with exponential backoff, and
publishing the latest frame for whoever wants to look at it (an MJPEG
endpoint today; the frame processing pipeline from Fase 2 onward).

Reading is decoupled from consumption: `latest_frame()` always returns the
most recently decoded frame without blocking, so a slow consumer never
slows down ingestion, and ingestion never piles up a backlog of stale
frames waiting to be consumed.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Literal

from vip.cameras.base import CameraSource, CameraSourceError, Frame

logger = logging.getLogger(__name__)

CameraStatus = Literal["starting", "running", "reconnecting", "stopped"]


@dataclass(frozen=True)
class CameraState:
    status: CameraStatus
    last_error: str | None = None
    reconnect_attempts: int = 0


class CameraWorker:
    def __init__(
        self,
        source: CameraSource,
        *,
        reconnect_delay_seconds: float = 1.0,
        reconnect_max_delay_seconds: float = 30.0,
        on_status_change: Callable[[str, CameraState], None] | None = None,
    ) -> None:
        self._source = source
        self._reconnect_delay_seconds = reconnect_delay_seconds
        self._reconnect_max_delay_seconds = reconnect_max_delay_seconds
        self._on_status_change = on_status_change

        self._lock = threading.Lock()
        self._latest_frame: Frame | None = None
        self._state = CameraState(status="starting")

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def camera_id(self) -> str:
        return self._source.camera_id

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"camera-worker-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._source.close()
        self._set_state(CameraState(status="stopped"))

    def latest_frame(self) -> Frame | None:
        with self._lock:
            return self._latest_frame

    @property
    def state(self) -> CameraState:
        with self._lock:
            return self._state

    def _set_state(self, state: CameraState) -> None:
        with self._lock:
            self._state = state
        if self._on_status_change is not None:
            self._on_status_change(self.camera_id, state)

    def _on_failure(self, exc: CameraSourceError, attempts: int) -> None:
        self._set_state(
            CameraState(status="reconnecting", last_error=str(exc), reconnect_attempts=attempts)
        )
        logger.warning("camera %s: %s (attempt %d)", self.camera_id, exc, attempts)

    def _run(self) -> None:
        delay = self._reconnect_delay_seconds
        attempts = 0
        while not self._stop_event.is_set():
            try:
                self._source.open()
            except CameraSourceError as exc:
                attempts += 1
                self._on_failure(exc, attempts)
                if self._stop_event.wait(delay):
                    break
                delay = min(delay * 2, self._reconnect_max_delay_seconds)
                continue

            delay = self._reconnect_delay_seconds
            attempts = 0
            self._set_state(CameraState(status="running"))
            logger.info("camera %s: connected", self.camera_id)

            try:
                while not self._stop_event.is_set():
                    frame = self._source.read()
                    if frame is not None:
                        with self._lock:
                            self._latest_frame = frame
            except CameraSourceError as exc:
                attempts += 1
                self._on_failure(exc, attempts)
                self._source.close()
                if self._stop_event.wait(delay):
                    break
                delay = min(delay * 2, self._reconnect_max_delay_seconds)

        self._source.close()
