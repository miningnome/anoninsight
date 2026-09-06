from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pytest

from vip.cameras.base import CameraSource, CameraSourceError, Frame
from vip.pipeline.ingestion import CameraWorker


class _ScriptedSource(CameraSource):
    """A CameraSource whose open()/read() behavior is scripted, so
    CameraWorker's reconnect/backoff logic can be exercised without any
    real video decoding or camera hardware.

    `frames_per_session` defaults to a very large number so that, once
    connected, the worker settles into a stable "running" state instead of
    cycling through reconnects every time reads run out - which would make
    the "running" state too short-lived for a polling assertion to catch
    reliably. `first_session_frames`, when set, overrides that count only
    for the very first open() session, to model a camera that drops once
    early on and then stays connected.
    """

    def __init__(
        self,
        camera_id: str,
        open_failures: int = 0,
        frames_per_session: int = 10**6,
        first_session_frames: int | None = None,
    ) -> None:
        super().__init__(camera_id)
        self._open_failures_left = open_failures
        self._frames_per_session = frames_per_session
        self._first_session_frames = first_session_frames
        self._session_count = 0
        self._frames_left = 0
        self._opened = False
        self._sequence = 0
        self.open_calls = 0

    def open(self) -> None:
        self.open_calls += 1
        if self._open_failures_left > 0:
            self._open_failures_left -= 1
            raise CameraSourceError("scripted open failure")
        self._opened = True
        self._session_count += 1
        if self._session_count == 1 and self._first_session_frames is not None:
            self._frames_left = self._first_session_frames
        else:
            self._frames_left = self._frames_per_session

    def read(self) -> Frame | None:
        if not self._opened:
            raise CameraSourceError("read before open")
        if self._frames_left <= 0:
            raise CameraSourceError("scripted read failure")
        self._frames_left -= 1
        self._sequence += 1
        return Frame(
            camera_id=self.camera_id,
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            timestamp=datetime.now(timezone.utc),
            sequence=self._sequence,
        )

    def close(self) -> None:
        self._opened = False

    @property
    def is_open(self) -> bool:
        return self._opened


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_worker_reaches_running_and_publishes_frames():
    source = _ScriptedSource("cam-1")
    worker = CameraWorker(source, reconnect_delay_seconds=0.01, reconnect_max_delay_seconds=0.05)
    worker.start()
    try:
        assert _wait_until(lambda: worker.state.status == "running")
        assert _wait_until(lambda: worker.latest_frame() is not None)
    finally:
        worker.stop()
    assert worker.state.status == "stopped"


def test_worker_reconnects_after_open_failures():
    source = _ScriptedSource("cam-1", open_failures=3)
    worker = CameraWorker(source, reconnect_delay_seconds=0.01, reconnect_max_delay_seconds=0.05)
    worker.start()
    try:
        assert _wait_until(lambda: worker.state.status == "running")
    finally:
        worker.stop()
    assert source.open_calls >= 4  # 3 failed attempts + 1 successful


def test_worker_recovers_after_read_failure():
    # The first session yields exactly one frame before read() fails,
    # forcing a reconnect; the second session stays connected, so the
    # worker must settle back into "running" instead of giving up.
    source = _ScriptedSource("cam-1", first_session_frames=1)
    worker = CameraWorker(source, reconnect_delay_seconds=0.01, reconnect_max_delay_seconds=0.05)
    worker.start()
    try:
        assert _wait_until(lambda: worker.state.reconnect_attempts >= 1)
        assert _wait_until(
            lambda: worker.state.status == "running" and worker.latest_frame() is not None
        )
    finally:
        worker.stop()


def test_stop_before_start_is_a_noop():
    source = _ScriptedSource("cam-1", frames_per_session=1)
    worker = CameraWorker(source)
    worker.stop()
    assert worker.state.status == "stopped"


def test_on_status_change_callback_reports_running():
    events: list[tuple[str, str]] = []
    source = _ScriptedSource("cam-1", frames_per_session=100)
    worker = CameraWorker(
        source,
        reconnect_delay_seconds=0.01,
        on_status_change=lambda camera_id, state: events.append((camera_id, state.status)),
    )
    worker.start()
    try:
        assert _wait_until(lambda: ("cam-1", "running") in events)
    finally:
        worker.stop()
