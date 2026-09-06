from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np

from vip.cameras.base import Frame
from vip.pipeline.processor import FrameProcessor
from vip.vision.base import BoundingBox, Detection, VisionEngine


class _FakeWorker:
    """Stands in for CameraWorker: FrameProcessor only ever calls
    `camera_id` and `latest_frame()` on it."""

    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id
        self._frame: Frame | None = None

    def set_frame(self, frame: Frame) -> None:
        self._frame = frame

    def latest_frame(self) -> Frame | None:
        return self._frame


class _RecordingEngine(VisionEngine):
    name = "rec"

    def __init__(self, detections_by_sequence: dict[int, list[Detection]] | None = None) -> None:
        self.calls: list[int] = []
        self.closed = False
        self._detections_by_sequence = detections_by_sequence or {}

    def warmup(self) -> None:
        pass

    def process(self, frame: Frame) -> list[Detection]:
        self.calls.append(frame.sequence)
        return self._detections_by_sequence.get(frame.sequence, [])

    def close(self) -> None:
        self.closed = True


class _FailingEngine(VisionEngine):
    name = "failing"

    def warmup(self) -> None:
        pass

    def process(self, frame: Frame) -> list[Detection]:
        raise RuntimeError("boom")

    def close(self) -> None:
        pass


def _make_frame(sequence: int) -> Frame:
    return Frame(
        camera_id="cam-1",
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp=datetime.now(timezone.utc),
        sequence=sequence,
    )


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_processor_publishes_detections_and_processes_each_frame_once():
    worker = _FakeWorker("cam-1")
    detection = Detection(kind="face", bbox=BoundingBox(0, 0, 1, 1), score=0.9)
    engine = _RecordingEngine({1: [detection]})
    processor = FrameProcessor(worker, [engine], analysis_fps=200.0)

    worker.set_frame(_make_frame(1))
    processor.start()
    try:
        assert _wait_until(lambda: processor.latest_detections() == [detection])
        time.sleep(0.05)  # let several more cycles pass
    finally:
        processor.stop()

    assert engine.calls == [1]


def test_processor_reprocesses_when_a_new_frame_sequence_arrives():
    worker = _FakeWorker("cam-1")
    engine = _RecordingEngine()
    processor = FrameProcessor(worker, [engine], analysis_fps=200.0)

    worker.set_frame(_make_frame(1))
    processor.start()
    try:
        assert _wait_until(lambda: engine.calls == [1])
        worker.set_frame(_make_frame(2))
        assert _wait_until(lambda: engine.calls == [1, 2])
    finally:
        processor.stop()


def test_engine_exception_does_not_kill_the_processing_thread():
    worker = _FakeWorker("cam-1")
    processor = FrameProcessor(worker, [_FailingEngine()], analysis_fps=200.0)

    worker.set_frame(_make_frame(1))
    processor.start()
    try:
        time.sleep(0.05)
        assert processor.latest_detections() == []
    finally:
        processor.stop()


def test_stop_closes_engines():
    worker = _FakeWorker("cam-1")
    engine = _RecordingEngine()
    processor = FrameProcessor(worker, [engine], analysis_fps=200.0)

    processor.start()
    processor.stop()

    assert engine.closed
