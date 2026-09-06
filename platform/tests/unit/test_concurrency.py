from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import numpy as np

from vip.cameras.base import Frame
from vip.vision.base import ModelInfo, VisionEngine
from vip.vision.concurrency import SerializedVisionEngine


class _SlowEngine(VisionEngine):
    name = "slow"

    def __init__(self) -> None:
        self.warmup_calls = 0
        self.close_calls = 0
        self._active = 0
        self._max_active = 0
        self._lock = threading.Lock()

    @property
    def model_info(self) -> ModelInfo:
        return ModelInfo(model_id="slow", model_version="1")

    def warmup(self) -> None:
        self.warmup_calls += 1

    def process(self, frame: Frame):
        with self._lock:
            self._active += 1
            self._max_active = max(self._max_active, self._active)
        time.sleep(0.03)
        with self._lock:
            self._active -= 1
        return []

    def close(self) -> None:
        self.close_calls += 1


def _frame() -> Frame:
    return Frame(
        camera_id="cam-1",
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp=datetime.now(timezone.utc),
        sequence=1,
    )


def test_serialized_engine_delegates_lifecycle_calls():
    inner = _SlowEngine()
    engine = SerializedVisionEngine(inner)

    engine.warmup()
    engine.process(_frame())
    engine.close()

    assert inner.warmup_calls == 1
    assert inner.close_calls == 1
    assert engine.name == "slow"


def test_serialized_engine_prevents_concurrent_process_calls():
    inner = _SlowEngine()
    engine = SerializedVisionEngine(inner)

    threads = [threading.Thread(target=engine.process, args=(_frame(),)) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2.0)

    assert inner._max_active == 1
