"""Serializes access to a VisionEngine shared across multiple cameras.

A single VisionEngine instance is deliberately shared across all cameras
(one loaded model, not one per camera) so memory/VRAM usage does not grow
linearly with the number of cameras. Mutable internal state in some
detectors (e.g. SCRFD's center-point cache) is not guaranteed thread-safe,
so concurrent calls are serialized here. A bounded concurrency pool -
instead of a single lock - is a Fase 7 concern once multi-camera
throughput actually matters.
"""

from __future__ import annotations

import threading

from vip.cameras.base import Frame

from .base import Detection, VisionEngine


class SerializedVisionEngine(VisionEngine):
    def __init__(self, inner: VisionEngine) -> None:
        self.name = inner.name
        self._inner = inner
        self._lock = threading.Lock()

    def warmup(self) -> None:
        with self._lock:
            self._inner.warmup()

    def process(self, frame: Frame) -> list[Detection]:
        with self._lock:
            return self._inner.process(frame)

    def close(self) -> None:
        with self._lock:
            self._inner.close()
