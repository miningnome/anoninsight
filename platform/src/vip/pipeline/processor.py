"""FrameProcessor takes the latest frame from a CameraWorker at a
configurable analysis rate and runs it through one or more VisionEngine
instances, publishing the latest Detection[] for whoever wants them (the
HUD overlay in Fase 2; the identity matcher and event engine from Fase
3/4 onward).

Like CameraWorker, this decouples analysis from both capture and display:
frames are not reprocessed, and a slow engine never blocks ingestion.
"""

from __future__ import annotations

import logging
import threading
from typing import Sequence

from vip.pipeline.ingestion import CameraWorker
from vip.vision.base import Detection, VisionEngine

logger = logging.getLogger(__name__)


class FrameProcessor:
    def __init__(
        self,
        worker: CameraWorker,
        engines: Sequence[VisionEngine],
        *,
        analysis_fps: float = 5.0,
    ) -> None:
        self._worker = worker
        self._engines = list(engines)
        self._interval = 1.0 / analysis_fps if analysis_fps > 0 else 0.0

        self._lock = threading.Lock()
        self._latest_detections: list[Detection] = []
        self._last_sequence = -1

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        for engine in self._engines:
            engine.warmup()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"frame-processor-{self._worker.camera_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        for engine in self._engines:
            engine.close()

    def latest_detections(self) -> list[Detection]:
        with self._lock:
            return list(self._latest_detections)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            frame = self._worker.latest_frame()
            if frame is not None and frame.sequence != self._last_sequence:
                self._last_sequence = frame.sequence
                detections: list[Detection] = []
                for engine in self._engines:
                    # A broken model/engine must not kill this thread - the
                    # camera keeps streaming, it just stops annotating until
                    # the engine recovers.
                    try:
                        detections.extend(engine.process(frame))
                    except Exception:
                        logger.exception(
                            "camera %s: vision engine %s failed to process frame",
                            self._worker.camera_id,
                            engine.name,
                        )
                with self._lock:
                    self._latest_detections = detections
            self._stop_event.wait(self._interval)
