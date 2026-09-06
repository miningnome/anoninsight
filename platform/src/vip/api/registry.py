"""In-memory registry of configured cameras, their CameraWorker and their
FrameProcessor.

Fase 1 keeps this purely in-memory, built once from Settings at startup.
Persisting camera configuration and CRUD over the API arrive later (see
ARCHITECTURE.md, Fase 6/7) once there is a database.
"""

from __future__ import annotations

from dataclasses import dataclass

from vip.core.config import CameraConfig
from vip.pipeline import CameraWorker, FrameProcessor


@dataclass(frozen=True)
class CameraEntry:
    config: CameraConfig
    worker: CameraWorker
    processor: FrameProcessor


class CameraRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, CameraEntry] = {}

    def add(self, config: CameraConfig, worker: CameraWorker, processor: FrameProcessor) -> None:
        self._entries[config.id] = CameraEntry(config, worker, processor)

    def get(self, camera_id: str) -> CameraEntry | None:
        return self._entries.get(camera_id)

    def all(self) -> list[CameraEntry]:
        return list(self._entries.values())

    def stop_all(self) -> None:
        for entry in self._entries.values():
            entry.processor.stop()
            entry.worker.stop()
        self._entries.clear()
