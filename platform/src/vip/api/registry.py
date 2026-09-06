"""In-memory registry of configured cameras and their CameraWorker.

Fase 1 keeps this purely in-memory, built once from Settings at startup.
Persisting camera configuration and CRUD over the API arrive later (see
ARCHITECTURE.md, Fase 6/7) once there is a database.
"""

from __future__ import annotations

from vip.core.config import CameraConfig
from vip.pipeline import CameraWorker


class CameraRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[CameraConfig, CameraWorker]] = {}

    def add(self, config: CameraConfig, worker: CameraWorker) -> None:
        self._entries[config.id] = (config, worker)

    def get(self, camera_id: str) -> tuple[CameraConfig, CameraWorker] | None:
        return self._entries.get(camera_id)

    def all(self) -> list[tuple[CameraConfig, CameraWorker]]:
        return list(self._entries.values())

    def stop_all(self) -> None:
        for _, worker in self._entries.values():
            worker.stop()
        self._entries.clear()
