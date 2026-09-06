"""VisionEngine abstraction.

Converts a Frame into a list of Detection. FaceEngine (Fase 2: detection;
Fase 3/4: embeddings) is the first implementation; ObjectDetectionEngine,
OCREngine and TrackingEngine are added later (Fase 9) without changing
this interface or anything that depends on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vip.cameras.base import Frame


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Detection:
    kind: str  # "face" | "object" | "text" | ...
    bbox: BoundingBox
    score: float
    embedding: np.ndarray | None = None
    label: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


class VisionEngine(ABC):
    name: str

    @abstractmethod
    def warmup(self) -> None:
        """Load models / allocate resources. Called once before process()."""

    @abstractmethod
    def process(self, frame: Frame) -> list[Detection]: ...

    @abstractmethod
    def close(self) -> None: ...
