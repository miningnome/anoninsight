"""Builds a FaceEngine from a VisionConfig without the caller ever
importing InsightFaceEngine/MockFaceEngine directly."""

from __future__ import annotations

from vip.core.config import VisionConfig

from .base import FaceEngine
from .insightface_engine import InsightFaceEngine
from .mock import MockFaceEngine


def create_face_engine(config: VisionConfig) -> FaceEngine:
    if config.engine == "insightface":
        return InsightFaceEngine(
            model_pack=config.model_pack,
            ctx_id=config.ctx_id,
            min_score=config.min_score,
        )
    if config.engine == "mock":
        return MockFaceEngine()
    raise ValueError(f"unsupported vision engine: {config.engine!r}")
