"""Deterministic FaceEngine for tests and for running the platform without
real model weights or network access - mirrors the purpose of server/'s
own MockInferenceEngine (server/backend/insightface_server/inference/mock.py).

Detects nothing by default; call `stub_faces()` to make it return fixed
Detection objects, e.g. to exercise the HUD overlay end-to-end in tests.
"""

from __future__ import annotations

from vip.cameras.base import Frame
from vip.vision.base import Detection

from .base import FaceEngine


class MockFaceEngine(FaceEngine):
    def __init__(self) -> None:
        self._stub_faces: list[Detection] = []

    def warmup(self) -> None:
        pass

    def stub_faces(self, detections: list[Detection]) -> None:
        self._stub_faces = detections

    def process(self, frame: Frame) -> list[Detection]:
        return list(self._stub_faces)

    def close(self) -> None:
        pass
