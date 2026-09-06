from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from vip.cameras.base import Frame
from vip.core.config import VisionConfig
from vip.vision.base import BoundingBox, Detection
from vip.vision.face import InsightFaceEngine, MockFaceEngine, create_face_engine


def _frame() -> Frame:
    return Frame(
        camera_id="cam-1",
        image=np.zeros((8, 8, 3), dtype=np.uint8),
        timestamp=datetime.now(timezone.utc),
        sequence=1,
    )


def test_mock_face_engine_returns_no_faces_by_default():
    engine = MockFaceEngine()
    engine.warmup()
    assert engine.process(_frame()) == []


def test_mock_face_engine_returns_stubbed_faces():
    engine = MockFaceEngine()
    detection = Detection(kind="face", bbox=BoundingBox(1, 2, 3, 4), score=0.9)
    engine.stub_faces([detection])
    assert engine.process(_frame()) == [detection]


def test_create_face_engine_dispatches_mock():
    engine = create_face_engine(VisionConfig(engine="mock"))
    assert isinstance(engine, MockFaceEngine)


def test_create_face_engine_dispatches_insightface_without_importing_it():
    # Constructing InsightFaceEngine must not require the `insightface`
    # package or any model weights - only warmup() does.
    engine = create_face_engine(VisionConfig(engine="insightface", ctx_id=-1))
    assert isinstance(engine, InsightFaceEngine)
