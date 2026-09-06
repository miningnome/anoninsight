from __future__ import annotations

import numpy as np
import pytest

from vip.identity import EnrollmentService
from vip.storage import Database, Repository
from vip.vision.base import BoundingBox, Detection
from vip.vision.face import MockFaceEngine

cv2 = pytest.importorskip("cv2")


@pytest.fixture
def repository(tmp_path) -> Repository:
    database = Database(tmp_path / "vip.db")
    database.connect()
    yield Repository(database)
    database.close()


def _photo_bytes() -> bytes:
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def _face(score: float = 0.95, embedding: np.ndarray | None = None) -> Detection:
    return Detection(
        kind="face",
        bbox=BoundingBox(8, 8, 48, 48),
        score=score,
        embedding=np.ones(512, dtype=np.float32) if embedding is None else embedding,
    )


def _service(repository: Repository, engine: MockFaceEngine, **kwargs) -> EnrollmentService:
    return EnrollmentService(repository, engine, **kwargs)


def test_enrolls_a_photo_with_exactly_one_face(repository):
    engine = MockFaceEngine()
    engine.stub_faces([_face()])
    person = repository.create_person("Carlos")

    outcome = _service(repository, engine).enroll(person.id, [("carlos.jpg", _photo_bytes())])

    assert len(outcome.enrolled) == 1
    assert outcome.rejected == []

    stored = repository.list_face_embeddings(person.id)
    assert len(stored) == 1
    assert stored[0].model_id == "mock"
    assert stored[0].detection_score == pytest.approx(0.95)
    # The crop of the detected face is stored alongside the embedding.
    assert repository.get_face_crop(stored[0].id) is not None


def test_photo_without_faces_is_rejected(repository):
    engine = MockFaceEngine()  # detects nothing
    person = repository.create_person("Carlos")

    outcome = _service(repository, engine).enroll(person.id, [("empty.jpg", _photo_bytes())])

    assert outcome.enrolled == []
    assert [r.reason for r in outcome.rejected] == ["face_not_found"]


def test_photo_with_several_faces_is_rejected(repository):
    engine = MockFaceEngine()
    engine.stub_faces([_face(), _face()])
    person = repository.create_person("Carlos")

    outcome = _service(repository, engine).enroll(person.id, [("group.jpg", _photo_bytes())])

    assert [r.reason for r in outcome.rejected] == ["multiple_faces"]
    assert repository.list_face_embeddings(person.id) == []


def test_low_score_detection_is_rejected(repository):
    engine = MockFaceEngine()
    engine.stub_faces([_face(score=0.2)])
    person = repository.create_person("Carlos")

    outcome = _service(repository, engine, min_detection_score=0.5).enroll(
        person.id, [("blurry.jpg", _photo_bytes())]
    )

    assert [r.reason for r in outcome.rejected] == ["low_detection_score"]


def test_detection_without_embedding_is_rejected(repository):
    engine = MockFaceEngine()
    engine.stub_faces([Detection(kind="face", bbox=BoundingBox(0, 0, 10, 10), score=0.9)])
    person = repository.create_person("Carlos")

    outcome = _service(repository, engine).enroll(person.id, [("x.jpg", _photo_bytes())])

    assert [r.reason for r in outcome.rejected] == ["missing_embedding"]


def test_corrupt_upload_is_rejected_without_reaching_the_engine(repository):
    engine = MockFaceEngine()
    engine.stub_faces([_face()])
    person = repository.create_person("Carlos")

    outcome = _service(repository, engine).enroll(person.id, [("broken.jpg", b"not-an-image")])

    assert [r.reason for r in outcome.rejected] == ["invalid_image"]
    assert repository.list_face_embeddings(person.id) == []


def test_partial_success_reports_both_sides(repository):
    engine = MockFaceEngine()
    engine.stub_faces([_face()])
    person = repository.create_person("Carlos")

    outcome = _service(repository, engine).enroll(
        person.id,
        [("good.jpg", _photo_bytes()), ("broken.jpg", b"nope")],
    )

    assert len(outcome.enrolled) == 1
    assert [r.filename for r in outcome.rejected] == ["broken.jpg"]


def test_crops_can_be_disabled(repository):
    engine = MockFaceEngine()
    engine.stub_faces([_face()])
    person = repository.create_person("Carlos")

    outcome = _service(repository, engine, store_crops=False).enroll(
        person.id, [("carlos.jpg", _photo_bytes())]
    )

    assert repository.get_face_crop(outcome.enrolled[0].id) is None
