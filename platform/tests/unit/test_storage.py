from __future__ import annotations

import numpy as np
import pytest

from vip.storage import Database, DuplicateExternalIdError, PersonNotFoundError, Repository


@pytest.fixture
def repository(tmp_path) -> Repository:
    database = Database(tmp_path / "vip.db")
    database.connect()
    yield Repository(database)
    database.close()


def test_migrations_create_the_expected_tables(tmp_path):
    database = Database(tmp_path / "vip.db")
    database.connect()
    try:
        tables = {
            row["name"]
            for row in database.read("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {"persons", "face_embeddings", "schema_migrations"} <= tables
    finally:
        database.close()


def test_migrations_are_applied_once(tmp_path):
    path = tmp_path / "vip.db"
    first = Database(path)
    first.connect()
    first.close()

    second = Database(path)
    second.connect()
    try:
        applied = second.read("SELECT name FROM schema_migrations")
        assert len(applied) == len({row["name"] for row in applied})
    finally:
        second.close()


def test_create_and_read_person(repository):
    person = repository.create_person("Carlos Garcia", external_id="EMP-1")

    stored = repository.get_person(person.id)
    assert stored is not None
    assert stored.name == "Carlos Garcia"
    assert stored.external_id == "EMP-1"
    assert repository.list_persons() == [stored]


def test_duplicate_external_id_is_rejected(repository):
    repository.create_person("Carlos", external_id="EMP-1")
    with pytest.raises(DuplicateExternalIdError):
        repository.create_person("Otro", external_id="EMP-1")


def test_embedding_roundtrip_preserves_values(repository):
    person = repository.create_person("Carlos")
    embedding = np.linspace(-1.0, 1.0, 512, dtype=np.float32)

    record = repository.add_face_embedding(
        person.id,
        embedding,
        model_id="mock",
        model_version="1",
        detection_score=0.93,
        bounding_box=[1.0, 2.0, 3.0, 4.0],
    )

    stored = repository.list_face_embeddings(person.id)
    assert len(stored) == 1
    assert stored[0].id == record.id
    assert stored[0].bounding_box == [1.0, 2.0, 3.0, 4.0]
    assert stored[0].model_id == "mock"
    np.testing.assert_allclose(stored[0].embedding, embedding)


def test_face_for_unknown_person_is_rejected(repository):
    with pytest.raises(PersonNotFoundError):
        repository.add_face_embedding(
            "does-not-exist",
            np.zeros(4, dtype=np.float32),
            model_id="mock",
            model_version="1",
            detection_score=0.9,
            bounding_box=[0, 0, 1, 1],
        )


def test_deleting_a_person_cascades_to_their_embeddings(repository):
    person = repository.create_person("Carlos")
    repository.add_face_embedding(
        person.id,
        np.zeros(8, dtype=np.float32),
        model_id="mock",
        model_version="1",
        detection_score=0.9,
        bounding_box=[0, 0, 1, 1],
    )

    assert repository.delete_person(person.id) is True
    assert repository.get_person(person.id) is None
    assert repository.list_face_embeddings(person.id) == []


def test_face_crops_are_stored_and_retrievable(repository):
    person = repository.create_person("Carlos")
    record = repository.add_face_embedding(
        person.id,
        np.zeros(8, dtype=np.float32),
        model_id="mock",
        model_version="1",
        detection_score=0.9,
        bounding_box=[0, 0, 1, 1],
        crop_image=b"fake-jpeg-bytes",
        crop_media_type="image/jpeg",
    )

    crop = repository.get_face_crop(record.id)
    assert crop == (b"fake-jpeg-bytes", "image/jpeg")

    assert repository.delete_face_embedding(record.id) is True
    assert repository.get_face_crop(record.id) is None
