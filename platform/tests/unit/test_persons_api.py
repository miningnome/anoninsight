from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vip.api.app import create_app
from vip.core.config import Settings, StorageConfig, VisionConfig
from vip.vision.base import BoundingBox, Detection

cv2 = pytest.importorskip("cv2")


@pytest.fixture
def client(tmp_path):
    settings = Settings(
        vision=VisionConfig(engine="mock"),
        storage=StorageConfig(database_path=str(tmp_path / "vip.db")),
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def _stub_one_face(client: TestClient) -> None:
    """The mock engine detects nothing unless told to; the app shares one
    engine instance, so stubbing it here is what the enrollment path sees."""
    engine = client.app.state.enrollment._engine  # SerializedVisionEngine
    engine._inner.stub_faces(
        [
            Detection(
                kind="face",
                bbox=BoundingBox(8, 8, 48, 48),
                score=0.95,
                embedding=np.ones(512, dtype=np.float32),
            )
        ]
    )


def _photo() -> bytes:
    ok, buffer = cv2.imencode(".jpg", np.full((64, 64, 3), 120, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def test_persons_list_is_empty_initially(client):
    assert client.get("/api/persons").json() == []


def test_create_person_without_photos(client):
    response = client.post("/api/persons", data={"name": "Carlos Garcia"})

    assert response.status_code == 201
    body = response.json()
    assert body["name"] == "Carlos Garcia"
    assert body["faces"] == 0
    assert body["enrollment"] == {"enrolled": [], "rejected": []}


def test_create_person_with_a_photo_enrolls_a_face(client):
    _stub_one_face(client)

    response = client.post(
        "/api/persons",
        data={"name": "Carlos Garcia", "external_id": "EMP-1"},
        files=[("images", ("carlos.jpg", _photo(), "image/jpeg"))],
    )

    assert response.status_code == 201
    body = response.json()
    assert len(body["enrollment"]["enrolled"]) == 1
    assert body["enrollment"]["rejected"] == []

    detail = client.get(f"/api/persons/{body['id']}").json()
    assert detail["faces"] == 1

    faces = client.get(f"/api/persons/{body['id']}/faces").json()
    assert faces[0]["model_id"] == "mock"

    image = client.get(f"/api/persons/{body['id']}/faces/{faces[0]['id']}/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/jpeg"


def test_duplicate_external_id_conflicts(client):
    client.post("/api/persons", data={"name": "Carlos", "external_id": "EMP-1"})
    response = client.post("/api/persons", data={"name": "Otro", "external_id": "EMP-1"})
    assert response.status_code == 409


def test_photo_without_face_is_reported_as_rejected(client):
    response = client.post(
        "/api/persons",
        data={"name": "Carlos"},
        files=[("images", ("empty.jpg", _photo(), "image/jpeg"))],
    )

    body = response.json()
    assert body["enrollment"]["enrolled"] == []
    assert body["enrollment"]["rejected"] == [
        {"filename": "empty.jpg", "reason": "face_not_found"}
    ]


def test_add_faces_to_existing_person(client):
    person = client.post("/api/persons", data={"name": "Carlos"}).json()
    _stub_one_face(client)

    response = client.post(
        f"/api/persons/{person['id']}/faces",
        files=[("images", ("carlos.jpg", _photo(), "image/jpeg"))],
    )

    assert response.status_code == 200
    assert len(response.json()["enrolled"]) == 1
    assert client.get(f"/api/persons/{person['id']}").json()["faces"] == 1


def test_delete_person_and_unknown_lookups(client):
    person = client.post("/api/persons", data={"name": "Carlos"}).json()

    assert client.delete(f"/api/persons/{person['id']}").status_code == 204
    assert client.get(f"/api/persons/{person['id']}").status_code == 404
    assert client.delete(f"/api/persons/{person['id']}").status_code == 404
    assert client.get("/api/persons/unknown/faces").status_code == 404


def test_delete_face(client):
    _stub_one_face(client)
    person = client.post(
        "/api/persons",
        data={"name": "Carlos"},
        files=[("images", ("carlos.jpg", _photo(), "image/jpeg"))],
    ).json()
    face_id = person["enrollment"]["enrolled"][0]["id"]

    assert client.delete(f"/api/persons/{person['id']}/faces/{face_id}").status_code == 204
    assert client.get(f"/api/persons/{person['id']}").json()["faces"] == 0
