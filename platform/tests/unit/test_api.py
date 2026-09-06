from __future__ import annotations

from fastapi.testclient import TestClient

from vip.api.app import create_app
from vip.core.config import Settings


def _client() -> TestClient:
    # No cameras configured: exercises the API/web wiring without needing
    # any real camera hardware or video file.
    app = create_app(Settings())
    return TestClient(app)


def test_index_page_is_served():
    with _client() as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Video Intelligence Platform" in response.text


def test_list_cameras_empty_by_default():
    with _client() as client:
        response = client.get("/api/cameras")
    assert response.status_code == 200
    assert response.json() == []


def test_unknown_camera_returns_404():
    with _client() as client:
        response = client.get("/api/cameras/does-not-exist")
    assert response.status_code == 404
