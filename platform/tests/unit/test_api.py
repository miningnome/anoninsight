from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vip.api.app import create_app
from vip.core.config import Settings, StorageConfig


@pytest.fixture
def client(tmp_path):
    # No cameras configured: exercises the API/web wiring without needing
    # any real camera hardware or video file. The database goes to tmp_path
    # so tests never write into the working tree.
    settings = Settings(storage=StorageConfig(database_path=str(tmp_path / "vip.db")))
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_index_page_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Video Intelligence Platform" in response.text


def test_list_cameras_empty_by_default(client):
    response = client.get("/api/cameras")
    assert response.status_code == 200
    assert response.json() == []


def test_unknown_camera_returns_404(client):
    response = client.get("/api/cameras/does-not-exist")
    assert response.status_code == 404
