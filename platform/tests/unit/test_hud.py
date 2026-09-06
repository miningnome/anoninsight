"""Unit test for the frame-annotation logic used by the MJPEG stream
endpoint (vip.api.routes.cameras._draw_detections), isolated from the ASGI
streaming machinery - see test_api_camera_integration.py for why the live
stream itself is not exercised through TestClient."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from vip.api.routes.cameras import _draw_detections
from vip.vision.base import BoundingBox, Detection


def test_draw_detections_returns_a_new_array_without_mutating_the_input():
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    detection = Detection(kind="face", bbox=BoundingBox(2, 2, 10, 10), score=0.87)

    annotated = _draw_detections(image, [detection])

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated, image)  # a box was actually drawn
    assert np.array_equal(image, np.zeros((20, 20, 3), dtype=np.uint8))  # input untouched


def test_draw_detections_with_no_detections_returns_unchanged_copy():
    image = np.full((10, 10, 3), 5, dtype=np.uint8)

    annotated = _draw_detections(image, [])

    assert np.array_equal(annotated, image)
    assert annotated is not image
