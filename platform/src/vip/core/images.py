"""Decoding of user-uploaded images, with explicit limits.

Uploads are untrusted input: size is capped before decoding and the result
is validated to be a normal 3-channel BGR image, so a malformed or
oversized file fails cleanly instead of reaching the vision engine.
"""

from __future__ import annotations

import numpy as np

MAX_PIXELS = 50_000_000  # ~50 MP, well above any legitimate enrollment photo


class ImageDecodeError(ValueError):
    pass


def decode_image(data: bytes, *, max_bytes: int) -> np.ndarray:
    import cv2

    if not data:
        raise ImageDecodeError("empty image")
    if len(data) > max_bytes:
        raise ImageDecodeError(f"image is larger than {max_bytes} bytes")

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ImageDecodeError("unsupported or corrupt image")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImageDecodeError("expected a 3-channel colour image")
    if image.shape[0] * image.shape[1] > MAX_PIXELS:
        raise ImageDecodeError("image resolution is too large")
    return image


def encode_jpeg(image: np.ndarray, *, quality: int = 90) -> bytes:
    import cv2

    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise ImageDecodeError("could not encode image as JPEG")
    return buffer.tobytes()


def crop(image: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    """Crops to a bounding box, clamped to the image bounds."""
    height, width = image.shape[:2]
    x1 = max(0, min(int(box[0]), width - 1))
    y1 = max(0, min(int(box[1]), height - 1))
    x2 = max(x1 + 1, min(int(box[2]), width))
    y2 = max(y1 + 1, min(int(box[3]), height))
    return image[y1:y2, x1:x2]
