"""Enrollment: turns voluntarily provided photos into stored embeddings.

One photo in, at most one face out. A photo that cannot be used is
rejected with a stable reason code rather than silently ignored, so the UI
can tell the operator exactly why a picture was not accepted.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from vip.cameras.base import Frame
from vip.core.images import ImageDecodeError, crop, decode_image, encode_jpeg
from vip.storage.repository import FaceEmbeddingRecord, Repository
from vip.vision.base import Detection, VisionEngine

logger = logging.getLogger(__name__)

ENROLLMENT_CAMERA_ID = "enrollment"


@dataclass(frozen=True)
class Rejection:
    filename: str
    reason: str  # invalid_image | face_not_found | multiple_faces |
    # low_detection_score | missing_embedding


@dataclass(frozen=True)
class EnrollmentOutcome:
    enrolled: list[FaceEmbeddingRecord]
    rejected: list[Rejection]


class EnrollmentService:
    def __init__(
        self,
        repository: Repository,
        engine: VisionEngine,
        *,
        min_detection_score: float = 0.5,
        max_upload_bytes: int = 10 * 1024 * 1024,
        store_crops: bool = True,
    ) -> None:
        self._repository = repository
        self._engine = engine
        self._min_detection_score = min_detection_score
        self._max_upload_bytes = max_upload_bytes
        self._store_crops = store_crops

    def enroll(self, person_id: str, images: list[tuple[str, bytes]]) -> EnrollmentOutcome:
        enrolled: list[FaceEmbeddingRecord] = []
        rejected: list[Rejection] = []

        for filename, data in images:
            try:
                image = decode_image(data, max_bytes=self._max_upload_bytes)
            except ImageDecodeError as exc:
                logger.info("enrollment rejected %s: %s", filename, exc)
                rejected.append(Rejection(filename, "invalid_image"))
                continue

            detections = self._engine.process(self._as_frame(image))
            reason = self._rejection_reason(detections)
            if reason is not None:
                rejected.append(Rejection(filename, reason))
                continue

            detection = detections[0]
            model = self._engine.model_info
            crop_image = None
            crop_media_type = None
            if self._store_crops:
                box = detection.bbox
                crop_image = encode_jpeg(crop(image, (box.x1, box.y1, box.x2, box.y2)))
                crop_media_type = "image/jpeg"

            enrolled.append(
                self._repository.add_face_embedding(
                    person_id,
                    np.asarray(detection.embedding),
                    model_id=model.model_id,
                    model_version=model.model_version,
                    detection_score=detection.score,
                    bounding_box=[
                        detection.bbox.x1,
                        detection.bbox.y1,
                        detection.bbox.x2,
                        detection.bbox.y2,
                    ],
                    crop_image=crop_image,
                    crop_media_type=crop_media_type,
                )
            )

        return EnrollmentOutcome(enrolled=enrolled, rejected=rejected)

    def _rejection_reason(self, detections: list[Detection]) -> str | None:
        if not detections:
            return "face_not_found"
        if len(detections) > 1:
            # An enrollment photo must be unambiguous about whose face it is.
            return "multiple_faces"
        detection = detections[0]
        if detection.score < self._min_detection_score:
            return "low_detection_score"
        if detection.embedding is None or np.asarray(detection.embedding).size == 0:
            return "missing_embedding"
        return None

    @staticmethod
    def _as_frame(image: np.ndarray) -> Frame:
        # A VisionEngine consumes Frames; a still photo is just a frame that
        # did not come from a camera.
        return Frame(
            camera_id=ENROLLMENT_CAMERA_ID,
            image=image,
            timestamp=datetime.now(timezone.utc),
            sequence=0,
        )
