"""Repository: the only place that knows SQL.

Everything above works with the dataclasses defined here, so swapping
SQLite for PostgreSQL later does not leak into the rest of the platform.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .database import Database

EMBEDDING_DTYPE = np.float32


@dataclass(frozen=True)
class Person:
    id: str
    name: str
    external_id: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class FaceEmbeddingRecord:
    id: str
    person_id: str
    embedding: np.ndarray
    model_id: str
    model_version: str
    detection_score: float
    bounding_box: list[float]
    created_at: str = ""


class PersonNotFoundError(LookupError):
    pass


class DuplicateExternalIdError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encode_embedding(embedding: np.ndarray) -> bytes:
    return np.asarray(embedding, dtype=EMBEDDING_DTYPE).tobytes()


def _decode_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=EMBEDDING_DTYPE)


class Repository:
    def __init__(self, database: Database) -> None:
        self._db = database

    # --- persons -----------------------------------------------------

    def create_person(
        self,
        name: str,
        *,
        external_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Person:
        person = Person(
            id=str(uuid.uuid4()),
            name=name,
            external_id=external_id,
            metadata=metadata or {},
            created_at=_now(),
            updated_at=_now(),
        )
        if external_id is not None and self.get_person_by_external_id(external_id) is not None:
            raise DuplicateExternalIdError(f"external_id already registered: {external_id!r}")
        self._db.write(
            "INSERT INTO persons (id, name, external_id, metadata_json, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                person.id,
                person.name,
                person.external_id,
                json.dumps(person.metadata),
                person.created_at,
                person.updated_at,
            ),
        )
        return person

    def get_person(self, person_id: str) -> Person | None:
        rows = self._db.read("SELECT * FROM persons WHERE id = ?", (person_id,))
        return self._row_to_person(rows[0]) if rows else None

    def get_person_by_external_id(self, external_id: str) -> Person | None:
        rows = self._db.read("SELECT * FROM persons WHERE external_id = ?", (external_id,))
        return self._row_to_person(rows[0]) if rows else None

    def list_persons(self) -> list[Person]:
        rows = self._db.read("SELECT * FROM persons ORDER BY created_at ASC")
        return [self._row_to_person(row) for row in rows]

    def delete_person(self, person_id: str) -> bool:
        if self.get_person(person_id) is None:
            return False
        # face_embeddings rows go with it (ON DELETE CASCADE).
        self._db.write("DELETE FROM persons WHERE id = ?", (person_id,))
        return True

    # --- face embeddings ---------------------------------------------

    def add_face_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        *,
        model_id: str,
        model_version: str,
        detection_score: float,
        bounding_box: list[float],
        crop_image: bytes | None = None,
        crop_media_type: str | None = None,
    ) -> FaceEmbeddingRecord:
        if self.get_person(person_id) is None:
            raise PersonNotFoundError(person_id)
        record = FaceEmbeddingRecord(
            id=str(uuid.uuid4()),
            person_id=person_id,
            embedding=np.asarray(embedding, dtype=EMBEDDING_DTYPE),
            model_id=model_id,
            model_version=model_version,
            detection_score=detection_score,
            bounding_box=bounding_box,
            created_at=_now(),
        )
        self._db.write(
            "INSERT INTO face_embeddings (id, person_id, embedding, embedding_dimension,"
            " model_id, model_version, detection_score, bounding_box_json, crop_image,"
            " crop_media_type, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.id,
                record.person_id,
                _encode_embedding(record.embedding),
                int(record.embedding.shape[0]),
                record.model_id,
                record.model_version,
                record.detection_score,
                json.dumps(record.bounding_box),
                crop_image,
                crop_media_type,
                record.created_at,
            ),
        )
        return record

    def list_face_embeddings(self, person_id: str | None = None) -> list[FaceEmbeddingRecord]:
        if person_id is None:
            rows = self._db.read("SELECT * FROM face_embeddings ORDER BY created_at ASC")
        else:
            rows = self._db.read(
                "SELECT * FROM face_embeddings WHERE person_id = ? ORDER BY created_at ASC",
                (person_id,),
            )
        return [self._row_to_face(row) for row in rows]

    def count_face_embeddings(self, person_id: str) -> int:
        rows = self._db.read(
            "SELECT COUNT(*) AS total FROM face_embeddings WHERE person_id = ?", (person_id,)
        )
        return int(rows[0]["total"])

    def get_face_crop(self, face_id: str) -> tuple[bytes, str] | None:
        rows = self._db.read(
            "SELECT crop_image, crop_media_type FROM face_embeddings WHERE id = ?", (face_id,)
        )
        if not rows or rows[0]["crop_image"] is None:
            return None
        return bytes(rows[0]["crop_image"]), rows[0]["crop_media_type"] or "image/jpeg"

    def delete_face_embedding(self, face_id: str) -> bool:
        rows = self._db.read("SELECT id FROM face_embeddings WHERE id = ?", (face_id,))
        if not rows:
            return False
        self._db.write("DELETE FROM face_embeddings WHERE id = ?", (face_id,))
        return True

    # --- mapping -----------------------------------------------------

    @staticmethod
    def _row_to_person(row) -> Person:
        return Person(
            id=row["id"],
            name=row["name"],
            external_id=row["external_id"],
            metadata=json.loads(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_face(row) -> FaceEmbeddingRecord:
        return FaceEmbeddingRecord(
            id=row["id"],
            person_id=row["person_id"],
            embedding=_decode_embedding(row["embedding"]),
            model_id=row["model_id"],
            model_version=row["model_version"],
            detection_score=row["detection_score"],
            bounding_box=json.loads(row["bounding_box_json"]),
            created_at=row["created_at"],
        )
