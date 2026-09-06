"""Person enrollment endpoints (Fase 3).

People are registered voluntarily, from photos the operator provides. No
external lookup of any kind happens here, by design.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from vip.identity import EnrollmentOutcome
from vip.storage import DuplicateExternalIdError, PersonNotFoundError

router = APIRouter()


def _person_payload(request: Request, person) -> dict:
    repository = request.app.state.repository
    return {
        "id": person.id,
        "name": person.name,
        "external_id": person.external_id,
        "metadata": person.metadata,
        "faces": repository.count_face_embeddings(person.id),
        "created_at": person.created_at,
    }


def _enrollment_payload(outcome: EnrollmentOutcome) -> dict:
    return {
        "enrolled": [
            {
                "id": record.id,
                "detection_score": record.detection_score,
                "bounding_box": record.bounding_box,
                "model_id": record.model_id,
            }
            for record in outcome.enrolled
        ],
        "rejected": [
            {"filename": rejection.filename, "reason": rejection.reason}
            for rejection in outcome.rejected
        ],
    }


async def _read_uploads(request: Request, images: list[UploadFile]) -> list[tuple[str, bytes]]:
    max_bytes = request.app.state.settings.storage.max_upload_bytes
    uploads: list[tuple[str, bytes]] = []
    for image in images:
        data = await image.read()
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"{image.filename}: image too large")
        uploads.append((image.filename or "upload", data))
    return uploads


@router.post("/persons", status_code=201)
async def create_person(
    request: Request,
    name: str = Form(...),
    external_id: str | None = Form(None),
    images: list[UploadFile] = File(default_factory=list),
) -> dict:
    repository = request.app.state.repository
    try:
        person = repository.create_person(name=name, external_id=external_id or None)
    except DuplicateExternalIdError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    outcome = request.app.state.enrollment.enroll(
        person.id, await _read_uploads(request, images)
    )
    return {**_person_payload(request, person), "enrollment": _enrollment_payload(outcome)}


@router.get("/persons")
def list_persons(request: Request) -> list[dict]:
    repository = request.app.state.repository
    return [_person_payload(request, person) for person in repository.list_persons()]


@router.get("/persons/{person_id}")
def get_person(person_id: str, request: Request) -> dict:
    person = request.app.state.repository.get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person not found")
    return _person_payload(request, person)


@router.delete("/persons/{person_id}", status_code=204)
def delete_person(person_id: str, request: Request) -> Response:
    if not request.app.state.repository.delete_person(person_id):
        raise HTTPException(status_code=404, detail="person not found")
    return Response(status_code=204)


@router.post("/persons/{person_id}/faces")
async def add_faces(
    person_id: str,
    request: Request,
    images: list[UploadFile] = File(...),
) -> dict:
    try:
        outcome = request.app.state.enrollment.enroll(
            person_id, await _read_uploads(request, images)
        )
    except PersonNotFoundError as exc:
        raise HTTPException(status_code=404, detail="person not found") from exc
    return _enrollment_payload(outcome)


@router.get("/persons/{person_id}/faces")
def list_faces(person_id: str, request: Request) -> list[dict]:
    repository = request.app.state.repository
    if repository.get_person(person_id) is None:
        raise HTTPException(status_code=404, detail="person not found")
    return [
        {
            "id": record.id,
            "detection_score": record.detection_score,
            "bounding_box": record.bounding_box,
            "model_id": record.model_id,
            "model_version": record.model_version,
            "created_at": record.created_at,
        }
        for record in repository.list_face_embeddings(person_id)
    ]


@router.get("/persons/{person_id}/faces/{face_id}/image")
def get_face_image(person_id: str, face_id: str, request: Request) -> Response:
    crop = request.app.state.repository.get_face_crop(face_id)
    if crop is None:
        raise HTTPException(status_code=404, detail="face image not found")
    content, media_type = crop
    return Response(content=content, media_type=media_type)


@router.delete("/persons/{person_id}/faces/{face_id}", status_code=204)
def delete_face(person_id: str, face_id: str, request: Request) -> Response:
    if not request.app.state.repository.delete_face_embedding(face_id):
        raise HTTPException(status_code=404, detail="face not found")
    return Response(status_code=204)
