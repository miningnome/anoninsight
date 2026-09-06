"""Camera endpoints: list configured cameras, their runtime status, and a
live MJPEG stream per camera. No AI analysis yet (Fase 1)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

router = APIRouter()


@router.get("/cameras")
def list_cameras(request: Request) -> list[dict]:
    registry = request.app.state.cameras
    return [
        {
            "id": config.id,
            "name": config.name,
            "source_type": config.source_type,
            "status": worker.state.status,
        }
        for config, worker in registry.all()
    ]


@router.get("/cameras/{camera_id}")
def get_camera(camera_id: str, request: Request) -> dict:
    entry = request.app.state.cameras.get(camera_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="camera not found")
    config, worker = entry
    state = worker.state
    return {
        "id": config.id,
        "name": config.name,
        "source_type": config.source_type,
        "status": state.status,
        "last_error": state.last_error,
        "reconnect_attempts": state.reconnect_attempts,
    }


@router.get("/cameras/{camera_id}/stream.mjpg")
async def stream_camera(camera_id: str, request: Request) -> StreamingResponse:
    entry = request.app.state.cameras.get(camera_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="camera not found")
    config, worker = entry

    async def generate():
        import cv2

        interval = 1.0 / config.display_fps if config.display_fps > 0 else 0.0
        last_sequence = -1
        while True:
            if await request.is_disconnected():
                break
            frame = worker.latest_frame()
            if frame is not None and frame.sequence != last_sequence:
                last_sequence = frame.sequence
                ok, buffer = cv2.imencode(".jpg", frame.image)
                if ok:
                    chunk = buffer.tobytes()
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(chunk)).encode() + b"\r\n\r\n"
                        + chunk
                        + b"\r\n"
                    )
            await asyncio.sleep(interval)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")
