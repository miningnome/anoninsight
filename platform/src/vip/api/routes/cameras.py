"""Camera endpoints: list configured cameras, their runtime status, and a
live MJPEG stream per camera annotated with the latest face detections
(Fase 2: bounding boxes only, no recognition yet)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from vip.vision.base import Detection

router = APIRouter()


@router.get("/cameras")
def list_cameras(request: Request) -> list[dict]:
    registry = request.app.state.cameras
    return [
        {
            "id": entry.config.id,
            "name": entry.config.name,
            "source_type": entry.config.source_type,
            "status": entry.worker.state.status,
        }
        for entry in registry.all()
    ]


@router.get("/cameras/{camera_id}")
def get_camera(camera_id: str, request: Request) -> dict:
    entry = request.app.state.cameras.get(camera_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="camera not found")
    state = entry.worker.state
    return {
        "id": entry.config.id,
        "name": entry.config.name,
        "source_type": entry.config.source_type,
        "status": state.status,
        "last_error": state.last_error,
        "reconnect_attempts": state.reconnect_attempts,
        "faces_detected": len(entry.processor.latest_detections()),
    }


def _draw_detections(image, detections: list[Detection]):
    import cv2

    annotated = image.copy()
    for detection in detections:
        bbox = detection.bbox
        top_left = (int(bbox.x1), int(bbox.y1))
        bottom_right = (int(bbox.x2), int(bbox.y2))
        cv2.rectangle(annotated, top_left, bottom_right, (0, 200, 0), 2)
        label = f"{detection.kind} {detection.score:.0%}"
        label_origin = (top_left[0], max(top_left[1] - 8, 12))
        cv2.putText(
            annotated, label, label_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA
        )
    return annotated


@router.get("/cameras/{camera_id}/stream.mjpg")
async def stream_camera(camera_id: str, request: Request) -> StreamingResponse:
    entry = request.app.state.cameras.get(camera_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="camera not found")

    async def generate():
        import cv2

        interval = 1.0 / entry.config.display_fps if entry.config.display_fps > 0 else 0.0
        last_sequence = -1
        while True:
            if await request.is_disconnected():
                break
            frame = entry.worker.latest_frame()
            if frame is not None and frame.sequence != last_sequence:
                last_sequence = frame.sequence
                # Detections come from the last analyzed frame, not
                # necessarily this exact one (analysis runs slower than
                # display) - the overlay always shows the most recent
                # known result until a newer one is ready.
                annotated = _draw_detections(frame.image, entry.processor.latest_detections())
                ok, buffer = cv2.imencode(".jpg", annotated)
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
