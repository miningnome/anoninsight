"""FaceEngine backed by the `insightface` package (SCRFD detector for Fase
2; ArcFace embeddings are already produced by the same model pack and get
used starting Fase 3/4).

The `insightface` import is deliberately deferred to warmup(), not module
import time: constructing this class (e.g. from a factory, or in a test)
must not require the package - or its model weights - to be installed.
"""

from __future__ import annotations

from vip.cameras.base import Frame
from vip.vision.base import BoundingBox, Detection

from .base import FaceEngine


class InsightFaceEngine(FaceEngine):
    def __init__(
        self,
        *,
        model_pack: str = "buffalo_l",
        ctx_id: int = -1,  # -1 = CPU, >=0 = CUDA device index
        detection_size: tuple[int, int] = (640, 640),
        min_score: float = 0.5,
    ) -> None:
        self._model_pack = model_pack
        self._ctx_id = ctx_id
        self._detection_size = detection_size
        self._min_score = min_score
        self._app = None

    def warmup(self) -> None:
        if self._app is not None:
            return
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name=self._model_pack)
        app.prepare(ctx_id=self._ctx_id, det_size=self._detection_size)
        self._app = app

    def process(self, frame: Frame) -> list[Detection]:
        if self._app is None:
            self.warmup()

        detections: list[Detection] = []
        for face in self._app.get(frame.image):
            score = float(face.det_score)
            if score < self._min_score:
                continue
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            landmarks = face.kps.tolist() if getattr(face, "kps", None) is not None else None
            detections.append(
                Detection(
                    kind="face",
                    bbox=BoundingBox(x1, y1, x2, y2),
                    score=score,
                    embedding=getattr(face, "normed_embedding", None),
                    attributes={"landmarks": landmarks},
                )
            )
        return detections

    def close(self) -> None:
        self._app = None
