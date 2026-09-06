"""FaceEngine: a VisionEngine specialized for faces.

process() always returns Detection(kind="face", ...). From Fase 2 it only
fills bbox/score; Fase 3/4 add the embedding once recognition needs it.
"""

from __future__ import annotations

from vip.vision.base import VisionEngine


class FaceEngine(VisionEngine):
    name = "face"
