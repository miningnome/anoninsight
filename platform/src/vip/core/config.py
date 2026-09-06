"""Configuration for the Video Intelligence Platform.

Fase 1 only needs server host/port and a list of cameras. Later phases add
sections for detection, GPU/CPU policy, thresholds, etc. (see
docs/video-intelligence-platform/ARCHITECTURE.md, section 6/7).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.toml"


class CameraConfig(BaseModel):
    id: str
    name: str
    source_type: Literal["rtsp", "usb", "file"] = "rtsp"
    url: str
    open_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 5.0
    reconnect_delay_seconds: float = 1.0
    reconnect_max_delay_seconds: float = 30.0
    display_fps: float = 15.0
    analysis_fps: float = 5.0
    loop: bool = True  # only relevant for source_type == "file"


class VisionConfig(BaseModel):
    """Shared vision engine configuration.

    One engine instance is loaded and shared across every camera (see
    vip.vision.concurrency.SerializedVisionEngine) rather than one per
    camera, so memory/VRAM usage does not grow linearly with the number of
    cameras. Default is "mock" so the platform starts cleanly without any
    model weights or network access; switch to "insightface" once a model
    pack is available locally.
    """

    engine: Literal["insightface", "mock"] = "mock"
    model_pack: str = "buffalo_l"
    ctx_id: int = -1  # -1 = CPU, >=0 = CUDA device index
    min_score: float = 0.5


class StorageConfig(BaseModel):
    database_path: str = "data/vip.db"
    store_face_crops: bool = True
    max_upload_bytes: int = 10 * 1024 * 1024


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8088


class Settings(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    cameras: list[CameraConfig] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        config_path = path or Path(os.environ.get("VIP_CONFIG_FILE", DEFAULT_CONFIG_PATH))
        if not config_path.exists():
            return cls()
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        return cls.model_validate(raw)
