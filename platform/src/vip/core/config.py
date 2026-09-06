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
    loop: bool = True  # only relevant for source_type == "file"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8088


class Settings(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    cameras: list[CameraConfig] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        config_path = path or Path(os.environ.get("VIP_CONFIG_FILE", DEFAULT_CONFIG_PATH))
        if not config_path.exists():
            return cls()
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        return cls.model_validate(raw)
