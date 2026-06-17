from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


CameraSource = str | int


@dataclass(frozen=True)
class RuntimeCameraConfig:
    index_or_path: CameraSource
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str | None = "YUYV"
    type: str = "opencv"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "index_or_path": self.index_or_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "fourcc": self.fourcc,
        }


class SessionState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass(frozen=True)
class RuntimeConfig:
    robot_port: str
    cameras: Dict[str, RuntimeCameraConfig | CameraSource]
    dry_run: bool
    robot_type: str = "so101_follower"
    robot_id: str = "follower1"
    teleop_type: str = "so101_leader"
    teleop_port: str | None = None
    teleop_id: str = "leader1"
    camera_aliases: Dict[str, str] = field(default_factory=dict)
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    step_delay_s: float = 0.01

    def __post_init__(self) -> None:
        normalized: Dict[str, RuntimeCameraConfig] = {}
        for name, value in self.cameras.items():
            if isinstance(value, RuntimeCameraConfig):
                normalized[name] = value
            else:
                normalized[name] = RuntimeCameraConfig(index_or_path=value)

        object.__setattr__(self, "cameras", normalized)

        missing_alias_targets = sorted(set(self.camera_aliases.values()) - set(normalized))
        if missing_alias_targets:
            raise ValueError(
                "camera_aliases must point at configured camera names. "
                f"Missing camera names: {missing_alias_targets}"
            )

    def configured_cameras(self) -> Dict[str, CameraSource]:
        return {
            name: camera.index_or_path
            for name, camera in self.cameras.items()
        }

    def camera_profiles(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: camera.to_dict()
            for name, camera in self.cameras.items()
        }


@dataclass(frozen=True)
class SessionRequest:
    task: str
    model_repo_id: str
    max_steps: Optional[int] = None
    max_duration_s: Optional[float] = None


@dataclass
class SessionStatus:
    state: SessionState = SessionState.IDLE
    task: Optional[str] = None
    model_repo_id: Optional[str] = None
    step_count: int = 0
    started_at: Optional[str] = None
    last_error: Optional[str] = None
    voice_mode_enabled: bool = False
    dry_run: bool = False
    latest_partial_transcript: str = ""
    latest_committed_transcript: Optional[str] = None
    audio_running: bool = False
    audio_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "task": self.task,
            "model_repo_id": self.model_repo_id,
            "step_count": self.step_count,
            "started_at": self.started_at,
            "last_error": self.last_error,
            "voice_mode_enabled": self.voice_mode_enabled,
            "dry_run": self.dry_run,
            "latest_partial_transcript": self.latest_partial_transcript,
            "latest_committed_transcript": self.latest_committed_transcript,
            "audio_running": self.audio_running,
            "audio_error": self.audio_error,
        }


@dataclass
class HealthStatus:
    state: SessionState
    voice_mode_enabled: bool
    configured_cameras: Dict[str, CameraSource]
    camera_profiles: Dict[str, Dict[str, Any]]
    camera_aliases: Dict[str, str]
    dry_run: bool
    model_loaded: bool
    robot_type: str
    robot_id: str
    robot_port: str
    teleop_type: str
    teleop_port: str | None
    teleop_id: str
    audio_enabled: bool = False
    audio_running: bool = False
    audio_error: Optional[str] = None
    latest_partial_transcript: str = ""
    latest_committed_transcript: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "voice_mode_enabled": self.voice_mode_enabled,
            "configured_cameras": self.configured_cameras,
            "camera_profiles": self.camera_profiles,
            "camera_aliases": self.camera_aliases,
            "dry_run": self.dry_run,
            "model_loaded": self.model_loaded,
            "robot_type": self.robot_type,
            "robot_id": self.robot_id,
            "robot_port": self.robot_port,
            "teleop_type": self.teleop_type,
            "teleop_port": self.teleop_port,
            "teleop_id": self.teleop_id,
            "audio_enabled": self.audio_enabled,
            "audio_running": self.audio_running,
            "audio_error": self.audio_error,
            "latest_partial_transcript": self.latest_partial_transcript,
            "latest_committed_transcript": self.latest_committed_transcript,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
