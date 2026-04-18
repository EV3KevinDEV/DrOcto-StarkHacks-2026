from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


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
    cameras: Dict[str, str]
    dry_run: bool
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    step_delay_s: float = 0.01


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
    configured_cameras: Dict[str, str]
    dry_run: bool
    model_loaded: bool
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
            "dry_run": self.dry_run,
            "model_loaded": self.model_loaded,
            "audio_enabled": self.audio_enabled,
            "audio_running": self.audio_running,
            "audio_error": self.audio_error,
            "latest_partial_transcript": self.latest_partial_transcript,
            "latest_committed_transcript": self.latest_committed_transcript,
        }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
