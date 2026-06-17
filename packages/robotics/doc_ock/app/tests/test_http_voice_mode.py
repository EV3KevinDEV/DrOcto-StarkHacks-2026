from __future__ import annotations

from typing import Optional

from fastapi.testclient import TestClient

from doc_ock.api import create_app
from doc_ock.models import RuntimeConfig
from doc_ock.runtime import DocOckRuntime


class FakeAudioSource:
    def __init__(self) -> None:
        self.is_running = False
        self.latest_partial = "grab the arduino"
        self.latest_committed: Optional[str] = None
        self.latest_error: Optional[str] = None
        self.manual_commit_calls = 0

    def start(self) -> None:
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    def request_manual_commit(self) -> None:
        self.manual_commit_calls += 1
        self.latest_committed = self.latest_partial


def test_http_voice_mode_read_write(runtime) -> None:
    app = create_app(runtime)
    client = TestClient(app)

    response = client.get("/voice-mode")
    assert response.status_code == 200
    assert response.json() == {"enabled": False}

    response = client.post("/voice-mode", json={"enabled": True})
    assert response.status_code == 200
    assert response.json()["enabled"] is True

    response = client.get("/voice-mode")
    assert response.status_code == 200
    assert response.json() == {"enabled": True}

    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()
    assert payload["voice_mode_enabled"] is True
    assert payload["configured_cameras"] == {
        "top": "/dev/webcam_top",
        "front": "/dev/webcam_front",
    }
    assert payload["dry_run"] is True


def test_http_audio_commit_returns_409_without_audio(runtime) -> None:
    app = create_app(runtime)
    client = TestClient(app)

    response = client.post("/audio/commit")

    assert response.status_code == 409
    assert response.json()["detail"] == "Audio input is not configured"


def test_http_audio_commit_succeeds_when_voice_mode_is_enabled() -> None:
    audio = FakeAudioSource()
    runtime = DocOckRuntime(
        RuntimeConfig(
            robot_port="/dev/ttyACM_follower",
            cameras={"top": "/dev/webcam_top", "front": "/dev/webcam_front"},
            dry_run=True,
        ),
        audio_source=audio,
    )
    app = create_app(runtime)
    client = TestClient(app)

    voice_on = client.post("/voice-mode", json={"enabled": True})
    assert voice_on.status_code == 200

    response = client.post("/audio/commit")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["latest_committed_transcript"] == "grab the arduino"
    assert audio.manual_commit_calls == 1
