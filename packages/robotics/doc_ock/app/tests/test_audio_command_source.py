from __future__ import annotations

import threading
import time
from typing import Optional

import pytest

from doc_ock.commands import TaskCommand
from doc_ock.models import RuntimeConfig, SessionRequest, SessionState
from doc_ock.runtime import DocOckRuntime


class FakeAudioSource:
    """Duck-typed CommandSource that also mimics ElevenLabsCommandSource's
    lifecycle + status fields that DocOckRuntime inspects for UI/health.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._queue: list[TaskCommand] = []
        self._running = False
        self.latest_partial = ""
        self.latest_committed: Optional[str] = None
        self.latest_error: Optional[str] = None
        self.start_calls = 0
        self.stop_calls = 0
        self.manual_commit_calls = 0

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> None:
        with self._lock:
            self._running = True
            self.start_calls += 1

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self.stop_calls += 1

    def next_command(self, timeout_s: Optional[float] = None) -> Optional[TaskCommand]:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    def simulate_partial(self, text: str) -> None:
        with self._lock:
            self.latest_partial = text

    def simulate_commit(self, text: str) -> None:
        with self._lock:
            self.latest_partial = ""
            self.latest_committed = text
            self._queue.append(TaskCommand(task_text=text))

    def request_manual_commit(self) -> None:
        with self._lock:
            self.manual_commit_calls += 1


@pytest.fixture
def audio_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        robot_port="/dev/ttyACM_follower",
        cameras={"top": "/dev/webcam_top", "front": "/dev/webcam_front"},
        dry_run=True,
        step_delay_s=0.001,
    )


def test_voice_mode_toggle_starts_and_stops_audio_source(audio_runtime_config: RuntimeConfig) -> None:
    audio = FakeAudioSource()
    runtime = DocOckRuntime(audio_runtime_config, audio_source=audio)

    assert audio.start_calls == 0
    runtime.set_voice_mode(True)
    assert audio.is_running is True
    assert audio.start_calls == 1

    runtime.set_voice_mode(False)
    assert audio.is_running is False
    assert audio.stop_calls == 1

    enabled = runtime.toggle_voice_mode()
    assert enabled is True
    assert audio.start_calls == 2


def test_committed_transcript_becomes_active_task(audio_runtime_config: RuntimeConfig) -> None:
    audio = FakeAudioSource()
    runtime = DocOckRuntime(audio_runtime_config, audio_source=audio)
    runtime.set_voice_mode(True)

    request = SessionRequest(
        task="initial task",
        model_repo_id="org/smolvla-dry",
        max_duration_s=1.0,
    )
    runtime.start_session(request=request, background=True)

    audio.simulate_commit("pick up the red block")

    deadline = time.monotonic() + 2.0
    observed_task: Optional[str] = None
    while time.monotonic() < deadline:
        status = runtime.get_session_status()
        if status.task == "pick up the red block":
            observed_task = status.task
            break
        time.sleep(0.02)

    try:
        assert observed_task == "pick up the red block", (
            f"worker never picked up the committed transcript; last status={runtime.get_session_status().to_dict()}"
        )
    finally:
        runtime.stop_session()
        runtime.shutdown()


def test_session_and_health_status_surface_transcript_fields(audio_runtime_config: RuntimeConfig) -> None:
    audio = FakeAudioSource()
    runtime = DocOckRuntime(audio_runtime_config, audio_source=audio)

    audio.simulate_partial("pick up the")
    runtime.set_voice_mode(True)

    session = runtime.get_session_status().to_dict()
    assert session["latest_partial_transcript"] == "pick up the"
    assert session["audio_running"] is True
    assert session["voice_mode_enabled"] is True

    audio.simulate_commit("pick up the red block")

    health = runtime.get_health_status().to_dict()
    assert health["audio_enabled"] is True
    assert health["audio_running"] is True
    assert health["latest_committed_transcript"] == "pick up the red block"
    assert health["latest_partial_transcript"] == ""

    runtime.set_voice_mode(False)
    assert runtime.get_health_status().to_dict()["audio_running"] is False


def test_session_reaches_stopped_state(audio_runtime_config: RuntimeConfig) -> None:
    audio = FakeAudioSource()
    runtime = DocOckRuntime(audio_runtime_config, audio_source=audio)

    request = SessionRequest(
        task="warmup",
        model_repo_id="org/smolvla-dry",
        max_steps=5,
    )
    status = runtime.start_session(request=request, background=False)
    assert status.state == SessionState.STOPPED
    assert status.step_count == 5
    runtime.shutdown()


def test_manual_audio_commit_is_forwarded_to_audio_source(audio_runtime_config: RuntimeConfig) -> None:
    audio = FakeAudioSource()
    runtime = DocOckRuntime(audio_runtime_config, audio_source=audio)

    runtime.set_voice_mode(True)
    payload = runtime.request_audio_commit()

    assert audio.manual_commit_calls == 1
    assert payload["running"] is True

    runtime.shutdown()
