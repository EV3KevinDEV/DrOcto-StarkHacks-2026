from __future__ import annotations

import time
import threading
from typing import Callable, Optional

import logging

from doc_ock.adapters.policy import SmolVlaPolicyAdapter
from doc_ock.adapters.robot import DryRunRobotAdapter, RealRobotAdapter, RobotAdapter
from doc_ock.commands import CommandSource, TaskCommand
from doc_ock.models import HealthStatus, RuntimeConfig, SessionRequest, SessionStatus
from doc_ock.observation import ObservationBridge
from doc_ock.session import SessionConflictError, SessionManager, SessionRunner
from doc_ock.voice_mode import VoiceModeController

logger = logging.getLogger(__name__)

PolicyFactory = Callable[[str, RuntimeConfig], SmolVlaPolicyAdapter]
RobotFactory = Callable[[RuntimeConfig], RobotAdapter]


class SessionWorker(SessionRunner):
    def __init__(
        self,
        runtime_config: RuntimeConfig,
        request: SessionRequest,
        robot_adapter: RobotAdapter,
        policy_adapter: SmolVlaPolicyAdapter,
        command_source: Optional[CommandSource] = None,
    ):
        self._runtime_config = runtime_config
        self._request = request
        self._robot = robot_adapter
        self._policy = policy_adapter
        self._command_source = command_source

        self._bridge: ObservationBridge | None = None
        self._command_lock = threading.Lock()
        self._current_command = TaskCommand(task_text=request.task)

    @property
    def model_loaded(self) -> bool:
        return self._policy.model_loaded

    @property
    def active_task(self) -> str:
        with self._command_lock:
            return self._current_command.task_text

    def preflight(self) -> None:
        self._policy.load()
        self._policy.validate_camera_names(self._runtime_config.cameras.keys())
        self._bridge = ObservationBridge(
            self._policy.expected_image_feature_keys,
            camera_aliases=self._policy.effective_camera_aliases,
        )
        self._bridge.validate_camera_names(self._runtime_config.cameras.keys())

    def run(self, stop_event: threading.Event, on_step: Callable[[], None]) -> None:
        if self._bridge is None:
            self.preflight()

        self._robot.start()
        started_monotonic = time.monotonic()
        step_count = 0

        while not stop_event.is_set():
            if self._request.max_steps is not None and step_count >= self._request.max_steps:
                break

            if self._request.max_duration_s is not None:
                elapsed = time.monotonic() - started_monotonic
                if elapsed >= self._request.max_duration_s:
                    break

            self._update_task_command()
            with self._command_lock:
                current_text = self._current_command.task_text
            observation = self._robot.get_observation()
            prepared = self._bridge.to_policy_observation(observation, current_text)
            action = self._policy.infer(prepared)
            self._robot.send_action(action)

            step_count += 1
            on_step()

            if self._runtime_config.step_delay_s > 0:
                time.sleep(self._runtime_config.step_delay_s)

    def close(self) -> None:
        try:
            self._robot.stop()
        finally:
            self._policy.unload()

    def _update_task_command(self) -> None:
        if self._command_source is None:
            return

        next_command = self._command_source.next_command(timeout_s=0.0)
        if next_command is None:
            return

        with self._command_lock:
            self._current_command = next_command


class DocOckRuntime:
    def __init__(
        self,
        runtime_config: RuntimeConfig,
        robot_factory: RobotFactory | None = None,
        policy_factory: PolicyFactory | None = None,
        command_source: CommandSource | None = None,
        audio_source: "object | None" = None,
    ):
        self._runtime_config = runtime_config
        self._injected_command_source = command_source
        self._audio_source = audio_source

        self._voice_mode = VoiceModeController(initial_enabled=False)
        self._session_manager = SessionManager(
            self._voice_mode,
            dry_run=runtime_config.dry_run,
            audio_status_provider=self._audio_status,
        )

        self._robot_factory = robot_factory or _default_robot_factory
        self._policy_factory = policy_factory or _default_policy_factory

    @property
    def runtime_config(self) -> RuntimeConfig:
        return self._runtime_config

    def start_session(self, request: SessionRequest, background: bool) -> SessionStatus:
        if not request.task.strip():
            raise ValueError("task is required")
        if not request.model_repo_id.strip():
            raise ValueError("model_repo_id is required")
        if request.max_steps is not None and request.max_steps <= 0:
            raise ValueError("max_steps must be positive when provided")
        if request.max_duration_s is not None and request.max_duration_s <= 0:
            raise ValueError("max_duration_s must be positive when provided")

        self._session_manager.ensure_startable()

        robot = self._robot_factory(self._runtime_config)
        policy = self._policy_factory(
            request.model_repo_id,
            self._runtime_config,
        )
        worker = SessionWorker(
            runtime_config=self._runtime_config,
            request=request,
            robot_adapter=robot,
            policy_adapter=policy,
            command_source=self._effective_command_source(),
        )

        worker.preflight()
        return self._session_manager.start_session(
            request=request,
            runner=worker,
            background=background,
            initial_model_loaded=worker.model_loaded,
        )

    def stop_session(self) -> SessionStatus:
        return self._session_manager.stop_session()

    def get_session_status(self) -> SessionStatus:
        return self._session_manager.get_status()

    def get_health_status(self) -> HealthStatus:
        status = self._session_manager.get_status()
        audio = self._audio_status()
        return HealthStatus(
            state=status.state,
            voice_mode_enabled=self._voice_mode.get_enabled(),
            configured_cameras=self._runtime_config.configured_cameras(),
            camera_profiles=self._runtime_config.camera_profiles(),
            camera_aliases=dict(self._runtime_config.camera_aliases),
            dry_run=self._runtime_config.dry_run,
            model_loaded=self._session_manager.model_loaded,
            robot_type=self._runtime_config.robot_type,
            robot_id=self._runtime_config.robot_id,
            robot_port=self._runtime_config.robot_port,
            teleop_type=self._runtime_config.teleop_type,
            teleop_port=self._runtime_config.teleop_port,
            teleop_id=self._runtime_config.teleop_id,
            audio_enabled=self._audio_source is not None,
            audio_running=audio.get("running", False),
            audio_error=audio.get("error"),
            latest_partial_transcript=audio.get("latest_partial", ""),
            latest_committed_transcript=audio.get("latest_committed"),
        )

    def get_voice_mode(self) -> bool:
        return self._voice_mode.get_enabled()

    def set_voice_mode(self, enabled: bool) -> bool:
        new_value = self._voice_mode.set_enabled(enabled)
        self._apply_voice_mode(new_value)
        return new_value

    def toggle_voice_mode(self) -> bool:
        new_value = self._voice_mode.toggle()
        self._apply_voice_mode(new_value)
        return new_value

    def request_audio_commit(self) -> dict:
        audio = self._audio_source
        if audio is None:
            raise RuntimeError("Audio input is not configured")

        request_commit = getattr(audio, "request_manual_commit", None)
        if not callable(request_commit):
            raise RuntimeError("Audio source does not support manual commit")

        if not self._voice_mode.get_enabled():
            raise RuntimeError("Voice mode is off")

        request_commit()
        return self._audio_status()

    def shutdown(self) -> None:
        audio = self._audio_source
        if audio is not None:
            try:
                audio.stop()
            except Exception:
                logger.exception("Error stopping audio source on shutdown")

    def _apply_voice_mode(self, enabled: bool) -> None:
        audio = self._audio_source
        if audio is None:
            return
        try:
            if enabled:
                audio.start()
            else:
                audio.stop()
        except Exception:
            logger.exception("Voice-mode audio %s failed", "start" if enabled else "stop")

    def _effective_command_source(self) -> CommandSource | None:
        if self._injected_command_source is not None:
            return self._injected_command_source
        if self._audio_source is not None:
            return self._audio_source  # duck-typed: provides next_command
        return None

    def _audio_status(self) -> dict:
        audio = self._audio_source
        if audio is None:
            return {}
        return {
            "running": bool(getattr(audio, "is_running", False)),
            "latest_partial": getattr(audio, "latest_partial", "") or "",
            "latest_committed": getattr(audio, "latest_committed", None),
            "error": getattr(audio, "latest_error", None),
        }


def _default_robot_factory(runtime_config: RuntimeConfig) -> RobotAdapter:
    if runtime_config.dry_run:
        return DryRunRobotAdapter(runtime_config.cameras)
    return RealRobotAdapter(runtime_config=runtime_config)


def _default_policy_factory(model_repo_id: str, runtime_config: RuntimeConfig) -> SmolVlaPolicyAdapter:
    return SmolVlaPolicyAdapter(
        model_repo_id=model_repo_id,
        camera_names=list(runtime_config.cameras.keys()),
        dry_run=runtime_config.dry_run,
        camera_aliases=runtime_config.camera_aliases,
        robot_type=runtime_config.robot_type,
    )


__all__ = [
    "DocOckRuntime",
    "SessionConflictError",
]
