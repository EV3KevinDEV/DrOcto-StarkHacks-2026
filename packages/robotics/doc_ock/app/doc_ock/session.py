from __future__ import annotations

import threading
from typing import Callable, Optional, Protocol

from doc_ock.models import SessionRequest, SessionState, SessionStatus, utc_now_iso
from doc_ock.voice_mode import VoiceModeController


class SessionConflictError(RuntimeError):
    """Raised when the caller attempts to start a second active session."""


class SessionRunner(Protocol):
    @property
    def model_loaded(self) -> bool:
        ...

    @property
    def active_task(self) -> Optional[str]:
        ...

    def run(self, stop_event: threading.Event, on_step: Callable[[], None]) -> None:
        ...

    def close(self) -> None:
        ...


AudioStatusProvider = Callable[[], dict]


class SessionManager:
    def __init__(
        self,
        voice_mode_controller: VoiceModeController,
        dry_run: bool,
        audio_status_provider: Optional[AudioStatusProvider] = None,
    ):
        self._voice_mode_controller = voice_mode_controller
        self._dry_run = dry_run
        self._audio_status_provider = audio_status_provider

        self._lock = threading.RLock()
        self._state: SessionState = SessionState.IDLE
        self._task: Optional[str] = None
        self._model_repo_id: Optional[str] = None
        self._step_count = 0
        self._started_at: Optional[str] = None
        self._last_error: Optional[str] = None

        self._runner: Optional[SessionRunner] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._model_loaded = False

    @property
    def model_loaded(self) -> bool:
        with self._lock:
            return self._model_loaded

    def ensure_startable(self) -> None:
        with self._lock:
            if self._state in (SessionState.STARTING, SessionState.RUNNING, SessionState.STOPPING):
                raise SessionConflictError("A session is already active")

    def start_session(
        self,
        request: SessionRequest,
        runner: SessionRunner,
        background: bool,
        initial_model_loaded: bool = False,
    ) -> SessionStatus:
        with self._lock:
            if self._state in (SessionState.STARTING, SessionState.RUNNING, SessionState.STOPPING):
                raise SessionConflictError("A session is already active")

            self._state = SessionState.STARTING
            self._task = request.task
            self._model_repo_id = request.model_repo_id
            self._step_count = 0
            self._started_at = utc_now_iso()
            self._last_error = None

            self._runner = runner
            self._stop_event = threading.Event()
            self._model_loaded = bool(initial_model_loaded)

            if background:
                self._thread = threading.Thread(target=self._run_runner, name="doc-ock-session", daemon=True)
                self._thread.start()
                return self._status_locked()

        self._run_runner()
        return self.get_status()

    def stop_session(self, wait_timeout_s: float = 5.0) -> SessionStatus:
        with self._lock:
            if self._state in (SessionState.IDLE, SessionState.STOPPED, SessionState.ERROR):
                return self._status_locked()

            self._state = SessionState.STOPPING
            thread = self._thread
            stop_event = self._stop_event

        if stop_event is not None:
            stop_event.set()

        if thread is not None and thread.is_alive():
            thread.join(wait_timeout_s)

        with self._lock:
            if self._state == SessionState.STOPPING:
                self._state = SessionState.STOPPED
            return self._status_locked()

    def get_status(self) -> SessionStatus:
        with self._lock:
            return self._status_locked()

    def _run_runner(self) -> None:
        with self._lock:
            runner = self._runner
            stop_event = self._stop_event
            self._state = SessionState.RUNNING

        if runner is None or stop_event is None:
            with self._lock:
                self._state = SessionState.ERROR
                self._last_error = "Session runner was not initialized"
            return

        try:
            runner.run(stop_event, self._increment_step)
            with self._lock:
                if self._state != SessionState.ERROR:
                    self._state = SessionState.STOPPED
        except Exception as exc:
            with self._lock:
                self._state = SessionState.ERROR
                self._last_error = str(exc)
        finally:
            try:
                runner.close()
            finally:
                with self._lock:
                    self._runner = None
                    self._thread = None
                    self._stop_event = None
                    self._model_loaded = False

    def _increment_step(self) -> None:
        with self._lock:
            self._step_count += 1

    def _status_locked(self) -> SessionStatus:
        audio = self._audio_status_provider() if self._audio_status_provider else {}
        active_task = self._task
        if self._runner is not None:
            try:
                runner_task = getattr(self._runner, "active_task", None)
                if runner_task:
                    active_task = runner_task
            except Exception:
                pass
        return SessionStatus(
            state=self._state,
            task=active_task,
            model_repo_id=self._model_repo_id,
            step_count=self._step_count,
            started_at=self._started_at,
            last_error=self._last_error,
            voice_mode_enabled=self._voice_mode_controller.get_enabled(),
            dry_run=self._dry_run,
            latest_partial_transcript=audio.get("latest_partial", ""),
            latest_committed_transcript=audio.get("latest_committed"),
            audio_running=audio.get("running", False),
            audio_error=audio.get("error"),
        )
