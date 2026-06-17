from __future__ import annotations

import threading
import time

from doc_ock.models import SessionRequest, SessionState
from doc_ock.session import SessionManager
from doc_ock.voice_mode import VoiceModeController


class StoppableRunner:
    @property
    def model_loaded(self) -> bool:
        return True

    def run(self, stop_event: threading.Event, on_step) -> None:
        while not stop_event.is_set():
            on_step()
            time.sleep(0.005)

    def close(self) -> None:
        return None


class ErrorRunner:
    @property
    def model_loaded(self) -> bool:
        return True

    def run(self, stop_event: threading.Event, on_step) -> None:
        del stop_event
        on_step()
        raise RuntimeError("intentional test failure")

    def close(self) -> None:
        return None


def test_graceful_stop() -> None:
    manager = SessionManager(VoiceModeController(), dry_run=True)
    request = SessionRequest(task="task", model_repo_id="org/model")

    manager.start_session(request=request, runner=StoppableRunner(), background=True, initial_model_loaded=True)
    status = manager.stop_session()

    assert status.state in (SessionState.STOPPING, SessionState.STOPPED)


def test_error_propagation() -> None:
    manager = SessionManager(VoiceModeController(), dry_run=True)
    request = SessionRequest(task="task", model_repo_id="org/model")

    status = manager.start_session(request=request, runner=ErrorRunner(), background=False, initial_model_loaded=True)

    assert status.state == SessionState.ERROR
    assert "intentional test failure" in (status.last_error or "")
