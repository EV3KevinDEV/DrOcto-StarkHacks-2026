from __future__ import annotations

import threading
import time

import pytest

from doc_ock.models import SessionRequest, SessionState
from doc_ock.session import SessionConflictError, SessionManager
from doc_ock.voice_mode import VoiceModeController


class ImmediateRunner:
    @property
    def model_loaded(self) -> bool:
        return True

    def run(self, stop_event: threading.Event, on_step) -> None:
        del stop_event
        on_step()
        on_step()

    def close(self) -> None:
        return None


class BlockingRunner:
    @property
    def model_loaded(self) -> bool:
        return True

    def run(self, stop_event: threading.Event, on_step) -> None:
        while not stop_event.is_set():
            on_step()
            time.sleep(0.005)

    def close(self) -> None:
        return None


def test_session_state_transitions_sync() -> None:
    manager = SessionManager(VoiceModeController(), dry_run=True)
    request = SessionRequest(task="stack blocks", model_repo_id="org/model", max_steps=2)

    status = manager.start_session(request=request, runner=ImmediateRunner(), background=False, initial_model_loaded=True)

    assert status.state == SessionState.STOPPED
    assert status.step_count == 2
    assert status.task == "stack blocks"


def test_one_session_exclusivity() -> None:
    manager = SessionManager(VoiceModeController(), dry_run=True)
    request = SessionRequest(task="pick cube", model_repo_id="org/model")

    manager.start_session(request=request, runner=BlockingRunner(), background=True, initial_model_loaded=True)

    with pytest.raises(SessionConflictError):
        manager.start_session(request=request, runner=BlockingRunner(), background=True)

    status = manager.stop_session()
    assert status.state in (SessionState.STOPPING, SessionState.STOPPED)
