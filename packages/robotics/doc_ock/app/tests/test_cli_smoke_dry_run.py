from __future__ import annotations

import json
import time

from doc_ock.cli import main
from doc_ock.models import RuntimeConfig, SessionRequest, SessionState
from doc_ock.runtime import DocOckRuntime


def test_cli_run_smoke_dry_run(capsys) -> None:
    exit_code = main(
        [
            "run",
            "--model-repo-id",
            "org/model",
            "--task",
            "stack blocks",
            "--robot-port",
            "/dev/ttyACM_follower",
            "--camera",
            "top=/dev/webcam_top",
            "--camera",
            "front=/dev/webcam_front",
            "--dry-run",
            "--max-steps",
            "3",
            "--no-interactive",
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)
    assert payload["step_count"] == 3
    assert payload["state"] == SessionState.STOPPED.value


def test_voice_mode_toggle_while_session_runs() -> None:
    runtime = DocOckRuntime(
        RuntimeConfig(
            robot_port="/dev/ttyACM_follower",
            cameras={"top": "/dev/webcam_top", "front": "/dev/webcam_front"},
            dry_run=True,
            step_delay_s=0.01,
        )
    )

    runtime.start_session(
        SessionRequest(
            task="stack blocks",
            model_repo_id="org/model",
            max_steps=250,
        ),
        background=True,
    )

    toggled = runtime.toggle_voice_mode()
    assert toggled is True

    for _ in range(40):
        status = runtime.get_session_status()
        if status.state in (SessionState.RUNNING, SessionState.STOPPING, SessionState.STOPPED):
            break
        time.sleep(0.01)

    assert runtime.get_voice_mode() is True

    runtime.stop_session()
    final_status = runtime.get_session_status()
    assert final_status.voice_mode_enabled is True
