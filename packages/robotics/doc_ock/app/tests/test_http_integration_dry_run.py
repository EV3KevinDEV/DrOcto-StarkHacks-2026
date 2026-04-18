from __future__ import annotations

import time

from fastapi.testclient import TestClient

from doc_ock.api import create_app


def test_http_integration_dry_run(runtime) -> None:
    app = create_app(runtime)
    client = TestClient(app)

    start = client.post(
        "/session/start",
        json={
            "task": "pick up cube",
            "model_repo_id": "org/model",
            "max_steps": 400,
        },
    )
    assert start.status_code == 200
    assert start.json()["state"] in {"starting", "running"}

    second_start = client.post(
        "/session/start",
        json={
            "task": "another task",
            "model_repo_id": "org/model",
        },
    )
    assert second_start.status_code == 409

    running_seen = False
    for _ in range(100):
        status = client.get("/session/status")
        assert status.status_code == 200
        state = status.json()["state"]
        if state == "running":
            running_seen = True
            break
        if state == "error":
            raise AssertionError(status.json())
        time.sleep(0.01)

    assert running_seen

    voice_on = client.post("/voice-mode", json={"enabled": True})
    assert voice_on.status_code == 200
    assert voice_on.json()["enabled"] is True

    voice_off = client.post("/voice-mode", json={"enabled": False})
    assert voice_off.status_code == 200
    assert voice_off.json()["enabled"] is False

    stop = client.post("/session/stop")
    assert stop.status_code == 200
    assert stop.json()["state"] in {"stopping", "stopped"}


def test_http_rejects_session_dry_run_override(runtime) -> None:
    app = create_app(runtime)
    client = TestClient(app)

    response = client.post(
        "/session/start",
        json={
            "task": "task",
            "model_repo_id": "org/model",
            "dry_run": False,
        },
    )

    assert response.status_code == 400
    assert "fixed at server startup" in response.json()["detail"]
