from __future__ import annotations

from fastapi.testclient import TestClient

from doc_ock.api import create_app


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
