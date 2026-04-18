from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from doc_ock.models import SessionRequest
from doc_ock.runtime import DocOckRuntime, SessionConflictError

_UI_DIR = Path(__file__).parent / "ui"


class SessionStartBody(BaseModel):
    task: str = Field(min_length=1)
    model_repo_id: str = Field(min_length=1)
    max_steps: int | None = Field(default=None, gt=0)
    max_duration_s: float | None = Field(default=None, gt=0)
    dry_run: bool | None = None


class VoiceModeBody(BaseModel):
    enabled: bool


def create_app(runtime: DocOckRuntime) -> FastAPI:
    app = FastAPI(title="Doc Ock API", version="0.1.0")

    @app.post("/session/start")
    def start_session(body: SessionStartBody) -> dict:
        if body.dry_run is not None and body.dry_run != runtime.runtime_config.dry_run:
            raise HTTPException(
                status_code=400,
                detail=(
                    "dry_run is fixed at server startup. "
                    f"Server dry_run={runtime.runtime_config.dry_run}."
                ),
            )

        request = SessionRequest(
            task=body.task,
            model_repo_id=body.model_repo_id,
            max_steps=body.max_steps,
            max_duration_s=body.max_duration_s,
        )

        try:
            status = runtime.start_session(request=request, background=True)
        except SessionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return status.to_dict()

    @app.post("/session/stop")
    def stop_session() -> dict:
        status = runtime.stop_session()
        return status.to_dict()

    @app.get("/session/status")
    def session_status() -> dict:
        return runtime.get_session_status().to_dict()

    @app.get("/voice-mode")
    def get_voice_mode() -> dict:
        return {"enabled": runtime.get_voice_mode()}

    @app.post("/voice-mode")
    def set_voice_mode(body: VoiceModeBody) -> dict:
        enabled = runtime.set_voice_mode(body.enabled)
        status = runtime.get_session_status()
        payload = {"enabled": enabled}
        payload.update({"voice_mode_enabled": status.voice_mode_enabled, "state": status.state.value})
        return payload

    @app.get("/health")
    def health() -> dict:
        return runtime.get_health_status().to_dict()

    if _UI_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")

        @app.get("/", include_in_schema=False)
        def _root_redirect() -> RedirectResponse:
            return RedirectResponse(url="/ui/")

    return app
