from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Iterable

from doc_ock.api import create_app
from doc_ock.models import RuntimeConfig, SessionRequest
from doc_ock.runtime import DocOckRuntime


def parse_camera_mappings(camera_args: Iterable[str]) -> dict[str, str]:
    cameras: dict[str, str] = {}
    for item in camera_args:
        if "=" not in item:
            raise ValueError(f"Invalid --camera value '{item}'. Expected format: name=/dev/videoX")
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise ValueError(f"Invalid --camera value '{item}'. Camera name is required")
        if not path:
            raise ValueError(f"Invalid --camera value '{item}'. Camera path is required")
        if name in cameras:
            raise ValueError(f"Duplicate camera name '{name}'")
        cameras[name] = path

    if not cameras:
        raise ValueError("At least one --camera mapping is required")
    return cameras


class InteractiveVoiceController:
    def __init__(self, runtime: DocOckRuntime, enabled: bool):
        self._runtime = runtime
        self._enabled = enabled
        self._stop_event = threading.Event()
        self._hotkey_thread: threading.Thread | None = None
        self._text_thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._enabled:
            return

        initial = "ON" if self._runtime.get_voice_mode() else "OFF"
        print(f"Voice mode is {initial}. Press 'v' to toggle.")

        started = self._start_hotkey_listener()
        if started:
            return

        print(
            "Hotkey capture unavailable. "
            "Type 'v' then Enter to toggle voice mode while the CLI is running.",
            file=sys.stderr,
        )
        self._text_thread = threading.Thread(target=self._text_loop, name="doc-ock-text-toggle", daemon=True)
        self._text_thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _start_hotkey_listener(self) -> bool:
        if not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
            return False

        if os.name == "nt":
            self._hotkey_thread = threading.Thread(target=self._windows_hotkey_loop, name="doc-ock-hotkey", daemon=True)
            self._hotkey_thread.start()
            return True

        self._hotkey_thread = threading.Thread(target=self._posix_hotkey_loop, name="doc-ock-hotkey", daemon=True)
        self._hotkey_thread.start()
        return True

    def _toggle_voice_mode(self) -> None:
        enabled = self._runtime.toggle_voice_mode()
        print(f"Voice mode is now {'ON' if enabled else 'OFF'}")

    def _windows_hotkey_loop(self) -> None:
        try:
            import msvcrt  # type: ignore
        except Exception:
            return

        while not self._stop_event.is_set():
            if msvcrt.kbhit():
                char = msvcrt.getwch().lower()
                if char == "v":
                    self._toggle_voice_mode()
            time.sleep(0.05)

    def _posix_hotkey_loop(self) -> None:
        try:
            import select
            import termios
            import tty
        except Exception:
            return

        if not sys.stdin.isatty():
            return

        file_descriptor = sys.stdin.fileno()
        original_attrs = termios.tcgetattr(file_descriptor)
        tty.setcbreak(file_descriptor)
        try:
            while not self._stop_event.is_set():
                ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                if ready:
                    char = sys.stdin.read(1).lower()
                    if char == "v":
                        self._toggle_voice_mode()
        finally:
            termios.tcsetattr(file_descriptor, termios.TCSADRAIN, original_attrs)

    def _text_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                line = input().strip().lower()
            except EOFError:
                return
            if line == "v":
                self._toggle_voice_mode()


def _create_runtime(args: argparse.Namespace) -> DocOckRuntime:
    cameras = parse_camera_mappings(args.camera)
    config = RuntimeConfig(
        robot_port=args.robot_port,
        cameras=cameras,
        dry_run=args.dry_run,
        http_host=getattr(args, "host", "0.0.0.0"),
        http_port=getattr(args, "port", 8080),
    )
    return DocOckRuntime(runtime_config=config)


def _run_command(args: argparse.Namespace) -> int:
    runtime = _create_runtime(args)
    interactive = InteractiveVoiceController(runtime, enabled=not args.no_interactive)
    interactive.start()

    request = SessionRequest(
        task=args.task,
        model_repo_id=args.model_repo_id,
        max_steps=args.max_steps,
        max_duration_s=args.max_duration_s,
    )

    try:
        status = runtime.start_session(request=request, background=False)
    except Exception as exc:
        print(f"Failed to run session: {exc}", file=sys.stderr)
        return 1
    finally:
        interactive.stop()

    print(json.dumps(status.to_dict(), indent=2))
    return 0 if status.state.value != "error" else 1


def _serve_command(args: argparse.Namespace) -> int:
    runtime = _create_runtime(args)
    app = create_app(runtime)
    interactive = InteractiveVoiceController(runtime, enabled=not args.no_interactive)
    interactive.start()

    try:
        import uvicorn  # type: ignore

        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"Failed to start server: {exc}", file=sys.stderr)
        return 1
    finally:
        interactive.stop()


def _health_command(args: argparse.Namespace) -> int:
    url = args.url or f"http://{args.host}:{args.port}/health"

    try:
        with urllib.request.urlopen(url, timeout=args.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"Health request failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Doc Ock runtime CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a single local Doc Ock session")
    run_parser.add_argument("--model-repo-id", required=True)
    run_parser.add_argument("--task", required=True)
    run_parser.add_argument("--robot-port", required=True)
    run_parser.add_argument("--camera", action="append", default=[], help="name=/dev/videoX")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--max-steps", type=int, default=None)
    run_parser.add_argument("--max-duration-s", type=float, default=None)
    run_parser.add_argument("--no-interactive", action="store_true", help=argparse.SUPPRESS)
    run_parser.set_defaults(handler=_run_command)

    serve_parser = subparsers.add_parser("serve", help="Run HTTP API server")
    serve_parser.add_argument("--robot-port", required=True)
    serve_parser.add_argument("--camera", action="append", default=[], help="name=/dev/videoX")
    serve_parser.add_argument("--dry-run", action="store_true")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--log-level", default="info")
    serve_parser.add_argument("--no-interactive", action="store_true", help=argparse.SUPPRESS)
    serve_parser.set_defaults(handler=_serve_command)

    health_parser = subparsers.add_parser("health", help="Query the Doc Ock health endpoint")
    health_parser.add_argument("--url", default=None)
    health_parser.add_argument("--host", default="127.0.0.1")
    health_parser.add_argument("--port", type=int, default=8080)
    health_parser.add_argument("--timeout-s", type=float, default=3.0)
    health_parser.set_defaults(handler=_health_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
