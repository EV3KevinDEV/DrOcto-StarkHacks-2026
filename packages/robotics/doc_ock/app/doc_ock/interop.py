from __future__ import annotations

from dataclasses import dataclass, field
import shlex
from typing import Any

from doc_ock.models import RuntimeCameraConfig, RuntimeConfig


def _split_top_level_commas(raw_value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0

    for char in raw_value:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        elif char == "," and depth == 0:
            segment = "".join(current).strip()
            if segment:
                parts.append(segment)
            current = []
            continue
        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _parse_scalar(raw_value: str) -> Any:
    value = raw_value.strip().strip("\"'")
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_camera_config_map(raw_value: str) -> dict[str, RuntimeCameraConfig]:
    payload = raw_value.strip()
    if not (payload.startswith("{") and payload.endswith("}")):
        raise ValueError("robot.cameras must be an object-like string wrapped in braces")

    inner = payload[1:-1].strip()
    if not inner:
        return {}

    cameras: dict[str, RuntimeCameraConfig] = {}
    for segment in _split_top_level_commas(inner):
        if ":" not in segment:
            raise ValueError(f"Invalid camera segment '{segment}'")
        camera_name, raw_config = segment.split(":", 1)
        camera_name = camera_name.strip()
        config_payload = raw_config.strip()
        if not camera_name:
            raise ValueError(f"Invalid camera segment '{segment}'. Camera name is required")
        if not (config_payload.startswith("{") and config_payload.endswith("}")):
            raise ValueError(f"Camera '{camera_name}' config must be wrapped in braces")

        config_values: dict[str, Any] = {}
        for item in _split_top_level_commas(config_payload[1:-1]):
            if ":" not in item:
                raise ValueError(f"Invalid camera config entry '{item}' for camera '{camera_name}'")
            key, raw_item_value = item.split(":", 1)
            config_values[key.strip()] = _parse_scalar(raw_item_value)

        index_or_path = config_values.get("index_or_path")
        if index_or_path is None:
            raise ValueError(f"Camera '{camera_name}' is missing index_or_path")

        cameras[camera_name] = RuntimeCameraConfig(
            type=str(config_values.get("type", "opencv")),
            index_or_path=index_or_path,
            width=int(float(config_values.get("width", 640))),
            height=int(float(config_values.get("height", 480))),
            fps=int(float(config_values.get("fps", 30))),
            fourcc=config_values.get("fourcc"),
        )
    return cameras


@dataclass(frozen=True)
class LerobotRecordTranslation:
    robot_type: str
    robot_port: str
    robot_id: str
    cameras: dict[str, RuntimeCameraConfig]
    task: str
    model_repo_id: str
    max_duration_s: float | None = None
    ignored_arguments: dict[str, str] = field(default_factory=dict)

    def session_start_body(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": self.task,
            "model_repo_id": self.model_repo_id,
        }
        if self.max_duration_s is not None:
            payload["max_duration_s"] = self.max_duration_s
        return payload

    def runtime_summary(self) -> dict[str, Any]:
        return {
            "robot_type": self.robot_type,
            "robot_port": self.robot_port,
            "robot_id": self.robot_id,
            "cameras": {name: camera.to_dict() for name, camera in self.cameras.items()},
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime_summary(),
            "session_start": self.session_start_body(),
            "ignored_arguments": dict(self.ignored_arguments),
        }


def translate_lerobot_record_command(command: str) -> LerobotRecordTranslation:
    tokens = shlex.split(command)
    if not tokens:
        raise ValueError("Command is empty")
    if tokens[0] != "lerobot-record":
        raise ValueError(f"Unsupported command '{tokens[0]}'. Expected 'lerobot-record'")

    options: dict[str, str] = {}
    for token in tokens[1:]:
        if not token.startswith("--"):
            raise ValueError(f"Unsupported positional token '{token}'")
        option = token[2:]
        if "=" in option:
            key, value = option.split("=", 1)
        else:
            key, value = option, "true"
        options[key] = value

    required = {
        "robot.type",
        "robot.port",
        "robot.id",
        "robot.cameras",
        "dataset.single_task",
        "policy.path",
    }
    missing = sorted(required - set(options))
    if missing:
        raise ValueError(f"Missing required lerobot-record options: {missing}")

    ignored = {
        key: value
        for key, value in options.items()
        if key.startswith("dataset.") and key not in {"dataset.single_task", "dataset.episode_time_s"}
    }

    max_duration_s: float | None = None
    if "dataset.episode_time_s" in options:
        max_duration_s = float(options["dataset.episode_time_s"])

    return LerobotRecordTranslation(
        robot_type=options["robot.type"],
        robot_port=options["robot.port"],
        robot_id=options["robot.id"],
        cameras=_parse_camera_config_map(options["robot.cameras"]),
        task=options["dataset.single_task"],
        model_repo_id=options["policy.path"],
        max_duration_s=max_duration_s,
        ignored_arguments=ignored,
    )


def validate_translation_against_runtime(
    translation: LerobotRecordTranslation,
    runtime_config: RuntimeConfig,
) -> None:
    mismatches: list[str] = []

    if translation.robot_type != runtime_config.robot_type:
        mismatches.append(
            f"robot.type mismatch: command={translation.robot_type}, server={runtime_config.robot_type}"
        )
    if translation.robot_port != runtime_config.robot_port:
        mismatches.append(
            f"robot.port mismatch: command={translation.robot_port}, server={runtime_config.robot_port}"
        )
    if translation.robot_id != runtime_config.robot_id:
        mismatches.append(
            f"robot.id mismatch: command={translation.robot_id}, server={runtime_config.robot_id}"
        )

    for command_camera_name, command_camera in translation.cameras.items():
        runtime_camera_name = runtime_config.camera_aliases.get(command_camera_name, command_camera_name)
        runtime_camera = runtime_config.cameras.get(runtime_camera_name)
        if runtime_camera is None:
            mismatches.append(
                "camera mismatch: "
                f"command camera '{command_camera_name}' is not configured on the server "
                f"(resolved name '{runtime_camera_name}')"
            )
            continue

        if runtime_camera.index_or_path != command_camera.index_or_path:
            mismatches.append(
                "camera source mismatch: "
                f"{command_camera_name} command={command_camera.index_or_path}, "
                f"server={runtime_camera.index_or_path}"
            )
        if runtime_camera.width != command_camera.width:
            mismatches.append(
                f"camera width mismatch for {command_camera_name}: command={command_camera.width}, "
                f"server={runtime_camera.width}"
            )
        if runtime_camera.height != command_camera.height:
            mismatches.append(
                f"camera height mismatch for {command_camera_name}: command={command_camera.height}, "
                f"server={runtime_camera.height}"
            )
        if runtime_camera.fps != command_camera.fps:
            mismatches.append(
                f"camera fps mismatch for {command_camera_name}: command={command_camera.fps}, "
                f"server={runtime_camera.fps}"
            )

    if mismatches:
        raise ValueError("lerobot-record command does not match the running server config. " + " ".join(mismatches))


__all__ = [
    "LerobotRecordTranslation",
    "translate_lerobot_record_command",
    "validate_translation_against_runtime",
]
