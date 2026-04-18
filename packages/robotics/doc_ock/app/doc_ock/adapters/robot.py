from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from doc_ock.models import RuntimeCameraConfig, RuntimeConfig

logger = logging.getLogger(__name__)

_DEFAULT_STATE_KEYS = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
]


class RobotAdapter(Protocol):
    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def get_observation(self) -> dict[str, Any]:
        ...

    def send_action(self, action: np.ndarray) -> None:
        ...


class DryRunRobotAdapter:
    def __init__(self, camera_paths: Mapping[str, Any], state_keys: list[str] | None = None):
        self.camera_paths = dict(camera_paths)
        self.state_keys = list(state_keys) if state_keys else list(_DEFAULT_STATE_KEYS)
        self._started = False
        self._step_index = 0
        self._action_log: list[np.ndarray] = []

    @property
    def action_count(self) -> int:
        return len(self._action_log)

    @property
    def action_log(self) -> list[np.ndarray]:
        return list(self._action_log)

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def get_observation(self) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("DryRunRobotAdapter.start() must be called before get_observation()")

        images = {name: self._synthetic_image(name) for name in self.camera_paths}
        state = {
            key: float(np.sin((self._step_index / 15.0) + idx))
            for idx, key in enumerate(self.state_keys)
        }
        state_vector = np.asarray([state[key] for key in self.state_keys], dtype=np.float32)
        self._step_index += 1
        return {"images": images, "state": state, "state_vector": state_vector}

    def send_action(self, action: np.ndarray) -> None:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        self._action_log.append(arr)
        logger.info("Dry-run action %d: %s", len(self._action_log), arr.tolist())

    def _synthetic_image(self, camera_name: str) -> np.ndarray:
        seed = (sum(ord(ch) for ch in camera_name) + self._step_index) % 255
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[..., 0] = (seed + 10) % 255
        image[..., 1] = (seed + 70) % 255
        image[..., 2] = (seed + 130) % 255
        return image


class RealRobotAdapter:
    """Best-effort adapter for SO-101 follower control through LeRobot runtime APIs."""

    def __init__(self, runtime_config: RuntimeConfig, state_keys: list[str] | None = None):
        self._runtime_config = runtime_config
        self.robot_port = runtime_config.robot_port
        self.robot_type = runtime_config.robot_type
        self.robot_id = runtime_config.robot_id
        self.camera_configs = dict(runtime_config.cameras)
        self.state_keys = list(state_keys) if state_keys else []
        self._action_keys = list(state_keys) if state_keys else []
        self._robot: Any = None
        self._captures: dict[str, Any] = {}
        self._last_action: np.ndarray | None = None
        self._started = False

    def start(self) -> None:
        self._robot = self._try_create_lerobot_robot()
        if self._robot is None:
            self._captures = self._start_cameras()
        else:
            self._captures = {}
            connect = getattr(self._robot, "connect", None)
            if callable(connect):
                connect()
            self._initialize_feature_keys()
        self._started = True

    def stop(self) -> None:
        for capture in self._captures.values():
            try:
                capture.release()
            except Exception:  # pragma: no cover - defensive cleanup
                pass
        self._captures = {}

        if self._robot is not None:
            for method_name in ("disconnect", "close", "stop"):
                method = getattr(self._robot, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:  # pragma: no cover - defensive cleanup
                        logger.exception("Failed while closing robot via %s", method_name)
                    break
        self._robot = None
        self._started = False

    def get_observation(self) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("RealRobotAdapter.start() must be called before get_observation()")

        observation: dict[str, Any] = {
            "images": self._read_images(),
            "state": self._read_state_mapping(),
        }
        observation["state_vector"] = np.asarray(
            [float(observation["state"][key]) for key in self.state_keys], dtype=np.float32
        )
        return observation

    def send_action(self, action: np.ndarray) -> None:
        self._last_action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._robot is None:
            logger.warning("Real mode is active but no LeRobot robot handle is available; action will be dropped")
            return

        if self._action_keys:
            if self._last_action.size != len(self._action_keys):
                raise RuntimeError(
                    "Policy action dimension does not match robot action features. "
                    f"Expected {len(self._action_keys)} values for {self._action_keys}, "
                    f"got shape {self._last_action.shape}."
                )

            action_dict = {
                key: float(self._last_action[idx])
                for idx, key in enumerate(self._action_keys)
            }
            self._robot.send_action(action_dict)
            return

        for method_name in ("send_action", "apply_action", "step"):
            method = getattr(self._robot, method_name, None)
            if not callable(method):
                continue
            try:
                method(self._last_action)
                return
            except TypeError:
                method(action=self._last_action)
                return
            except Exception as exc:
                logger.warning("Robot action dispatch via %s failed: %s", method_name, exc)
                return

        logger.warning("No supported action method found on LeRobot robot handle")

    def _start_cameras(self) -> dict[str, Any]:
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for real camera capture") from exc

        captures: dict[str, Any] = {}
        for name, config in self.camera_configs.items():
            capture = cv2.VideoCapture(self._camera_source(config))
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
            capture.set(cv2.CAP_PROP_FPS, config.fps)
            if config.fourcc:
                capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.fourcc))
            captures[name] = capture
        return captures

    def _read_images(self) -> dict[str, np.ndarray]:
        if self._robot is not None:
            raw = self._robot.get_observation()
            images: dict[str, np.ndarray] = {}
            for name, config in self.camera_configs.items():
                frame = raw.get(name)
                if frame is None:
                    frame = self._empty_frame(config)
                images[name] = np.asarray(frame)
            return images

        images: dict[str, np.ndarray] = {}
        for name, capture in self._captures.items():
            ok, frame = capture.read()
            if not ok or frame is None:
                frame = self._empty_frame(self.camera_configs[name])
            images[name] = frame
        return images

    def _read_state_mapping(self) -> dict[str, float]:
        if self._robot is not None:
            try:
                raw = self._robot.get_observation()
                keys = self.state_keys or self._action_keys or self._discover_state_keys(raw)
                return {
                    key: float(raw.get(key, 0.0))
                    for key in keys
                }
            except Exception:
                logger.exception("Failed to read robot state via get_observation")

        return {key: 0.0 for key in self.state_keys}

    def _try_create_lerobot_robot(self) -> Any:
        robot = self._try_create_current_lerobot_robot()
        if robot is not None:
            return robot

        try:
            module = importlib.import_module("lerobot.common.robot_devices.robots.factory")
        except Exception as exc:
            logger.warning("LeRobot factory import failed, continuing without direct robot handle: %s", exc)
            return None

        for func_name in ("make_robot_from_config", "make_robot", "create_robot"):
            func = getattr(module, func_name, None)
            if callable(func):
                robot = self._invoke_factory(func)
                if robot is not None:
                    logger.info("Initialized LeRobot robot with factory method %s", func_name)
                    return robot

        logger.warning("No known LeRobot factory method found; continuing with camera-only observations")
        return None

    def _try_create_current_lerobot_robot(self) -> Any:
        try:
            from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # type: ignore
            from lerobot.robots import RobotConfig as LeRobotRobotConfig  # type: ignore
            from lerobot.robots import make_robot_from_config  # type: ignore
        except Exception as exc:
            logger.warning("Current LeRobot robot import failed, falling back to legacy APIs: %s", exc)
            return None

        try:
            module = importlib.import_module(f"lerobot.robots.{self.robot_type}")
        except Exception as exc:
            logger.warning("Unable to import robot module for type '%s': %s", self.robot_type, exc)
            return None

        config_class = None
        for value in vars(module).values():
            if inspect.isclass(value) and issubclass(value, LeRobotRobotConfig) and value is not LeRobotRobotConfig:
                config_class = value
                break

        if config_class is None:
            logger.warning("Unable to locate a LeRobot config class for robot type '%s'", self.robot_type)
            return None

        camera_cfg = {
            name: OpenCVCameraConfig(
                index_or_path=self._camera_source(config),
                width=config.width,
                height=config.height,
                fps=config.fps,
                fourcc=config.fourcc,
            )
            for name, config in self.camera_configs.items()
        }
        candidate_kwargs = {
            "port": self.robot_port,
            "id": self.robot_id,
            "cameras": camera_cfg,
        }

        try:
            signature = inspect.signature(config_class)
        except (TypeError, ValueError):
            signature = None

        kwargs = {}
        if signature is not None:
            for name in signature.parameters:
                if name in candidate_kwargs:
                    kwargs[name] = candidate_kwargs[name]
        else:
            kwargs = dict(candidate_kwargs)

        try:
            robot = make_robot_from_config(config_class(**kwargs))
            logger.info("Initialized LeRobot robot using current API for type %s", self.robot_type)
            return robot
        except Exception as exc:
            logger.warning("Current LeRobot robot creation failed: %s", exc)
            return None

    def _invoke_factory(self, func: Any) -> Any:
        camera_cfg = {
            name: {
                "type": config.type,
                "index_or_path": self._camera_source(config),
                "width": config.width,
                "height": config.height,
                "fps": config.fps,
                "fourcc": config.fourcc,
            }
            for name, config in self.camera_configs.items()
        }
        candidate_kwargs = {
            "robot_type": self.robot_type,
            "type": self.robot_type,
            "port": self.robot_port,
            "id": self.robot_id,
            "cameras": camera_cfg,
        }

        try:
            signature = inspect.signature(func)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            kwargs = {}
            for name in signature.parameters:
                if name in candidate_kwargs:
                    kwargs[name] = candidate_kwargs[name]
            if kwargs:
                try:
                    return func(**kwargs)
                except Exception as exc:
                    logger.warning("LeRobot factory call with kwargs failed: %s", exc)

            if len(signature.parameters) == 1:
                only_name = next(iter(signature.parameters.keys()))
                payload = {
                    "type": self.robot_type,
                    "port": self.robot_port,
                    "id": self.robot_id,
                    "cameras": camera_cfg,
                }
                try:
                    return func(**{only_name: payload})
                except Exception as exc:
                    logger.warning("LeRobot single-arg factory call failed: %s", exc)

        try:
            return func(self.robot_type, self.robot_port)
        except Exception:
            return None

    def _initialize_feature_keys(self) -> None:
        action_keys = self._discover_action_keys()
        if action_keys:
            self._action_keys = action_keys
        if not self.state_keys:
            self.state_keys = list(self._action_keys)
        if not self.state_keys:
            self.state_keys = list(_DEFAULT_STATE_KEYS)

    def _discover_action_keys(self) -> list[str]:
        action_features = getattr(self._robot, "action_features", None)
        if isinstance(action_features, Mapping):
            return list(action_features.keys())
        return []

    def _discover_state_keys(self, raw: Mapping[str, Any] | None = None) -> list[str]:
        if raw is None and self._robot is not None:
            raw = getattr(self._robot, "observation_features", None)
        if isinstance(raw, Mapping):
            return [
                key for key in raw
                if key not in self.camera_configs and not isinstance(raw[key], tuple)
            ]
        return list(self._action_keys)

    def _camera_source(self, config: RuntimeCameraConfig) -> int | str:
        source = config.index_or_path
        if isinstance(source, Path):
            return str(source)
        return source

    def _empty_frame(self, config: RuntimeCameraConfig) -> np.ndarray:
        return np.zeros((config.height, config.width, 3), dtype=np.uint8)
