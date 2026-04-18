from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, Mapping, Protocol

import numpy as np

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
    def __init__(self, camera_paths: Mapping[str, str], state_keys: list[str] | None = None):
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

    def __init__(self, robot_port: str, camera_paths: Mapping[str, str], state_keys: list[str] | None = None):
        self.robot_port = robot_port
        self.camera_paths = dict(camera_paths)
        self.state_keys = list(state_keys) if state_keys else list(_DEFAULT_STATE_KEYS)
        self._robot: Any = None
        self._captures: dict[str, Any] = {}
        self._last_action: np.ndarray | None = None
        self._started = False

    def start(self) -> None:
        self._robot = self._try_create_lerobot_robot()
        self._captures = self._start_cameras()
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

        for method_name in ("send_action", "apply_action", "step"):
            method = getattr(self._robot, method_name, None)
            if not callable(method):
                continue
            try:
                method(self._last_action)
                return
            except TypeError:
                # Some APIs use keyword arguments; retry with a generic kwarg.
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
        for name, path in self.camera_paths.items():
            capture = cv2.VideoCapture(path)
            captures[name] = capture
        return captures

    def _read_images(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        for name, capture in self._captures.items():
            ok, frame = capture.read()
            if not ok or frame is None:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
            images[name] = frame
        return images

    def _read_state_mapping(self) -> dict[str, float]:
        if self._robot is not None:
            for method_name in ("get_state", "read_state", "state"):
                method = getattr(self._robot, method_name, None)
                if callable(method):
                    try:
                        state = method()
                        if isinstance(state, Mapping):
                            return {
                                key: float(state.get(key, 0.0))
                                for key in self.state_keys
                            }
                    except Exception:
                        logger.exception("Failed to read robot state via %s", method_name)

        return {key: 0.0 for key in self.state_keys}

    def _try_create_lerobot_robot(self) -> Any:
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

    def _invoke_factory(self, func: Any) -> Any:
        camera_cfg = {
            name: {
                "type": "opencv",
                "index_or_path": path,
                "width": 640,
                "height": 480,
                "fps": 30,
            }
            for name, path in self.camera_paths.items()
        }
        candidate_kwargs = {
            "robot_type": "so101_follower",
            "type": "so101_follower",
            "port": self.robot_port,
            "id": "doc_ock_follower",
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
                    "type": "so101_follower",
                    "port": self.robot_port,
                    "id": "doc_ock_follower",
                    "cameras": camera_cfg,
                }
                try:
                    return func(**{only_name: payload})
                except Exception as exc:
                    logger.warning("LeRobot single-arg factory call failed: %s", exc)

        try:
            return func("so101_follower", self.robot_port)
        except Exception:
            return None
