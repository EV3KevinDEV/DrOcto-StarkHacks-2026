from __future__ import annotations

from collections import deque
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from doc_ock.observation import feature_key_to_camera_name, is_placeholder_camera_name

logger = logging.getLogger(__name__)


class PolicyLoadError(RuntimeError):
    """Raised when LeRobot policy loading fails."""


class PolicyCameraValidationError(ValueError):
    """Raised when configured camera names do not match model features."""


def _resolve_local_model_source(raw_value: str) -> str | None:
    path_candidate = Path(raw_value).expanduser()
    if path_candidate.exists():
        return str(path_candidate.resolve())

    if path_candidate.is_absolute() or not raw_value.startswith((".", "~")):
        return None

    normalized_parts = [part for part in path_candidate.parts if part not in ("", ".")]
    while normalized_parts and normalized_parts[0] == "..":
        normalized_parts = normalized_parts[1:]
    if not normalized_parts:
        return None

    search_roots = [Path.cwd(), *Path.cwd().parents, Path.home()]
    seen: set[Path] = set()
    for root in search_roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = root.joinpath(*normalized_parts)
        if candidate.exists():
            return str(candidate.resolve())

    return None


def _discover_expected_image_feature_keys(policy_obj: Any) -> list[str]:
    config = getattr(policy_obj, "config", policy_obj)
    if config is None:
        return []

    candidates: list[str] = []
    for attr in ("input_features", "observation_features", "features"):
        value = getattr(config, attr, None)
        if isinstance(value, dict):
            candidates.extend(str(key) for key in value.keys())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "name" in item:
                    candidates.append(str(item["name"]))

    filtered = [
        key
        for key in candidates
        if key == "observation.image" or key.startswith("observation.images.")
    ]
    return sorted(set(filtered))


class SmolVlaPolicyAdapter:
    """Compatibility wrapper around current LeRobot pretrained policies."""

    def __init__(
        self,
        model_repo_id: str,
        camera_names: Sequence[str] | None = None,
        dry_run: bool = False,
        device: str | None = None,
        expected_image_feature_keys: Sequence[str] | None = None,
        camera_aliases: dict[str, str] | None = None,
        robot_type: str | None = None,
    ):
        self.model_repo_id = model_repo_id.strip()
        self.camera_names = list(camera_names) if camera_names else []
        self.dry_run = dry_run
        self.device = device
        self.camera_aliases = dict(camera_aliases or {})
        self.robot_type = robot_type or ""
        self._expected_image_feature_keys = list(expected_image_feature_keys) if expected_image_feature_keys else None

        self._policy: Any = None
        self._preprocessor: Any = None
        self._postprocessor: Any = None
        self._loaded = False
        self._action_buffer: deque[np.ndarray] = deque()
        self._legacy_smolvla = False
        self._inferred_camera_aliases: dict[str, str] = {}

    @property
    def resolved_model_source(self) -> str:
        raw_value = self.model_repo_id.strip()
        if not raw_value:
            return raw_value

        resolved_local_source = _resolve_local_model_source(raw_value)
        if resolved_local_source is not None:
            return resolved_local_source

        return raw_value

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    @property
    def expected_image_feature_keys(self) -> list[str]:
        if self._expected_image_feature_keys is None:
            return []
        return list(self._expected_image_feature_keys)

    @property
    def buffered_action_count(self) -> int:
        return len(self._action_buffer)

    @property
    def effective_camera_aliases(self) -> dict[str, str]:
        aliases = dict(self._inferred_camera_aliases)
        aliases.update(self.camera_aliases)
        return aliases

    def load(self) -> None:
        if self._loaded:
            return

        if not self.model_repo_id:
            raise PolicyLoadError("model_repo_id is required")

        model_source = self.resolved_model_source
        self._inferred_camera_aliases = self._discover_local_camera_aliases(model_source)

        if self.dry_run:
            self._load_dry_run_policy()
            self._loaded = True
            return

        try:
            self._load_current_lerobot_policy(model_source)
        except Exception as current_exc:  # pragma: no cover - depends on runtime image
            logger.warning("Current LeRobot policy load failed, falling back to legacy SmolVLA path: %s", current_exc)
            try:
                self._load_legacy_smolvla_policy(model_source)
            except Exception as legacy_exc:  # pragma: no cover - depends on runtime image
                if any(sep in self.model_repo_id for sep in ("/", "\\")) and model_source == self.model_repo_id:
                    raise PolicyLoadError(
                        f"Failed to load policy '{self.model_repo_id}'. "
                        f"The value looks like a local path, but it does not exist from the current working directory "
                        f"'{Path.cwd()}'. Use an absolute path like "
                        "'/home/aup/outputs/act_so101_test/checkpoints/last/pretrained_model' "
                        "or the repo ID 'Ev3Dev/so101_act_arduino_box'."
                    ) from legacy_exc
                raise PolicyLoadError(
                    f"Failed to load policy '{self.model_repo_id}'. "
                    f"Current LeRobot loader error: {current_exc}"
                ) from legacy_exc

        self._loaded = True

    def unload(self) -> None:
        self._policy = None
        self._preprocessor = None
        self._postprocessor = None
        self._loaded = False
        self._action_buffer.clear()
        self._legacy_smolvla = False
        self._inferred_camera_aliases = {}

    def validate_camera_names(self, configured_camera_names: Iterable[str]) -> None:
        if not self._expected_image_feature_keys:
            raise PolicyCameraValidationError("Policy image feature keys are unavailable")

        expected = [feature_key_to_camera_name(key) for key in self._expected_image_feature_keys]
        configured = sorted(list(configured_camera_names))
        missing_required: list[str] = []
        aliases = self.effective_camera_aliases
        for camera_name in expected:
            resolved_name = aliases.get(camera_name, camera_name)
            if resolved_name in configured:
                continue
            if camera_name in aliases or not is_placeholder_camera_name(camera_name):
                missing_required.append(camera_name)

        if missing_required:
            raise PolicyCameraValidationError(
                "Camera feature mismatch. "
                f"Expected camera names from policy features: {sorted(expected)}. "
                f"Configured camera names: {configured}. "
                f"Camera aliases: {aliases}. "
                f"Missing required policy cameras: {sorted(missing_required)}."
            )

    def infer(self, prepared_observation: dict[str, Any]) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Policy is not loaded")

        if self._action_buffer:
            return self._action_buffer.popleft()

        if self.dry_run:
            return self._dry_run_action(prepared_observation)

        action_output = self._run_model_inference(prepared_observation)
        arr = np.asarray(action_output, dtype=np.float32)
        if arr.ndim == 1:
            return arr.reshape(-1)

        self.buffer_action_chunk(arr)

        if not self._action_buffer:
            raise RuntimeError("Policy produced an empty action output")
        return self._action_buffer.popleft()

    def buffer_action_chunk(self, action_chunk: Any) -> None:
        arr = np.asarray(action_chunk, dtype=np.float32)

        if arr.ndim == 1:
            self._action_buffer.append(arr)
            return

        if arr.ndim == 2:
            for row in arr:
                self._action_buffer.append(np.asarray(row, dtype=np.float32).reshape(-1))
            return

        if arr.ndim >= 3:
            # Common SmolVLA shape: [batch, chunk_size, action_dim]
            for row in arr[0]:
                self._action_buffer.append(np.asarray(row, dtype=np.float32).reshape(-1))
            return

        raise ValueError(f"Unsupported action chunk shape: {arr.shape}")

    def _load_dry_run_policy(self) -> None:
        if self._expected_image_feature_keys:
            return

        if not self.camera_names:
            raise PolicyLoadError("Dry-run policy requires camera_names to infer image feature keys")
        self._expected_image_feature_keys = [f"observation.images.{name}" for name in self.camera_names]

    def _dry_run_action(self, prepared_observation: dict[str, Any]) -> np.ndarray:
        state = np.asarray(prepared_observation.get("observation.state", []), dtype=np.float32).reshape(-1)
        if state.size == 0:
            return np.zeros(7, dtype=np.float32)
        return state

    def _run_model_inference(self, prepared_observation: dict[str, Any]) -> np.ndarray:
        if self._legacy_smolvla:
            return self._run_legacy_smolvla_inference(prepared_observation)

        observation = self._prepare_current_lerobot_observation(prepared_observation)
        if "task" not in observation:
            observation["task"] = ""
        if self.robot_type:
            observation.setdefault("robot_type", self.robot_type)

        processed = observation
        if self._preprocessor is not None:
            processed = self._preprocessor(processed)

        action = self._policy.select_action(processed)
        if self._postprocessor is not None:
            action = self._postprocessor(action)

        if hasattr(action, "detach"):
            action = action.detach().cpu().numpy()

        return np.asarray(action, dtype=np.float32)

    def _prepare_current_lerobot_observation(self, prepared_observation: dict[str, Any]) -> dict[str, Any]:
        import torch  # type: ignore

        device = self._get_torch_device()
        observation: dict[str, Any] = {}
        for key, value in prepared_observation.items():
            if key in {"task", "robot_type"}:
                observation[key] = value
                continue

            if isinstance(value, torch.Tensor):
                tensor = value.detach()
            else:
                tensor = torch.as_tensor(np.asarray(value))

            if tensor.dtype != torch.float32:
                tensor = tensor.float()

            if key == "observation.state" and tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if (key == "observation.image" or key.startswith("observation.images.")) and tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)

            observation[key] = tensor.to(device)

        return observation

    def _get_torch_device(self) -> "Any":
        import torch  # type: ignore

        if self.device:
            return torch.device(self.device)

        policy = self._policy
        if policy is not None:
            parameters = getattr(policy, "parameters", None)
            if callable(parameters):
                try:
                    first_parameter = next(parameters())
                except StopIteration:
                    first_parameter = None
                except TypeError:
                    first_parameter = None
                if first_parameter is not None:
                    return first_parameter.device

        return torch.device("cpu")

    def _load_current_lerobot_policy(self, model_source: str) -> None:
        try:
            from lerobot.configs.policies import PreTrainedConfig  # type: ignore
            from lerobot.policies.factory import (  # type: ignore
                get_policy_class,
                make_pre_post_processors,
            )
        except Exception as exc:
            raise PolicyLoadError(
                "LeRobot pretrained policy dependencies are unavailable in the current runtime."
            ) from exc

        policy_config = PreTrainedConfig.from_pretrained(model_source)
        if self.device:
            policy_config.device = self.device

        policy_cls = get_policy_class(policy_config.type)
        policy = policy_cls.from_pretrained(model_source, config=policy_config).to(policy_config.device)
        policy.eval()
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_config,
            pretrained_path=model_source,
        )

        discovered = _discover_expected_image_feature_keys(policy_config)
        if discovered:
            self._expected_image_feature_keys = discovered

        if not self._expected_image_feature_keys:
            raise PolicyLoadError("Unable to discover policy image feature keys for camera validation")

        self._policy = policy
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._legacy_smolvla = False

    def _discover_local_camera_aliases(self, model_source: str) -> dict[str, str]:
        model_path = Path(model_source)
        if not model_path.is_dir():
            return {}

        preprocessor_path = model_path / "policy_preprocessor.json"
        if not preprocessor_path.is_file():
            return {}

        try:
            payload = json.loads(preprocessor_path.read_text())
        except Exception:
            logger.warning("Failed to parse policy preprocessor config at %s", preprocessor_path, exc_info=True)
            return {}

        steps = payload.get("steps")
        if not isinstance(steps, list):
            return {}

        aliases: dict[str, str] = {}
        for step in steps:
            if not isinstance(step, dict):
                continue
            if step.get("registry_name") != "rename_observations_processor":
                continue
            config = step.get("config")
            if not isinstance(config, dict):
                continue
            rename_map = config.get("rename_map")
            if not isinstance(rename_map, dict):
                continue
            for source_key, target_key in rename_map.items():
                if not isinstance(source_key, str) or not isinstance(target_key, str):
                    continue
                if not source_key.startswith("observation.images.") or not target_key.startswith("observation.images."):
                    continue
                aliases[feature_key_to_camera_name(target_key)] = feature_key_to_camera_name(source_key)

        return aliases

    def _load_legacy_smolvla_policy(self, model_source: str) -> None:
        from transformers import AutoProcessor  # type: ignore
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore

        try:
            device = self.device or "cuda"
            policy = SmolVLAPolicy.from_pretrained(model_source).to(device)
            policy.eval()
            if not hasattr(policy, "language_tokenizer"):
                processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)
                policy.language_tokenizer = processor.tokenizer
        except Exception as exc:
            raise PolicyLoadError(f"Failed to load SmolVLA model from repo '{self.model_repo_id}': {exc}") from exc

        discovered = _discover_expected_image_feature_keys(policy)
        if discovered:
            self._expected_image_feature_keys = discovered

        if not self._expected_image_feature_keys:
            raise PolicyLoadError("Unable to discover policy image feature keys for camera validation")

        self._policy = policy
        self._preprocessor = None
        self._postprocessor = None
        self._legacy_smolvla = True

    def _run_legacy_smolvla_inference(self, prepared_observation: dict[str, Any]) -> np.ndarray:
        import torch  # type: ignore

        task = prepared_observation.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("Prepared observation must include a non-empty task string")

        batch: dict[str, Any] = {"task": [task]}
        device = self.device or "cuda"

        for key, value in prepared_observation.items():
            if key == "task":
                continue

            tensor = torch.as_tensor(np.asarray(value), device=device)
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            if key == "observation.state" and tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if (key == "observation.image" or key.startswith("observation.images.")) and tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            batch[key] = tensor

        normalized = self._policy.normalize_inputs(batch)
        images, image_masks = self._policy.prepare_images(normalized)
        state = self._policy.prepare_state(normalized)
        language_tokens, language_masks = self._policy.prepare_language(normalized)

        with torch.no_grad():
            actions = self._policy.model.sample_actions(
                images,
                image_masks,
                language_tokens,
                language_masks,
                state,
            )

        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()

        return np.asarray(actions, dtype=np.float32)
