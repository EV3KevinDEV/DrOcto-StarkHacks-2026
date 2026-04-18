from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Sequence

import numpy as np

from doc_ock.observation import feature_key_to_camera_name


class PolicyLoadError(RuntimeError):
    """Raised when SmolVLA policy loading fails."""


class PolicyCameraValidationError(ValueError):
    """Raised when configured camera names do not match model features."""


def _discover_expected_image_feature_keys(policy_obj: Any) -> list[str]:
    config = getattr(policy_obj, "config", None)
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
    def __init__(
        self,
        model_repo_id: str,
        camera_names: Sequence[str] | None = None,
        dry_run: bool = False,
        device: str | None = None,
        expected_image_feature_keys: Sequence[str] | None = None,
    ):
        self.model_repo_id = model_repo_id.strip()
        self.camera_names = list(camera_names) if camera_names else []
        self.dry_run = dry_run
        self.device = device or "cuda"
        self._expected_image_feature_keys = list(expected_image_feature_keys) if expected_image_feature_keys else None

        self._policy: Any = None
        self._loaded = False
        self._action_buffer: deque[np.ndarray] = deque()

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

    def load(self) -> None:
        if self._loaded:
            return

        if not self.model_repo_id:
            raise PolicyLoadError("model_repo_id is required")

        if self.dry_run:
            self._load_dry_run_policy()
            self._loaded = True
            return

        try:
            import torch  # type: ignore
            from transformers import AutoProcessor  # type: ignore
            from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on runtime image
            raise PolicyLoadError(
                "SmolVLA dependencies are unavailable. Ensure LeRobot with smolvla extras is installed."
            ) from exc

        try:
            policy = SmolVLAPolicy.from_pretrained(self.model_repo_id).to(self.device)
            policy.eval()
            if not hasattr(policy, "language_tokenizer"):
                processor = AutoProcessor.from_pretrained(policy.config.vlm_model_name)
                policy.language_tokenizer = processor.tokenizer
        except Exception as exc:  # pragma: no cover - depends on remote model access
            raise PolicyLoadError(f"Failed to load SmolVLA model from repo '{self.model_repo_id}': {exc}") from exc

        discovered = _discover_expected_image_feature_keys(policy)
        if discovered:
            self._expected_image_feature_keys = discovered

        if not self._expected_image_feature_keys:
            raise PolicyLoadError(
                "Unable to discover policy image feature keys for camera validation"
            )

        self._policy = policy
        self._loaded = True

    def unload(self) -> None:
        self._policy = None
        self._loaded = False
        self._action_buffer.clear()

    def validate_camera_names(self, configured_camera_names: Iterable[str]) -> None:
        if not self._expected_image_feature_keys:
            raise PolicyCameraValidationError("Policy image feature keys are unavailable")

        expected = sorted(feature_key_to_camera_name(key) for key in self._expected_image_feature_keys)
        configured = sorted(list(configured_camera_names))
        if expected != configured:
            raise PolicyCameraValidationError(
                "Camera feature mismatch. "
                f"Expected camera names from policy features: {expected}. "
                f"Configured camera names: {configured}."
            )

    def infer(self, prepared_observation: dict[str, Any]) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("Policy is not loaded")

        if self._action_buffer:
            return self._action_buffer.popleft()

        if self.dry_run:
            return self._dry_run_action(prepared_observation)

        action_chunk = self._run_model_inference(prepared_observation)
        self.buffer_action_chunk(action_chunk)

        if not self._action_buffer:
            raise RuntimeError("SmolVLA produced an empty action chunk")
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
        import torch  # type: ignore

        task = prepared_observation.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError("Prepared observation must include a non-empty task string")

        batch: dict[str, Any] = {"task": [task]}

        for key, value in prepared_observation.items():
            if key == "task":
                continue

            tensor = torch.as_tensor(np.asarray(value), device=self.device)
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
