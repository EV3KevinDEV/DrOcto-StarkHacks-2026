from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class CameraValidationError(ValueError):
    """Raised when configured camera names do not match policy feature names."""


def feature_key_to_camera_name(feature_key: str) -> str:
    if feature_key.startswith("observation.images."):
        return feature_key[len("observation.images.") :]
    if feature_key == "observation.image":
        return "image"
    raise ValueError(f"Unsupported image feature key: {feature_key}")


class ObservationBridge:
    """Bridge from robot observations to policy-ready observation dicts."""

    def __init__(self, expected_image_feature_keys: Sequence[str], state_keys: Sequence[str] | None = None):
        if not expected_image_feature_keys:
            raise ValueError("expected_image_feature_keys must not be empty")
        self._expected_image_feature_keys = list(expected_image_feature_keys)
        self._state_keys = list(state_keys) if state_keys else None

    @property
    def expected_image_feature_keys(self) -> list[str]:
        return list(self._expected_image_feature_keys)

    def expected_camera_names(self) -> list[str]:
        return [feature_key_to_camera_name(k) for k in self._expected_image_feature_keys]

    def validate_camera_names(self, configured_camera_names: Iterable[str]) -> None:
        expected = sorted(self.expected_camera_names())
        configured = sorted(list(configured_camera_names))
        if expected != configured:
            raise CameraValidationError(
                "Camera feature mismatch. "
                f"Expected camera names from policy features: {expected}. "
                f"Configured camera names: {configured}."
            )

    def to_policy_observation(self, observation: Mapping[str, Any], task_text: str) -> dict[str, Any]:
        images = observation.get("images")
        if not isinstance(images, Mapping):
            raise ValueError("Robot observation must include an 'images' mapping")

        bridged: dict[str, Any] = {}
        for feature_key in self._expected_image_feature_keys:
            camera_name = feature_key_to_camera_name(feature_key)
            image = self._resolve_image(images, camera_name)
            bridged[feature_key] = self._to_chw_float(image)

        bridged["observation.state"] = self._extract_state_vector(observation)
        bridged["task"] = task_text
        return bridged

    def _resolve_image(self, images: Mapping[str, Any], camera_name: str) -> Any:
        if camera_name == "image":
            if len(images) != 1:
                raise ValueError("Policy expects a single 'observation.image', but multiple cameras were provided")
            return next(iter(images.values()))

        if camera_name not in images:
            raise ValueError(f"Missing image for camera name: {camera_name}")
        return images[camera_name]

    def _to_chw_float(self, image: Any) -> np.ndarray:
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)

        if arr.ndim != 3:
            raise ValueError(f"Expected image with 3 dimensions, got shape {arr.shape}")

        if arr.shape[-1] in (1, 3):
            if arr.shape[-1] == 1:
                arr = np.repeat(arr, repeats=3, axis=-1)
            chw = np.transpose(arr, (2, 0, 1))
        elif arr.shape[0] in (1, 3):
            chw = arr
            if arr.shape[0] == 1:
                chw = np.repeat(arr, repeats=3, axis=0)
        else:
            raise ValueError(f"Unsupported image shape for conversion: {arr.shape}")

        chw = chw.astype(np.float32)
        if chw.max(initial=0.0) > 1.0:
            chw /= 255.0
        return chw

    def _extract_state_vector(self, observation: Mapping[str, Any]) -> np.ndarray:
        state_vector = observation.get("state_vector")
        if state_vector is not None:
            arr = np.asarray(state_vector, dtype=np.float32)
            return arr.reshape(-1)

        state = observation.get("state")
        if isinstance(state, Mapping):
            keys = self._state_keys if self._state_keys is not None else sorted(state.keys())
            values = [float(state[key]) for key in keys if key in state]
            if not values:
                raise ValueError("State mapping is empty after applying state keys")
            return np.asarray(values, dtype=np.float32)

        if state is not None:
            arr = np.asarray(state, dtype=np.float32)
            return arr.reshape(-1)

        raise ValueError("Robot observation must include 'state' or 'state_vector'")
