from __future__ import annotations

import pytest

from doc_ock.adapters.policy import PolicyCameraValidationError, SmolVlaPolicyAdapter
from doc_ock.observation import CameraValidationError, ObservationBridge


def test_camera_validation_against_policy_features() -> None:
    adapter = SmolVlaPolicyAdapter(
        model_repo_id="org/model",
        dry_run=True,
        expected_image_feature_keys=[
            "observation.images.top",
            "observation.images.front",
        ],
    )
    adapter.load()

    adapter.validate_camera_names(["front", "top"])

    with pytest.raises(PolicyCameraValidationError):
        adapter.validate_camera_names(["top"])


def test_camera_validation_in_bridge() -> None:
    bridge = ObservationBridge(["observation.images.top", "observation.images.front"])
    bridge.validate_camera_names(["top", "front"])

    with pytest.raises(CameraValidationError):
        bridge.validate_camera_names(["top", "wrist"])
