from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


def test_camera_aliases_allow_policy_and_runtime_names_to_differ() -> None:
    adapter = SmolVlaPolicyAdapter(
        model_repo_id="org/model",
        dry_run=True,
        expected_image_feature_keys=[
            "observation.images.camera1",
            "observation.images.camera2",
        ],
        camera_aliases={
            "camera1": "top",
            "camera2": "side",
        },
    )
    adapter.load()
    adapter.validate_camera_names(["top", "side"])

    bridge = ObservationBridge(
        ["observation.images.camera1", "observation.images.camera2"],
        camera_aliases={
            "camera1": "top",
            "camera2": "side",
        },
    )
    bridge.validate_camera_names(["top", "side"])


def test_policy_infers_camera_aliases_from_local_preprocessor(tmp_path: Path) -> None:
    (tmp_path / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "rename_observations_processor",
                        "config": {
                            "rename_map": {
                                "observation.images.side": "observation.images.camera1",
                                "observation.images.top": "observation.images.camera2",
                            }
                        },
                    }
                ]
            }
        )
    )

    adapter = SmolVlaPolicyAdapter(
        model_repo_id=str(tmp_path),
        dry_run=True,
        expected_image_feature_keys=[
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ],
    )
    adapter.load()

    assert adapter.effective_camera_aliases == {
        "camera1": "side",
        "camera2": "top",
    }
    adapter.validate_camera_names(["top", "side"])


def test_bridge_synthesizes_placeholder_camera_images() -> None:
    bridge = ObservationBridge(
        [
            "observation.images.camera1",
            "observation.images.camera2",
            "observation.images.camera3",
        ],
        camera_aliases={
            "camera1": "side",
            "camera2": "top",
        },
    )
    bridge.validate_camera_names(["top", "side"])

    observation = {
        "images": {
            "top": np.ones((4, 5, 3), dtype=np.uint8),
            "side": np.full((4, 5, 3), fill_value=9, dtype=np.uint8),
        },
        "state": [1, 2, 3, 4, 5, 6],
    }
    prepared = bridge.to_policy_observation(observation, "grab the arduino box and move it")

    assert prepared["observation.images.camera1"].shape == (3, 4, 5)
    assert prepared["observation.images.camera2"].shape == (3, 4, 5)
    assert prepared["observation.images.camera3"].shape == (3, 4, 5)
    assert np.count_nonzero(prepared["observation.images.camera3"]) == 0


def test_policy_resolves_parent_relative_model_path_from_app_cwd(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "workspace" / "packages" / "robotics" / "doc_ock" / "app"
    model_dir = tmp_path / "outputs" / "act_so101_test" / "checkpoints" / "last" / "pretrained_model"
    app_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    monkeypatch.chdir(app_dir)

    adapter = SmolVlaPolicyAdapter(
        model_repo_id="../outputs/act_so101_test/checkpoints/last/pretrained_model",
        dry_run=True,
    )

    assert adapter.resolved_model_source == str(model_dir.resolve())
