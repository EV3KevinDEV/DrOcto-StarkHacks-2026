from __future__ import annotations

import numpy as np

from doc_ock.adapters.robot import DryRunRobotAdapter


def test_dry_run_robot_adapter_observation_and_action_logging() -> None:
    adapter = DryRunRobotAdapter(
        camera_paths={"top": "/dev/webcam_top", "front": "/dev/webcam_front"}
    )
    adapter.start()

    observation = adapter.get_observation()

    assert set(observation["images"].keys()) == {"top", "front"}
    assert observation["state_vector"].ndim == 1

    adapter.send_action(np.asarray([0.1, -0.2, 0.3], dtype=np.float32))
    assert adapter.action_count == 1

    adapter.stop()
