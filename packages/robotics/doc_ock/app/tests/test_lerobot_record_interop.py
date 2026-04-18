from __future__ import annotations

from fastapi.testclient import TestClient

from doc_ock.api import create_app
from doc_ock.interop import translate_lerobot_record_command
from doc_ock.models import RuntimeCameraConfig, RuntimeConfig
from doc_ock.runtime import DocOckRuntime


LEROBOT_RECORD_COMMAND = """\
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower1 \
  --robot.cameras="{camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
  --dataset.single_task="start stroking it" \
  --dataset.repo_id=Ev3Dev/eval_act_arduino_test \
  --dataset.episode_time_s=60 \
  --dataset.num_episodes=1 \
  --policy.path=outputs/act_so101_test/checkpoints/last/pretrained_model \
  --dataset.push_to_hub=false
"""


def test_translate_lerobot_record_command() -> None:
    translation = translate_lerobot_record_command(LEROBOT_RECORD_COMMAND)

    assert translation.robot_type == "so101_follower"
    assert translation.robot_port == "/dev/ttyACM0"
    assert translation.robot_id == "follower1"
    assert translation.task == "start stroking it"
    assert translation.model_repo_id == "outputs/act_so101_test/checkpoints/last/pretrained_model"
    assert translation.max_duration_s == 60.0
    assert translation.cameras["camera1"] == RuntimeCameraConfig(
        type="opencv",
        index_or_path=0,
        width=640,
        height=480,
        fps=30,
        fourcc=None,
    )
    assert translation.cameras["camera2"].index_or_path == 2
    assert translation.ignored_arguments == {
        "dataset.repo_id": "Ev3Dev/eval_act_arduino_test",
        "dataset.num_episodes": "1",
        "dataset.push_to_hub": "false",
    }


def test_http_start_from_lerobot_record_command() -> None:
    runtime = DocOckRuntime(
        RuntimeConfig(
            robot_port="/dev/ttyACM0",
            robot_type="so101_follower",
            robot_id="follower1",
            cameras={
                "top": RuntimeCameraConfig(index_or_path=0, width=640, height=480, fps=30, fourcc=None),
                "side": RuntimeCameraConfig(index_or_path=2, width=640, height=480, fps=30, fourcc=None),
            },
            camera_aliases={
                "camera1": "top",
                "camera2": "side",
            },
            dry_run=True,
            step_delay_s=0.002,
        )
    )
    app = create_app(runtime)
    client = TestClient(app)

    response = client.post(
        "/session/start-from-lerobot-record",
        json={"command": LEROBOT_RECORD_COMMAND},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["state"] in {"starting", "running"}
    assert payload["task"] == "start stroking it"
    assert payload["model_repo_id"] == "outputs/act_so101_test/checkpoints/last/pretrained_model"
    assert payload["lerobot_translation"]["session_start"] == {
        "task": "start stroking it",
        "model_repo_id": "outputs/act_so101_test/checkpoints/last/pretrained_model",
        "max_duration_s": 60.0,
    }
    assert payload["lerobot_translation"]["runtime"]["cameras"]["camera1"]["index_or_path"] == 0
    assert payload["lerobot_translation"]["runtime"]["cameras"]["camera2"]["index_or_path"] == 2
