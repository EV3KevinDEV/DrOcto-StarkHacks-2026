# Doc Ock (LeRobot)

Doc Ock is a thin local runtime layer on top of LeRobot, with:

- one active session lifecycle manager
- SmolVLA-first inference adapter
- named-camera observation/action bridge
- CLI + HTTP control surfaces
- real dry-run mode
- process-wide voice-mode toggle state (keyboard + HTTP)

## Install Into The Existing `lerobot` Environment

Activate the environment on this device:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
```

If needed, refresh the local LeRobot checkout with SmolVLA support:

```bash
cd /home/aup/lerobot
pip install -e ".[smolvla]"
```

Install the Doc Ock package from this repo into that same environment:

```bash
cd /home/aup/DrOcto-StarkHacks-2026/packages/robotics/doc_ock
pip install -e ./app
```

## Optional Container Layer

If you still want a container image, the included `Dockerfile` now expects a LeRobot checkout at `/lerobot` by default. Override `LEROBOT_HOME` at build time if your base image uses a different path.

## SO-101 hardware mapping used on this machine

- Follower arm: `so101_follower`, port `/dev/ttyACM0`, id `follower1`
- Leader arm: `so101_leader`, port `/dev/ttyACM1`, id `leader1`
- Bird camera: `/dev/video0`
- Robot camera: `/dev/video2`
- Default camera profile: `640x480 @ 30 FPS`, FOURCC `YUYV`

## CLI examples

```bash
python -m doc_ock.cli run \
  --model-repo-id outputs/act_so101_test/checkpoints/last/pretrained_model \
  --task "grab the arduino box and move it" \
  --robot-type so101_follower \
  --robot-port /dev/ttyACM0 \
  --robot-id follower1 \
  --teleop-type so101_leader \
  --teleop-port /dev/ttyACM1 \
  --teleop-id leader1 \
  --camera top=/dev/video0 \
  --camera side=/dev/video2 \
  --camera-alias camera1=top \
  --camera-alias camera2=side
```

```bash
python -m doc_ock.cli serve \
  --robot-type so101_follower \
  --robot-port /dev/ttyACM0 \
  --robot-id follower1 \
  --teleop-type so101_leader \
  --teleop-port /dev/ttyACM1 \
  --teleop-id leader1 \
  --camera top=/dev/video0 \
  --camera side=/dev/video2 \
  --camera-alias camera1=top \
  --camera-alias camera2=side
```

When using the HTTP API after `serve`, the session body stays the same and `model_repo_id` can be either a Hugging Face repo ID or a local pretrained policy directory:

```json
{
  "task": "grab the arduino box and move it",
  "model_repo_id": "outputs/act_so101_test/checkpoints/last/pretrained_model"
}
```

Press `v` in `run` or `serve` to toggle voice mode. Voice mode is state-only in v1 and does not pause inference/motion.
