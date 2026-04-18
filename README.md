# DrOcto-StarkHacks-2026

This repo is configured to run Doc Ock from the existing `lerobot` environment on this device.

The earlier `Ryzers` packaging was only kept around for cross-device testing. It is now legacy scaffolding, not the primary runtime path here.

## Local environment

Activate the existing Conda environment:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lerobot
```

If you need to refresh the upstream LeRobot checkout on this machine:

```bash
cd /home/aup/lerobot
pip install -e ".[smolvla]"
```

Install the Doc Ock app layer from this repo into that same environment:

```bash
cd /home/aup/DrOcto-StarkHacks-2026
pip install -e packages/robotics/doc_ock/app
```

## Run

CLI mode:

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

HTTP mode:

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

The local Doc Ock runtime serves HTTP on port `8080`.

## Repo layout

- `packages/robotics/doc_ock/`: active Doc Ock runtime layer for LeRobot + SmolVLA control
- `packages/init/ryzer_env/`: legacy builder shim from earlier `Ryzers` testing; not used for the on-device `lerobot` flow
