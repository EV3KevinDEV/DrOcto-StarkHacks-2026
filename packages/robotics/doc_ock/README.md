# Doc Ock (Ryzers + LeRobot)

Doc Ock is a thin local runtime layer on top of a LeRobot image, with:

- one active session lifecycle manager
- SmolVLA-first inference adapter
- named-camera observation/action bridge
- CLI + HTTP control surfaces
- real dry-run mode
- process-wide voice-mode toggle state (keyboard + HTTP)

## Build flow (two stages)

1. Build upstream LeRobot image from vendor packages:

```bash
ryzers build lerobot --base_path ./vendor/Ryzers/packages --name lerobot-upstream
```

2. Build local Doc Ock package using that image as init image:

```bash
ryzers build doc_ock --base_path . --name doc-ock-ryzers --init_image lerobot-upstream
```

## Run

```bash
ryzers run --name doc-ock-ryzers
```

## CLI examples

```bash
python -m doc_ock.cli run \
  --model-repo-id <hf-org-or-user>/<smolvla-model> \
  --task "pick the green block" \
  --robot-port /dev/ttyACM_follower \
  --camera top=/dev/webcam_top \
  --camera front=/dev/webcam_front
```

```bash
python -m doc_ock.cli serve \
  --robot-port /dev/ttyACM_follower \
  --camera top=/dev/webcam_top \
  --camera front=/dev/webcam_front
```

Press `v` in `run` or `serve` to toggle voice mode. Voice mode is state-only in v1 and does not pause inference/motion.
