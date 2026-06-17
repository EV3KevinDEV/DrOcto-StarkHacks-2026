#!/usr/bin/env bash

# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set -euo pipefail

python -m doc_ock.cli run \
  --model-repo-id local/smoke-smolvla \
  --task "smoke test" \
  --robot-port /dev/ttyACM_follower \
  --camera top=/dev/webcam_top \
  --camera front=/dev/webcam_front \
  --dry-run \
  --max-steps 2 \
  --no-interactive

echo "doc_ock smoke test passed"
