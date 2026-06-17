# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LEROBOT_HOME=/lerobot
ENV LEROBOT_HOME=${LEROBOT_HOME}

WORKDIR ${LEROBOT_HOME}
COPY test.sh ${LEROBOT_HOME}/test_ryzer_env.sh
RUN chmod +x ${LEROBOT_HOME}/test_ryzer_env.sh

CMD "${LEROBOT_HOME}/test_ryzer_env.sh"
