# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG LEROBOT_HOME=/lerobot
ENV LEROBOT_HOME=${LEROBOT_HOME}

WORKDIR ${LEROBOT_HOME}

RUN cd ${LEROBOT_HOME} && pip install --no-cache-dir ".[smolvla]"

RUN pip install --no-cache-dir --break-system-packages \
    "fastapi>=0.115,<1" "uvicorn>=0.30,<1" "pydantic>=2.7,<3" \
    "pytest>=8.3,<9" "httpx>=0.27,<1"

COPY app ${LEROBOT_HOME}/app
RUN pip install --no-cache-dir --break-system-packages -e ${LEROBOT_HOME}/app

COPY test.sh ${LEROBOT_HOME}/test_doc_ock.sh
RUN chmod +x ${LEROBOT_HOME}/test_doc_ock.sh

EXPOSE 8080
CMD "${LEROBOT_HOME}/test_doc_ock.sh"
