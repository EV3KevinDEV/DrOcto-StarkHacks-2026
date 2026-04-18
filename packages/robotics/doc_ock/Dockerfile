# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# The vendor lerobot base image clones LeRobot into /ryzers/lerobot but only
# installs a subset of extras. SmolVLA is a required dependency for real-mode
# inference in SmolVlaPolicyAdapter.load(), so install the [smolvla] extra
# against the same checkout.
RUN if [ -d /ryzers/lerobot ]; then \
      cd /ryzers/lerobot && pip install --no-cache-dir --break-system-packages ".[smolvla]" ; \
    else \
      echo "WARNING: /ryzers/lerobot not found in base image; skipping smolvla extras install" ; \
    fi

RUN pip install --no-cache-dir --break-system-packages \
    "fastapi>=0.115,<1" \
    "uvicorn>=0.30,<1" \
    "pydantic>=2.7,<3" \
    "pytest>=8.3,<9" \
    "httpx>=0.27,<1"

COPY app /ryzers/app
RUN pip install --no-cache-dir --break-system-packages -e /ryzers/app

COPY test.sh /ryzers/test_doc_ock.sh
RUN chmod +x /ryzers/test_doc_ock.sh

EXPOSE 8080

CMD /ryzers/test_doc_ock.sh
