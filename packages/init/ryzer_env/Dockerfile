# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers
COPY test.sh /ryzers/test_ryzer_env.sh
RUN chmod +x /ryzers/test_ryzer_env.sh

CMD /ryzers/test_ryzer_env.sh
