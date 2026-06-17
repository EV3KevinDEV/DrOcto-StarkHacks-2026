from __future__ import annotations

import pytest

from doc_ock.models import RuntimeConfig
from doc_ock.runtime import DocOckRuntime


@pytest.fixture
def runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        robot_port="/dev/ttyACM_follower",
        cameras={
            "top": "/dev/webcam_top",
            "front": "/dev/webcam_front",
        },
        dry_run=True,
        step_delay_s=0.002,
    )


@pytest.fixture
def runtime(runtime_config: RuntimeConfig) -> DocOckRuntime:
    return DocOckRuntime(runtime_config)
