from __future__ import annotations

import numpy as np

from doc_ock.adapters.policy import SmolVlaPolicyAdapter


def test_action_chunk_buffering() -> None:
    adapter = SmolVlaPolicyAdapter(
        model_repo_id="org/model",
        dry_run=True,
        camera_names=["top"],
    )
    adapter.load()

    adapter.buffer_action_chunk(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))

    first = adapter.infer({"task": "test", "observation.state": np.asarray([9.0, 9.0], dtype=np.float32)})
    second = adapter.infer({"task": "test", "observation.state": np.asarray([9.0, 9.0], dtype=np.float32)})

    np.testing.assert_allclose(first, np.asarray([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(second, np.asarray([3.0, 4.0], dtype=np.float32))
    assert adapter.buffered_action_count == 0
