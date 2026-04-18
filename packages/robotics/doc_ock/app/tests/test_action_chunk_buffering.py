from __future__ import annotations

import numpy as np
import torch

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


def test_current_policy_path_batches_image_and_state_tensors() -> None:
    class FakePolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
            assert batch["observation.images.camera1"].shape == (1, 3, 480, 640)
            assert batch["observation.state"].shape == (1, 6)
            assert batch["observation.images.camera1"].dtype == torch.float32
            assert batch["observation.state"].dtype == torch.float32
            return torch.tensor([0.25, -0.5], dtype=torch.float32, device=self._anchor.device)

    adapter = SmolVlaPolicyAdapter(model_repo_id="org/model", dry_run=False)
    adapter._policy = FakePolicy()
    adapter._preprocessor = lambda batch: batch
    adapter._postprocessor = None
    adapter._loaded = True

    action = adapter.infer(
        {
            "task": "start stroking it",
            "observation.state": np.asarray([1, 2, 3, 4, 5, 6], dtype=np.float32),
            "observation.images.camera1": np.zeros((3, 480, 640), dtype=np.float32),
        }
    )

    np.testing.assert_allclose(action, np.asarray([0.25, -0.5], dtype=np.float32))
