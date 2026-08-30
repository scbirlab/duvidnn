import torch

from aspect.collate.torch import TorchColumnCollator


def test_torch_column_collator():
    collator = TorchColumnCollator()
    batch = [
        {"x": [1., 2.], "y": 1.},
        {"x": [30, 4.], "y": 2.},
    ]
    batch = collator(batch)

    assert torch.is_tensor(batch["x"])
    assert torch.is_tensor(batch["y"])

    assert batch["x"].shape == (2, 2)
    assert batch["y"].shape == (2,)
