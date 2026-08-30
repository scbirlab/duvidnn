import torch
from torch import nn

from duvidnn.models import MultiTower


class Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full(
            (x.shape[0], 1),
            self.value,
        )


def test_multitower_concat_uses_tower_order():
    model = MultiTower(
        towers={
            "first": Constant(1),
            "second": Constant(2),
        },
        fusion=nn.Identity(),
        merge="concat",
    )

    output = model(
        second=torch.zeros(3, 1),
        first=torch.zeros(3, 1),
    )

    expected = torch.tensor(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ]
    )

    assert torch.equal(
        output,
        expected,
    )

class StructuredModule(nn.Module):
    def forward(self, *, x, multiplier):
        return x * multiplier


def test_multitower_dispatches_mapping_as_kwargs():
    model = MultiTower(
        towers={
            "structured": StructuredModule(),
            "plain": nn.Identity(),
        },
        fusion=nn.Identity(),
        merge="concat",
    )

    output = model(
        structured={
            "x": torch.ones(2, 3),
            "multiplier": 2,
        },
        plain=torch.zeros(2, 2),
    )

    assert output.shape == (2, 5)
    assert torch.equal(
        output[:, :3],
        torch.full((2, 3), 2.0),
    )
