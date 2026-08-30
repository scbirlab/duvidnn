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


class MappingTower(nn.Module):
    def forward(self, x):
        return x["value"]


def test_multitower_passes_mapping_as_single_input():
    model = MultiTower(
        towers={
            "left": MappingTower(),
            "right": nn.Identity(),
        },
        fusion=nn.Identity(),
        merge="sum",
    )

    left = {
        "value": torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
    }

    right = torch.tensor([
        [10.0, 20.0],
        [30.0, 40.0],
    ])

    output = model(
        left=left,
        right=right,
    )

    assert torch.equal(
        output,
        torch.tensor([
            [11.0, 22.0],
            [33.0, 44.0],
        ]),
    )