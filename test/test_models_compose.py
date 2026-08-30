import pytest
import torch
from torch import nn

from duvidnn.models import MLP, TwoTower


def test_two_tower_concat():
    model = TwoTower(
        left=MLP(
            input_dim=8,
            hidden_dims=[],
            output_dim=4,
        ),
        right=MLP(
            input_dim=6,
            hidden_dims=[],
            output_dim=3,
        ),
        fusion=MLP(
            input_dim=7,
            hidden_dims=[5],
            output_dim=1,
        ),
        merge="concat",
    )

    y = model(
        left=torch.randn(10, 8),
        right=torch.randn(10, 6),
    )

    assert y.shape == (10, 1)


@pytest.mark.parametrize(
    "merge",
    [
        "sum",
        "product",
    ],
)
def test_two_tower_elementwise_merge(merge):
    model = TwoTower(
        left=MLP(
            input_dim=8,
            hidden_dims=[],
            output_dim=4,
        ),
        right=MLP(
            input_dim=6,
            hidden_dims=[],
            output_dim=4,
        ),
        fusion=MLP(
            input_dim=4,
            hidden_dims=[],
            output_dim=1,
        ),
        merge=merge,
    )

    y = model(
        left=torch.randn(10, 8),
        right=torch.randn(10, 6),
    )

    assert y.shape == (10, 1)


def test_two_tower_accepts_arbitrary_modules():
    model = TwoTower(
        left=nn.Identity(),
        right=nn.Identity(),
        fusion=nn.Identity(),
        merge="concat",
    )

    left = torch.randn(3, 4)
    right = torch.randn(3, 2)

    y = model(left=left, right=right)

    assert y.shape == (3, 6)


def test_two_tower_gradients_reach_both_towers():
    model = TwoTower(
        left=nn.Linear(4, 3),
        right=nn.Linear(5, 3),
        fusion=nn.Linear(6, 1),
    )

    y = model(
        left=torch.randn(8, 4),
        right=torch.randn(8, 5),
    )

    y.sum().backward()

    assert model.left.weight.grad is not None
    assert model.right.weight.grad is not None
    assert model.fusion.weight.grad is not None


def test_two_tower_rejects_unknown_merge():
    with pytest.raises(ValueError):
        TwoTower(
            left=nn.Identity(),
            right=nn.Identity(),
            fusion=nn.Identity(),
            merge="nonsense",
        )
