import torch
from torch import nn

from duvidnn.models import (
    CNN2D,
    Ensemble,
    MLP,
    ResidualBlock,
)


def test_residual_block():

    block = ResidualBlock(
        module=MLP(
            in_features=4,
            hidden_dims=8,
            out_features=4,
        ),
        in_features=4,
        out_features=4,
    )

    observed = block(torch.randn(3, 4))

    assert observed.shape == (3, 4)


def test_residual_block_projection():

    block = ResidualBlock(
        module=MLP(
            in_features=4,
            hidden_dims=8,
            out_features=6,
        ),
        in_features=4,
        out_features=6,
    )

    observed = block(torch.randn(3, 4))

    assert observed.shape == (3, 6)


def test_ensemble():

    ensemble = Ensemble([
        nn.Linear(3, 2) for _ in range(3)
    ])
    observed = ensemble(
        torch.randn(4, 3),
    )

    assert observed.shape == (4, 2, 3)


def test_cnn2d():

    model = CNN2D(
        in_channels=3,
        channels=[8, 16],
        hidden_dims=32,
        out_features=2,
    )

    observed = model(
        torch.randn(
            4,
            3,
            32,
            32,
        )
    )

    assert observed.shape == (4, 2)
