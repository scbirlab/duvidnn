
import torch
from torch import nn
import pytest

from duvidnn.models import FiLM
from duvidnn.config import instantiate_model


def test_film():

    modulator = nn.Linear(2, 6, bias=False)

    with torch.no_grad():
        modulator.weight.zero_()
        modulator.weight[:3, 0] = 2.
        modulator.weight[3:, 1] = 1.

    model = FiLM(modulator=modulator)

    input = torch.tensor([
        [1., 2., 3.],
    ])

    context = torch.tensor([
        [1., 1.],
    ])

    observed = model(
        input,
        context,
    )

    expected = torch.tensor([
        [3., 5., 7.],
    ])

    assert torch.allclose(observed, expected)


def test_soft_film_zero_modulation_is_identity():

    modulator = nn.Linear(2, 6)

    with torch.no_grad():
        modulator.weight.zero_()
        modulator.bias.zero_()

    model = FiLM(
        modulator=modulator,
        soft=True,
    )

    input = torch.randn(4, 3)
    context = torch.randn(4, 2)
    observed = model(input, context)

    assert torch.allclose(observed, input)


def test_film_validates_modulation_width():

    model = FiLM(
        modulator=nn.Linear(2, 4),
    )

    with pytest.raises(
        ValueError,
        match="must match input width",
    ):
        model(
            torch.randn(3, 3),
            torch.randn(3, 2),
        )


def test_film_from_config():

    model = instantiate_model({
        "class_path": (
            "duvidnn.models.FiLM"
        ),
        "init_args": {
            "modulator": {
                "class_path": (
                    "duvidnn.models.MLP"
                ),
                "init_args": {
                    "in_features": 2,
                    "hidden_dims": 4,
                    "out_features": 6,
                },
            },
            "soft": True,
        },
    })

    assert isinstance(model, FiLM)

    observed = model(
        torch.randn(5, 3),
        torch.randn(5, 2),
    )
    assert observed.shape == (5, 3)
