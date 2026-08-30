import torch

from duvidnn.models import MLP


def test_mlp_explicit_input_dim():
    model = MLP(
        input_dim=8,
        hidden_dims=[16, 4],
        output_dim=2,
    )

    x = torch.randn(5, 8)
    y = model(x)

    assert y.shape == (5, 2)


def test_mlp_lazy_input_dim():
    model = MLP(
        hidden_dims=[16],
        output_dim=3,
    )

    x = torch.randn(7, 11)
    y = model(x)

    assert y.shape == (7, 3)


def test_mlp_no_hidden_layers():
    model = MLP(
        input_dim=8,
        hidden_dims=[],
        output_dim=1,
    )

    x = torch.randn(5, 8)
    y = model(x)

    assert y.shape == (5, 1)


def test_mlp_json_friendly_configuration():
    config = {
        "input_dim": 8,
        "hidden_dims": [16, 4],
        "output_dim": 2,
        "activation": "silu",
        "dropout": 0.1,
        "batch_norm": True,
    }

    model = MLP(**config)

    x = torch.randn(5, 8)
    y = model(x)

    assert y.shape == (5, 2)
