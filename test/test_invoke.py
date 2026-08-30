import torch
from torch import nn

from duvidnn.invoke import ModelInvoker
from duvidnn.mapping import ColumnMap
from duvidnn.models import MLP, MultiTower


def test_invoke_model():
    model = MultiTower(
        towers={
            "left": MLP(
                input_dim=4,
                hidden_dims=[],
                output_dim=3,
            ),
            "right": MLP(
                input_dim=5,
                hidden_dims=[],
                output_dim=3,
            ),
        },
        fusion=MLP(
            hidden_dims=[],
            output_dim=1,
        ),
    )

    mapping = ColumnMap(
        inputs={
            "left": "features_a",
            "right": "features_b",
        },
        target="y",
    )

    batch = {
        "features_a": torch.randn(8, 4),
        "features_b": torch.randn(8, 5),
        "y": torch.randn(8, 1),
    }

    invoker = ModelInvoker(
        model=model,
        input_map=mapping,
    )
    output = invoker(batch)

    assert output.shape == (8, 1)


def test_invoke_supervised():
    model = MultiTower(
        towers={
            "left": MLP(
                input_dim=4,
                hidden_dims=[],
                output_dim=3,
            ),
            "right": MLP(
                input_dim=5,
                hidden_dims=[],
                output_dim=3,
            ),
        },
        fusion=MLP(
            hidden_dims=[],
            output_dim=1,
        ),
    )

    mapping = ColumnMap(
        inputs={
            "left": "features_a",
            "right": "features_b",
        },
        target="y",
    )

    target = torch.randn(8, 1)

    batch = {
        "features_a": torch.randn(8, 4),
        "features_b": torch.randn(8, 5),
        "y": target,
    }

    invoker = ModelInvoker(
        model=model,
        input_map=mapping,
    )
    prediction, observed = invoker.supervised(batch)

    assert prediction.shape == (8, 1)
    assert observed is target


class StructuredTower(nn.Module):

    def forward(self, *, x, scale):
        return x * scale


def test_invoke_structured_input():
    model = MultiTower(
        towers={
            "compound": StructuredTower(),
            "context": nn.Identity(),
        },
        fusion=nn.Identity(),
    )

    mapping = ColumnMap(
        inputs={
            "compound": "chemprop_like",
            "context": "context_features",
        },
        target="y",
    )

    batch = {
        "chemprop_like": {
            "x": torch.ones(3, 2),
            "scale": 2,
        },
        "context_features": torch.zeros(3, 2),
        "y": torch.zeros(3, 1),
    }

    invoker = ModelInvoker(
        model=model,
        input_map=mapping,
    )

    output = invoker(batch)

    assert output.shape == (3, 4)
