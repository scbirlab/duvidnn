
import pytest
import torch
from torch import nn
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
)

from duvidnn import Box, ColumnMap


def test_evaluate():

    model = nn.Linear(1, 1, bias=False)

    with torch.no_grad():
        model.weight.fill_(2.)

    box = Box(
        model=model,
        input_map=ColumnMap(
            inputs={"input": "x"},
            target="y",
        ),
    )

    observed = box.evaluate(
        {
            "x": [
                [1.],
                [2.],
                [3.],
            ],
            "y": [
                [2.],
                [4.],
                [7.],
            ],
        },
        metrics={
            "mae": MeanAbsoluteError(),
            "mse": MeanSquaredError(),
        },
        batch_size=2,
    )

    assert observed["mae"] == pytest.approx(1 / 3)
    assert observed["mse"] == pytest.approx(1 / 3)


def test_evaluate_preserves_model_mode():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.)

    box = Box(
        model=model,
        input_map=ColumnMap(
            inputs={"input": "x"},
            target="y",
        ),
    )
    box.model.train()

    data = {
        "x": [
            [1.],
            [2.],
            [3.],
        ],
        "y": [
            [2.],
            [4.],
            [7.],
        ],
    }
    box.evaluate(
        data,
        metrics={"mae": MeanAbsoluteError()},
    )

    assert box.model.training
