import torch

from aspect.data import DataPipeline

from duvidnn.box import Box
from duvidnn.mapping import ColumnMap
from duvidnn.models.mlp import MLP
import numpy as np


def test_experiment_composes_model_and_mapping():

    box = Box(
        model=MLP(
            input_dim=2,
            hidden_dims=4,
            output_dim=1,
        ),
        input_map=ColumnMap(
            inputs={"x": "input_x"},
            target="y",
        ),
    )

    batch = {
        "input_x": torch.tensor([
            [1., 2.],
            [3., 4.],
        ]),
        "y": torch.tensor([1., 2.]),
    }

    prediction, target = box.supervised_batch(batch)

    assert prediction.shape == (2, 1)
    assert torch.equal(target, batch["y"])


def test_box_prepares_data_with_pipeline():
    pipeline = DataPipeline(
        column_transforms={
            "renamed_x": ("x", "identity"),
            "logx": ("x", "log"),
        },
    )

    box = Box(
        pipeline=pipeline,
        model=MLP(
            input_dim=2,
            hidden_dims=4,
            output_dim=1,
        ),
        input_map=ColumnMap(
            inputs={"x": "logx"},
            target="y",
        ),
    )
    raw_data = {
        "x": [
            [1., 2.],
            [3., 4.],
        ],
        "y": [1., 2.],
    }

    prepared = box.prepare(raw_data)

    assert "logx" in prepared.column_names
    assert "renamed_x" in prepared.column_names
