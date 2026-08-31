"""Integration test for heterogeneous Chemprop × tensor TwoTower input."""

import torch

from aspect.collate.chemprop import chemprop_collate

from duvidnn.invoke import ModelInvoker
from duvidnn.mapping import ColumnMap
from duvidnn.models.chemprop import ChempropEncoder
from duvidnn.models.composition import TwoTower
from duvidnn.models.mlp import MLP

from utils.data import _make_chemprop_rows


def test_chemprop_vectome_two_tower():
    chemprop_batch = chemprop_collate(_make_chemprop_rows())

    vectome = torch.tensor(
        [
            [.1, .2, .3, .4],
            [.5, .6, .7, .8],
        ],
        dtype=torch.float32,
    )

    target = torch.tensor(
        [1., 2.],
        dtype=torch.float32,
    )

    batch = {
        "chemprop": chemprop_batch,
        "vectome": vectome,
        "mic": target,
    }

    model = TwoTower(
        left=ChempropEncoder(
            output_dim=8,
        ),
        right=MLP(
            input_dim=4,
            hidden_dims=8,
            output_dim=8,
        ),
        fusion=MLP(
            input_dim=16,
            hidden_dims=8,
            output_dim=1,
        ),
        merge="concat",
    )

    column_map = ColumnMap(
        inputs={
            "left": "chemprop",
            "right": "vectome",
        },
        target="mic",
    )

    invoker = ModelInvoker(
        model=model,
        input_map=column_map,
    )

    prediction, observed = invoker.supervised(batch)

    assert prediction.shape == (2, 1)
    assert torch.equal(
        observed,
        target,
    )
