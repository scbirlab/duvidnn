"""Integration test for Aspect Chemprop collation into duvidnn."""

import torch
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit import Chem

from aspect.collate.chemprop import chemprop_collate

from duvidnn.invoke import ModelInvoker
from duvidnn.mapping import ColumnMap
from duvidnn.models.chemprop import ChempropEncoder

from utils.data import _make_chemprop_rows


def test_chemprop_collate_to_model():
    chemprop_batch = chemprop_collate(_make_chemprop_rows())

    batch = {
        "chemprop": chemprop_batch,
        "target": torch.tensor([1.0, 2.0]),
    }

    model = ChempropEncoder(
        output_dim=1,
    )

    column_map = ColumnMap(
        inputs={
            "input": "chemprop",
        },
        target="target",
    )

    invoker = ModelInvoker(
        model=model,
        input_map=column_map,
    )

    prediction, target = invoker.supervised(batch)

    assert prediction.shape == (2, 1)
    assert torch.equal(
        target,
        torch.tensor([1.0, 2.0]),
    )
