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
    invoker = ModelInvoker(
        model=ChempropEncoder(output_dim=1),
        input_map=ColumnMap(
            inputs={"input": "chemprop"},
            target="target",
        ),
    )

    chemprop_batch = chemprop_collate(_make_chemprop_rows())
    batch = {
        "chemprop": chemprop_batch,
        "target": torch.tensor([1., 2.]),
    }
    prediction, target = invoker.supervised(batch)

    assert prediction.shape == (2, 1)
    assert torch.equal(target, batch["target"])
