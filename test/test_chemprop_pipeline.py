"""Integration test for Aspect Chemprop collation into duvidnn."""

import torch
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from rdkit import Chem

from aspect.collate.chemprop import chemprop_collate

from duvidnn.invoke import ModelInvoker
from duvidnn.mapping import ColumnMap
from duvidnn.models.chemprop import ChempropEncoder


def _cached_molgraph(smiles: str) -> dict:
    """Create an Arrow/HF-like serialized Chemprop MolGraph."""
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    molgraph = featurizer(
        Chem.MolFromSmiles(smiles)
    )

    return {
        "V": molgraph.V.tolist(),
        "E": molgraph.E.tolist(),
        "edge_index": molgraph.edge_index.tolist(),
        "rev_edge_index": molgraph.rev_edge_index.tolist(),
    }


def test_chemprop_collate_to_model():
    chemprop_batch = chemprop_collate(
        [
            {
                "bmg": _cached_molgraph("CCO"),
                "V_d": None,
                "X_d": None,
            },
            {
                "bmg": _cached_molgraph("CCN"),
                "V_d": None,
                "X_d": None,
            },
        ]
    )

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
