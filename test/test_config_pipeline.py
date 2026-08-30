from aspect.collate.chemprop import chemprop_collate
from jsonargparse import ArgumentParser
from torch import nn

from duvidnn.invoke import ModelInvoker
from duvidnn.mapping import ColumnMap

from utils.data import _make_chemprop_rows, _make_vectome_rows


CONFIG = {
    "model": {
        "class_path": "duvidnn.models.composition.TwoTower",
        "init_args": {
            "left": {
                "class_path": "duvidnn.models.chemprop.ChempropEncoder",
                "init_args": {
                    "output_dim": 8,
                },
            },
            "right": {
                "class_path": "duvidnn.models.mlp.MLP",
                "init_args": {
                    "input_dim": 4,
                    "hidden_dims": 8,
                    "output_dim": 8,
                },
            },
            "fusion": {
                "class_path": "duvidnn.models.mlp.MLP",
                "init_args": {
                    "input_dim": 16,
                    "hidden_dims": 8,
                    "output_dim": 1,
                },
            },
            "merge": "concat",
        },
    },
    "column_map": {
        "inputs": {
            "left": "chemprop",
            "right": "vectome",
        },
        "target": "mic",
    },
}


def test_construct_model_and_mapping_from_config():
    parser = ArgumentParser()

    parser.add_subclass_arguments(
        nn.Module,
        "model",
    )
    parser.add_class_arguments(
        ColumnMap,
        "column_map",
    )
    parsed = parser.parse_object(CONFIG)
    instantiated = parser.instantiate(parsed)

    assert isinstance(
        instantiated.column_map,
        ColumnMap,
    )
    assert instantiated.column_map.inputs == {
        "left": "chemprop",
        "right": "vectome",
    }
    assert instantiated.column_map.target == "mic"


def test_configured_model_pipeline():
    import torch

    parser = ArgumentParser()
    parser.add_subclass_arguments(
        nn.Module,
        "model",
    )
    parser.add_class_arguments(
        ColumnMap,
        "column_map",
    )
    parsed = parser.parse_object(CONFIG)
    instantiated = parser.instantiate(parsed)

    model = instantiated.model
    column_map = instantiated.column_map

    batch = {
        "chemprop": chemprop_collate(_make_chemprop_rows()),
        "vectome": _make_vectome_rows(),
        "mic": torch.tensor([1., 2.]),
    }

    invoker = ModelInvoker(
        model=model,
        input_map=column_map,
    )
    prediction, observed = invoker.supervised(batch)

    assert prediction.shape == (2, 1)
    assert torch.equal(observed, batch["mic"])
