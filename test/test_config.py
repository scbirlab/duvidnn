"""Tests for declarative model construction."""

from torch import nn

from aspect.collate.chemprop import chemprop_collate

from jsonargparse import ArgumentParser

from duvidnn.models.chemprop import ChempropEncoder
from duvidnn.models.composition import TwoTower
from duvidnn.models.mlp import MLP

from utils.data import _make_chemprop_rows


def _get_config():
    return {
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
    }

def test_construct_two_tower_from_config():
    parser = ArgumentParser()

    parser.add_subclass_arguments(
        nn.Module,
        "model",
    )

    config = _get_config()
    parsed = parser.parse_object(config)
    instantiated = parser.instantiate(parsed)

    model = instantiated.model

    assert isinstance(model, TwoTower)
    assert isinstance(model.left, ChempropEncoder)
    assert isinstance(model.right, MLP)
    assert isinstance(model.fusion, MLP)

    assert model.left.output_dim == 8
    assert model.right.input_dim == 4
    assert model.right.output_dim == 8
    assert model.fusion.input_dim == 16
    assert model.fusion.output_dim == 1
    assert model.merge == "concat"


def test_constructed_two_tower_runs():
    import torch

    parser = ArgumentParser()
    parser.add_subclass_arguments(
        nn.Module,
        "model",
    )
    config = _get_config()

    parsed = parser.parse_object(config)
    instantiated = parser.instantiate(parsed)

    model = instantiated.model

    vectome = torch.tensor(
        [
            [.1, .2, .3, .4],
            [.5, .6, .7, .8],
        ],
        dtype=torch.float32,
    )

    prediction = model(
        left=chemprop_collate(_make_chemprop_rows()),
        right=vectome,
    )

    assert prediction.shape == (2, 1)