import json

from duvidnn.box import Box
from duvidnn.mapping import ColumnMap

BOX_CONFIG = {
    "pipeline": {
        "logx": ["x", "log"],
    },
    "model": {
        "class_path": "duvidnn.models.mlp.MLP",
        "init_args": {
            "input_dim": 2,
            "hidden_dims": 4,
            "output_dim": 1,
        },
    },
    "input_map": {
        "inputs": {"x": "logx"},
        "target": "y",
    },
}


TWO_TOWER_CONFIG = {
    "pipeline": {
        "left": ["left", "identity"],
        "right": ["right", "identity"],
    },
    "model": {
        "class_path": "duvidnn.models.composition.TwoTower",
        "init_args": {
            "left": {
                "class_path": "duvidnn.models.mlp.MLP",
                "init_args": {
                    "input_dim": 2,
                    "hidden_dims": 4,
                    "output_dim": 3,
                },
            },
            "right": {
                "class_path": "duvidnn.models.mlp.MLP",
                "init_args": {
                    "input_dim": 2,
                    "hidden_dims": 4,
                    "output_dim": 3,
                },
            },
            "fusion": {
                "class_path": "duvidnn.models.mlp.MLP",
                "init_args": {
                    "input_dim": 6,
                    "hidden_dims": 4,
                    "output_dim": 1,
                },
            },
            "merge": "concat",
        },
    },
    "input_map": {
        "inputs": {
            "left": "left",
            "right": "right",
        },
        "target": "y",
    },
}


def test_box_from_config():
    from aspect.data import DataPipeline
    from duvidnn.models.mlp import MLP

    box = Box.from_config(BOX_CONFIG)

    assert isinstance(box, Box)
    assert isinstance(box.pipeline, DataPipeline)
    assert isinstance(box.model, MLP)
    assert isinstance(box.input_map, ColumnMap)

    assert box.model_config == BOX_CONFIG["model"]
    assert box.input_map.inputs == BOX_CONFIG["input_map"]["inputs"]
    assert box.input_map.target == BOX_CONFIG["input_map"]["target"]

    prepared = box.prepare({
        "x": [
            [1., 2.],
            [3., 4.],
        ],
        "y": [1., 2.],
    })
    assert "logx" in prepared.column_names
    assert "y" in prepared.column_names


def test_box_config_is_json_serializable():
    box = Box.from_config(BOX_CONFIG)

    config = box.to_config()
    serialized = json.dumps(config)
    restored = json.loads(serialized)

    assert restored["model"] == BOX_CONFIG["model"]
    assert restored["input_map"] == BOX_CONFIG["input_map"]


def test_configured_box_checkpoint_roundtrip(tmp_path):
    import torch
    box = Box.from_config(BOX_CONFIG)
    x = torch.tensor([
        [1., 2.],
        [3., 4.],
    ])

    expected = box.model(x).detach()
    checkpoint = tmp_path / "box"
    box.save(checkpoint)

    assert (checkpoint / "config.json").exists()
    assert (checkpoint / "weights.pt").exists()
    assert not (checkpoint / "model.pt").exists()

    restored = Box.load(checkpoint)
    observed = restored.model(x).detach()

    torch.testing.assert_close(observed,expected)
    assert restored.model_config == box.model_config
    assert restored.input_map == box.input_map


def test_nested_model_from_config():
    from duvidnn.models.mlp import MLP
    from duvidnn.models.composition import TwoTower
    
    box = Box.from_config(TWO_TOWER_CONFIG)

    assert isinstance(box.model, TwoTower)
    assert isinstance(box.model.left, MLP)
    assert isinstance(box.model.right, MLP)
    assert isinstance(box.model.fusion, MLP)
    assert box.model_config == TWO_TOWER_CONFIG["model"]


def test_nested_model_checkpoint_roundtrip(tmp_path):
    import torch
    from duvidnn.models.composition import TwoTower
    box = Box.from_config(TWO_TOWER_CONFIG)

    left = torch.tensor([
        [1., 2.],
        [3., 4.],
    ])
    right = torch.tensor([
        [5., 6.],
        [7., 8.],
    ])

    expected = box.model(
        left=left,
        right=right,
    ).detach()

    checkpoint = tmp_path / "two-tower"
    box.save(checkpoint)

    restored = Box.load(checkpoint)

    observed = restored.model(
        left=left,
        right=right,
    ).detach()

    torch.testing.assert_close(observed, expected)


def test_opaque_box_checkpoint_roundtrip(tmp_path):
    import torch
    from torch import nn
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )

    box = Box(
        model=model,
        pipeline={
            "x_identity": ["x", "identity"],
        },
        input_map=ColumnMap(
            inputs={"input": "x_identity"},
            target="y",
        ),
    )

    assert box.model_config is None

    x = torch.tensor([
        [1., 2.],
        [3., 4.],
    ])

    expected = box.model(x).detach()

    checkpoint = tmp_path / "opaque"
    box.save(checkpoint)

    assert (checkpoint / "config.json").exists()
    assert (checkpoint / "model.pt").exists()
    assert not (checkpoint / "weights.pt").exists()

    restored = Box.load(checkpoint)
    observed = restored.model(x).detach()

    torch.testing.assert_close(observed, expected)
    assert restored.model_config is None
