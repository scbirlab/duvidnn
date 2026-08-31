from aspect.data import DataPipeline
from jsonargparse import ArgumentParser

from duvidnn.box import Box
from duvidnn.mapping import ColumnMap
from duvidnn.models.mlp import MLP


BOX_CONFIG = {
    "box": {
        "pipeline": {
            "logx": ("x", "log"),
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
            "inputs": {
                "x": "logx",
            },
            "target": "y",
        },
    },
}


def test_box_from_config():
    parser = ArgumentParser()
    parser.add_class_arguments(
        Box,
        "box",
    )
    parsed = parser.parse_object(
        BOX_CONFIG,
    )
    instantiated = parser.instantiate(
        parsed,
    )
    box = instantiated.box

    assert isinstance(box, Box)
    assert isinstance(box.pipeline, DataPipeline)
    assert isinstance(box.model, MLP)
    assert isinstance(box.input_map, ColumnMap)

    assert box.input_map.inputs == {
        "x": "logx",
    }
    assert box.input_map.target == "y"

    prepared = box.prepare({
        "x": [
            [1., 2.],
            [3., 4.],
        ],
        "y": [1.,2.,],
    })

    assert "logx" in prepared.column_names
    assert "y" in prepared.column_names
