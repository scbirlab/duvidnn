import pytest
import torch

from duvidnn.mapping import ColumnMap


def test_input_map():
    mapping = ColumnMap(
        inputs={
            "compound": "chemprop",
            "species": "species_features",
        },
        target="mic",
    )

    batch = {
        "chemprop": {"foo": "bar"},
        "species_features": torch.randn(4, 8),
        "mic": torch.randn(4, 1),
        "ignored_metadata": ["a", "b", "c", "d"],
    }

    inputs, target = mapping.map_batch(batch)

    assert set(inputs) == {
        "compound",
        "species",
    }

    assert inputs["compound"] == {
        "foo": "bar",
    }

    assert torch.equal(
        inputs["species"],
        batch["species_features"],
    )

    assert torch.equal(
        target,
        batch["mic"],
    )


def test_input_map_missing_input():
    mapping = ColumnMap(
        inputs={
            "compound": "chemprop",
        },
        target="mic",
    )

    batch = {
        "mic": torch.randn(4, 1),
    }

    with pytest.raises(KeyError):
        mapping.map_inputs(batch)


def test_input_map_missing_target():
    mapping = ColumnMap(
        inputs={
            "compound": "chemprop",
        },
        target="mic",
    )

    batch = {
        "chemprop": {},
    }

    with pytest.raises(KeyError):
        mapping.map_target(batch)
