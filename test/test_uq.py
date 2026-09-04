import torch
from torch import nn

from aspect import DataPipeline
import numpy as np
import pytest

from duvidnn import Box
from duvidnn.mapping import ColumnMap
from duvidnn.training import Trainer
from duvidnn.uncertainty import (
    DoubtScore,
    InformationSensitivity,
    Tanimoto,
    Variance,
    normalize_uncertainty,
)


def test_normalize_uncertainty():

    variance = Variance()
    doubt = DoubtScore()

    assert normalize_uncertainty(variance) == {"variance": variance}
    assert normalize_uncertainty([variance, doubt]) == {
        "variance": variance,
        "doubtscore": doubt,
    }

    assert normalize_uncertainty({"foo": variance}) == {"foo": variance}


def test_variance():

    class EnsembleOutput(nn.Module):
        def forward(self, input):
            return torch.stack(
                (
                    input,
                    input + 1.,
                    input + 2.,
                ),
                dim=-1,
            )

    box = Box(
        model=EnsembleOutput(),
        input_map=ColumnMap(
            inputs={
                "input": "x",
            },
        ),
    )

    observed = box.predict(
        {
            "x": [
                [0.],
                [1.],
            ],
        },
        uncertainty=Variance(dim=-1),
        batch_size=2,
    )

    assert "prediction" in observed.column_names
    assert "variance" in observed.column_names

    expected = torch.tensor([
        [2. / 3.],
        [2. / 3.],
    ])

    torch.testing.assert_close(
        torch.tensor(np.array(observed["variance"])),
        expected,
    )


def test_doubtscore_uses_cached_fisher():

    torch.manual_seed(0)

    box = Box(
        model=nn.Linear(1, 1),
        pipeline=DataPipeline({
            "labels": ("y", "identity"),
        }),
        input_map=ColumnMap(
            inputs={
                "input": "x",
            },
            target="labels",
        ),
        trainer=Trainer(
            loss=nn.MSELoss(),
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        ),
    )

    training = {
        "x": [
            [0.],
            [1.],
            [2.],
            [3.],
        ],
        "y": [
            [1.],
            [3.],
            [5.],
            [7.],
        ],
    }

    box.prepare(training)
    box.compute_training_derivatives(
        fisher_score=True,
        batch_size=2,
    )
    cached = box.training_derivatives.fisher_score

    observed = box.predict(
        {
            "x": [
                [4.],
                [5.],
            ],
            "y": [
                [9.],
                [11.],
            ],
        },
        uncertainty=DoubtScore(),
        batch_size=2,
    )

    assert "doubtscore" in observed.column_names
    assert box.training_derivatives.fisher_score is cached


def test_information_sensitivity_uses_cached_derivatives():

    torch.manual_seed(0)

    box = Box(
        model=nn.Linear(1, 1),
        pipeline=DataPipeline({
            "labels": ("y", "identity"),
        }),
        input_map=ColumnMap(
            inputs={
                "input": "x",
            },
            target="labels",
        ),
        trainer=Trainer(
            loss=nn.MSELoss(),
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        ),
    )

    training = {
        "x": [
            [0.],
            [1.],
            [2.],
            [3.],
        ],
        "y": [
            [1.],
            [3.],
            [5.],
            [7.],
        ],
    }

    box.prepare(training)
    box.compute_training_derivatives(
        fisher_score=True,
        fisher_information={
            "approximator": "squared_jacobian",
        },
        batch_size=2,
    )

    observed = box.predict(
        {
            "x": [
                [4.],
                [5.],
            ],
            "y": [
                [9.],
                [11.],
            ],
        },
        uncertainty=InformationSensitivity(
            approximator="squared_jacobian",
        ),
        batch_size=2,
    )

    assert "information_sensitivity" in observed.column_names
    assert len(observed["information_sensitivity"]) == 2


def test_tanimoto():

    box = Box(
        model=nn.Identity(),
        input_map=ColumnMap(
            inputs={"input": "fingerprint"},
        ),
    )

    training = {
        "fingerprint": [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
    }

    prepared = box.prepare(training)

    assert "fingerprint" in prepared.column_names

    observed = box.predict(
        {
            "fingerprint": [
                training["fingerprint"][0],
                [1, 0, 1, 0],
            ],
        },
        uncertainty=Tanimoto(column="fingerprint"),
    )

    assert observed["tanimoto"][0] == pytest.approx(1.)
    assert observed["tanimoto"][1] == pytest.approx(1 / 3)


def test_tanimoto_derives_fingerprint_from_smiles():

    from duvidnn.models import ChempropEncoder

    box = Box(
        model=ChempropEncoder(
            output_dim=1,
            mp_hidden_dim=16,
            mp_depth=1,
            hidden_dims=8,
        ),
        pipeline=DataPipeline({"cp_in": ("smiles", "chemprop-mol")}),
        input_map=ColumnMap(
            inputs={"input": "cp_in"},
        ),
    )

    training = {
        "smiles": ["CCO", "c1ccccc1"],
    }

    box.prepare(training)

    observed = box.predict(
        {
            "smiles": [training["smiles"][0], "CCN"],
        },
        uncertainty=Tanimoto(),
    )

    assert "tanimoto" in observed.column_names
    assert observed["tanimoto"][0] == pytest.approx(1.)
    assert 0 <= observed["tanimoto"][1] <= 1


def test_tanimoto_does_not_mutate_box_pipeline():

    from duvidnn.models import ChempropEncoder

    box = Box(
        model=ChempropEncoder(
            output_dim=1,
            mp_hidden_dim=16,
            mp_depth=1,
            hidden_dims=8,
        ),
        pipeline=DataPipeline({"cp_in": ("smiles", "chemprop-mol")}),
        input_map=ColumnMap(
            inputs={"input": "cp_in"},
        ),
    )

    training = {
        "smiles": ["CCO", "c1ccccc1"],
    }

    box.prepare({
        "smiles": [
            "CCO",
            "CCN",
        ],
    })

    original = (
        box.pipeline
        .to_config()
    )

    box.predict(
        {"smiles": training["smiles"][:1]},
        uncertainty=Tanimoto(),
    )

    assert box.pipeline.to_config() == original
