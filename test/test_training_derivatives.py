import torch
from torch import nn

from aspect import DataPipeline

from duvidnn import Box
from duvidnn.invoke import TrainingDerivatives
from duvidnn.mapping import ColumnMap
from duvidnn.training import Trainer


def _make_box():

    pipeline = DataPipeline({
        "labels": ("y_raw", "identity"),
    })

    model = nn.Linear(
        1,
        1,
    )

    return Box(
        model=model,
        pipeline=pipeline,
        input_map=ColumnMap(
            inputs={
                "input": "x",
            },
            target="labels",
        ),
        trainer=Trainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={
                "lr": 0.05,
            },
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        ),
    )


def _training_data():

    return {
        "x": [
            [0.],
            [1.],
            [2.],
            [3.],
        ],
        "y_raw": [
            [1.],
            [3.],
            [5.],
            [7.],
        ],
    }


def test_box_fit_invalidates_training_derivatives():

    box = _make_box()

    box.training_derivatives = TrainingDerivatives(
        fisher_score={
            name: torch.ones_like(
                parameter
            )
            for name, parameter
            in box.model.named_parameters()
        },
        n_samples=4,
        loss_reduction="mean",
    )

    assert (
        box.training_derivatives
        is not None
    )

    box.fit(
        _training_data(),
        batch_size=4,
    )

    assert (
        box.training_derivatives
        is None
    )


def test_training_derivatives_checkpoint_roundtrip(
    tmp_path,
):

    box = _make_box()

    box.prepare(
        _training_data()
    )

    observed = box.compute_training_derivatives(
        fisher_score=True,
        batch_size=2,
    )

    assert observed is box

    assert (
        box.training_derivatives
        is not None
    )

    assert (
        box.training_derivatives
        .fisher_score
        is not None
    )

    assert (
        box.training_derivatives
        .n_samples
        == 4
    )

    expected_score = {
        name: value.detach().clone()
        for name, value
        in (
            box.training_derivatives
            .fisher_score
            .items()
        )
    }

    checkpoint = (
        tmp_path
        / "box"
    )

    box.save(
        checkpoint
    )

    assert (
        checkpoint
        / "training_derivatives.pt"
    ).exists()

    restored = Box.load(
        checkpoint
    )

    assert (
        restored.training_derivatives
        is not None
    )

    assert (
        restored.training_derivatives
        .fisher_score
        is not None
    )

    assert (
        restored.training_derivatives
        .n_samples
        == 4
    )

    assert (
        restored.training_derivatives
        .loss_reduction
        == "mean"
    )

    assert (
        set(
            restored.training_derivatives
            .fisher_score
        )
        == set(
            expected_score
        )
    )

    for name, expected in expected_score.items():
        torch.testing.assert_close(
            restored.training_derivatives
            .fisher_score[name],
            expected,
        )


def test_box_save_removes_stale_training_derivatives(
    tmp_path,
):

    box = _make_box()

    box.prepare(
        _training_data()
    )

    box.compute_training_derivatives(
        fisher_score=True,
        batch_size=2,
    )

    checkpoint = (
        tmp_path
        / "box"
    )

    box.save(
        checkpoint
    )

    derivatives_path = (
        checkpoint
        / "training_derivatives.pt"
    )

    assert derivatives_path.exists()

    box.training_derivatives = None

    box.save(
        checkpoint
    )

    assert not derivatives_path.exists()