import torch
from torch import nn
import pytest

from duvidnn.invoke import ModelInvoker
from duvidnn.mapping import ColumnMap
from duvidnn.training import LightningTask

def test_lightning_task_training_step():

    model = nn.Linear(1, 1)

    invoker = ModelInvoker(
        model=model,
        input_map=ColumnMap(
            inputs={"input": "x"},
            target="y",
        ),
    )

    task = LightningTask(
        model=model,
        invoker=invoker,
        loss=nn.MSELoss(),
        optimizer=torch.optim.Adam,
    )

    batch = {
        "x": torch.tensor([
            [1.],
            [2.],
        ]),
        "y": torch.tensor([
            [2.],
            [4.],
        ]),
    }

    loss = task.loss(batch)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_lightning_task_loss():
    import torch
    from torch import nn

    from duvidnn.invoke import ModelInvoker
    from duvidnn.mapping import ColumnMap
    from duvidnn.training import LightningTask

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.)

    invoker = ModelInvoker(
        model=model,
        input_map=ColumnMap(
            inputs={"input": "x"},
            target="y",
        ),
    )

    task = LightningTask(
        model=model,
        invoker=invoker,
        loss=nn.MSELoss(),
        optimizer=torch.optim.Adam,
    )

    batch = {
        "x": torch.tensor([
            [1.],
            [2.],
        ]),
        "y": torch.tensor([
            [2.],
            [4.],
        ]),
    }

    loss = task.loss(batch)
    torch.testing.assert_close(loss, torch.tensor(0.))


def test_lightning_task_censored_loss():
    import torch
    from torch import nn

    from duvidnn.invoke import ModelInvoker
    from duvidnn.training.losses import CensoredMSELoss
    from duvidnn.mapping import ColumnMap
    from duvidnn.training import LightningTask

    model = nn.Identity()

    invoker = ModelInvoker(
        model=model,
        input_map=ColumnMap(
            inputs={"input": "x"},
            target="y",
        ),
    )

    task = LightningTask(
        model=model,
        invoker=invoker,
        loss=CensoredMSELoss(),
        optimizer=torch.optim.Adam,
        loss_inputs={
            "censor": "censor",
        },
    )

    batch = {
        "x": torch.tensor([
            [10.],
            [2.],
        ]),
        "y": torch.tensor([
            [5.],
            [5.],
        ]),
        "censor": torch.tensor([
            [1],
            [1],
        ]),
    }

    loss = task.loss(batch)

    torch.testing.assert_close(loss, torch.tensor(4.5))


def test_l1_regularizer_changes_loss():

    from duvidnn.training import (
        L1Regularizer,
        LightningTask,
    )

    model = nn.Linear(2, 1, bias=False)

    with torch.no_grad():
        model.weight.fill_(1.)

    invoker = ModelInvoker(
        model=model,
        input_map=ColumnMap(
            inputs={
                "input": "x",
            },
            target="y",
        ),
    )

    batch = {
        "x": torch.zeros(2, 2),
        "y": torch.zeros(2, 1),
    }

    plain = LightningTask(
        model=model,
        invoker=invoker,
        loss=nn.MSELoss(),
        optimizer=torch.optim.Adam,
    )

    regularized = LightningTask(
        model=model,
        invoker=invoker,
        loss=nn.MSELoss(),
        optimizer=torch.optim.Adam,
        regularizer=L1Regularizer(
            weight=0.5,
        ),
    )

    assert plain.loss(batch).item() == 0.
    assert regularized.loss(batch).item() == pytest.approx(1.)


def test_reduce_lr_on_plateau_configuration():

    model = nn.Linear(2, 1)

    invoker = ModelInvoker(
        model=model,
        input_map=ColumnMap(
            inputs={
                "input": "x",
            },
            target="y",
        ),
    )

    task = LightningTask(
        model=model,
        invoker=invoker,
        loss=nn.MSELoss(),
        optimizer=torch.optim.Adam,
        scheduler=(
            torch.optim.lr_scheduler
            .ReduceLROnPlateau
        ),
        scheduler_kwargs={"patience": 3},
        scheduler_monitor="val_loss",
    )

    config = task.configure_optimizers()

    assert isinstance(
        config["lr_scheduler"]["scheduler"],
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    )
    assert config["lr_scheduler"]["monitor"] == "val_loss"


def test_step_lr_configuration():

    model = nn.Linear(2, 1)

    invoker = ModelInvoker(
        model=model,
        input_map=ColumnMap(
            inputs={
                "input": "x",
            },
            target="y",
        ),
    )

    task = LightningTask(
        model=model,
        invoker=invoker,
        loss=nn.MSELoss(),
        optimizer=torch.optim.Adam,
        scheduler=(
            torch.optim.lr_scheduler
            .StepLR
        ),
        scheduler_kwargs={
            "step_size": 5,
            "gamma": .5,
        },
    )

    config = task.configure_optimizers()
    scheduler_config = config["lr_scheduler"]

    assert isinstance(
        scheduler_config["scheduler"],
        torch.optim.lr_scheduler.StepLR,
    )

    assert "monitor" not in scheduler_config
