

def test_lightning_task_training_step():
    import torch
    from torch import nn

    from duvidnn.invoke import ModelInvoker
    from duvidnn.mapping import ColumnMap
    from duvidnn.training import LightningTask

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
