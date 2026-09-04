def test_lightning_task_metric_mask():
    import torch
    from torch import nn
    from torchmetrics.regression import MeanAbsoluteError

    from duvidnn.invoke import ModelInvoker
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
        loss=nn.MSELoss(),
        optimizer=torch.optim.Adam,
        metrics={"mae": MeanAbsoluteError()},
        metric_mask=lambda batch: (batch["censor"] == 0),
    )

    batch = {
        "x": torch.tensor([
            [1.],
            [100.],
        ]),
        "y": torch.tensor([
            [2.],
            [0.],
        ]),
        "censor": torch.tensor([
            [0],
            [1],
        ]),
    }

    ptl = task._prediction_target_loss(batch)

    task._update_metrics(
        prediction=ptl.prediction,
        target=ptl.target,
        batch=batch,
        stage="train",
    )

    observed = (
        task.train_metrics["mae"]
        .compute()
    )

    torch.testing.assert_close(observed, torch.ones_like(observed))
