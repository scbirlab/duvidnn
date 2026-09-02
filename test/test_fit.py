def test_box_fit():
    import torch
    from torch import nn

    from aspect import DataPipeline

    from duvidnn import Box
    from duvidnn.mapping import ColumnMap
    from duvidnn.training import Trainer

    torch.manual_seed(0)

    pipeline = DataPipeline({
        "x": ("x_raw", "identity"),
        "labels": ("y_raw", "identity"),
    })

    model = nn.Linear(1, 1)

    box = Box(
        model=model,
        pipeline=pipeline,
        input_map=ColumnMap(
            inputs={"input": "x"},
            target="labels",
        ),
        trainer=Trainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": 0.05},
            max_epochs=2,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        ),
    )

    data = {
        "x_raw": [
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

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    observed = box.fit(
        data,
        batch_size=4,
    )

    assert observed is box
    assert any(
        not torch.equal(
            before[name],
            parameter,
        )
        for name, parameter
        in model.named_parameters()
    )


def test_box_fit_with_validation():
    import torch
    from torch import nn

    from aspect import DataPipeline

    from duvidnn import Box
    from duvidnn.mapping import ColumnMap
    from duvidnn.training import Trainer

    pipeline = DataPipeline({
        "x": ("x_raw", "identity"),
        "labels": ("y_raw", "identity"),
    })

    box = Box(
        model=nn.Linear(1, 1),
        pipeline=pipeline,
        input_map=ColumnMap(
            inputs={"input": "x"},
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

    train = {
        "x_raw": [
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

    validation = {
        "x_raw": [
            [4.],
            [5.],
        ],
        "y_raw": [
            [9.],
            [11.],
        ],
    }

    observed = box.fit(
        train,
        validation=validation,
        batch_size=4,
    )

    assert observed is box
