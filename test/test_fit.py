from torch import nn

def test_box_fit():
    import torch

    from aspect import DataPipeline

    from duvidnn import Box
    from duvidnn.mapping import ColumnMap
    from duvidnn.training import Trainer
    torch.manual_seed(0)

    pipeline = DataPipeline({
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


def test_box_fit_hill_readout():
    import torch

    from aspect import DataPipeline

    from duvidnn import Box
    from duvidnn.mapping import ColumnMap
    from duvidnn.models import HillCurve, MLP, Readout
    from duvidnn.training import Trainer

    torch.manual_seed(0)

    pipeline = DataPipeline({
        "features": ("features_raw", "identity"),
        "concentration": ("concentration_raw", "identity"),
        "labels": ("labels_raw", "identity"),
    })

    model = Readout(
        latent=MLP(
            input_dim=2,
            output_dim=1,
            hidden_dims=4,
        ),
        readout=HillCurve(slope=1.),
    )

    box = Box(
        model=model,
        pipeline=pipeline,
        input_map=ColumnMap(
            inputs={
                "x": "features",
                "context": "concentration",
            },
            target="labels",
        ),
        trainer=Trainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": 0.05},
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        ),
    )

    data = {
        "features_raw": [
            [0., 0.],
            [0., 0.],
            [1., 1.],
            [1., 1.],
        ],
        "concentration_raw": [
            [0.5],
            [2.0],
            [0.5],
            [2.0],
        ],
        "labels_raw": [
            [.2],
            [.7],
            [.1],
            [.5],
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

    batch = next(iter(pipeline.dataloader(batch_size=4)))
    prediction = box.predict_batch(batch)

    assert prediction.shape == (4, 1)
    assert torch.all(prediction >= 0.)
    assert torch.all(prediction <= 1.)


def test_box_fit_hill_readout_trainable_slope():
    import torch

    from aspect import DataPipeline

    from duvidnn import Box
    from duvidnn.mapping import ColumnMap
    from duvidnn.models import HillCurve, MLP, Readout
    from duvidnn.training import Trainer

    torch.manual_seed(0)

    pipeline = DataPipeline({
        "features": ("features_raw", "identity"),
        "concentration": ("concentration_raw", "identity"),
        "labels": ("labels_raw", "identity"),
    })

    model = Readout(
        latent=MLP(
            input_dim=2,
            output_dim=1,
            hidden_dims=4,
        ),
        readout=HillCurve(
            slope=1., 
            trainable_slope=True,
        ),
    )

    box = Box(
        model=model,
        pipeline=pipeline,
        input_map=ColumnMap(
            inputs={
                "x": "features",
                "context": "concentration",
            },
            target="labels",
        ),
        trainer=Trainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": 0.05},
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        ),
    )

    data = {
        "features_raw": [
            [0., 0.],
            [0., 0.],
            [1., 1.],
            [1., 1.],
        ],
        "concentration_raw": [
            [0.5],
            [2.0],
            [0.5],
            [2.0],
        ],
        "labels_raw": [
            [.2],
            [.7],
            [.1],
            [.5],
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

    batch = next(iter(pipeline.dataloader(batch_size=4)))
    prediction = box.predict_batch(batch)

    assert prediction.shape == (4, 1)
    assert torch.all(prediction >= 0.)
    assert torch.all(prediction <= 1.)


class CategoricalEncoder(nn.Module):
        def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
        ):
            super().__init__()

            self.embedding = nn.Embedding(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
            )

        def forward(self, x):
            return self.embedding(x.squeeze(-1).long())


def test_box_fit_heterogeneous_two_tower():
    import torch

    from aspect import DataPipeline

    from duvidnn import Box
    from duvidnn.mapping import ColumnMap
    from duvidnn.models import TwoTower
    from duvidnn.training import Trainer

    torch.manual_seed(0)
    pipeline = DataPipeline({
        "continuous": ("continuous_raw", "identity"),
        "category": ("category_raw", "identity"),
        "labels": ("labels_raw", "identity"),
    })

    model = TwoTower(
        left=nn.Linear(2, 3),
        right=CategoricalEncoder(
            num_embeddings=3,
            embedding_dim=2,
        ),
        fusion=nn.Linear(5, 1),
        merge="concat",
    )

    box = Box(
        model=model,
        pipeline=pipeline,
        input_map=ColumnMap(
            inputs={
                "left": "continuous",
                "right": "category",
            },
            target="labels",
        ),
        trainer=Trainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": .05},
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        ),
    )

    data = {
        "continuous_raw": [
            [0., 1.],
            [1., 0.],
            [1., 1.],
            [2., 1.],
        ],
        "category_raw": [
            [0],
            [1],
            [2],
            [0],
        ],
        "labels_raw": [
            [1.],
            [2.],
            [3.],
            [4.],
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
    print(before.keys())
    assert not torch.equal(before["towers.left.weight"], model.left.weight)
    assert not torch.equal(
        before["towers.right.embedding.weight"],
        model.right.embedding.weight,
    )
    assert not torch.equal(
        before["fusion.weight"],
        model.fusion.weight,
    )
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter
        in model.named_parameters()
    )

    batch = next(iter(pipeline.dataloader(batch_size=4)))
    prediction = box.predict_batch(batch)

    assert prediction.shape == (4, 1)
