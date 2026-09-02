def test_box_fit_heterogeneous_chemprop_hill_readout():
    import torch
    from torch import nn

    from aspect import DataPipeline

    from duvidnn import Box
    from duvidnn.mapping import ColumnMap
    from duvidnn.models import (
        ChempropEncoder,
        HillCurve,
        MLP,
        Readout,
        TwoTower,
    )
    from duvidnn.training import Trainer

    torch.manual_seed(0)

    def parameters(module):
        return {
            name: parameter.detach().clone()
            for name, parameter
            in module.named_parameters()
        }

    pipeline = DataPipeline({
        "molecule": ("smiles", "chemprop-mol"),
    })

    latent = TwoTower(
        left=ChempropEncoder(
            output_dim=4,
            mp_hidden_dim=16,
            mp_depth=1,
            hidden_dims=8,
        ),
        right=MLP(
            input_dim=2,
            output_dim=3,
            hidden_dims=4,
        ),
        fusion=MLP(
            input_dim=7,
            output_dim=1,
            hidden_dims=4,
        ),
    )

    model = Readout(
        latent=latent,
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
                "left": "molecule",
                "right": "features",
                "context": "concentration",
            },
            target="labels",
        ),
        trainer=Trainer(
            loss=nn.MSELoss(),
            optimizer_kwargs={"lr": .01},
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        ),
    )

    data = {
        "smiles": [
            "CCO",
            "CCN",
            "CC(=O)O",
            "c1ccccc1",
        ],
        "features": [
            [0., 0.],
            [0., 1.],
            [1., 0.],
            [1., 1.],
        ],
        "concentration": [
            [0.5],
            [1.0],
            [2.0],
            [4.0],
        ],
        "labels": [
            [.1],
            [.3],
            [.6],
            [.8],
        ],
    }

    before_chemprop = parameters(model.latent.left)
    before_right = parameters(model.latent.right)
    before_fusion = parameters(model.latent.fusion)

    assert "molecule" in pipeline.collators

    observed = box.fit(
        data,
        batch_size=4,
    )

    assert observed is box

    assert any(
        not torch.equal(
            before_chemprop[name],
            parameter,
        )
        for name, parameter
        in model.latent.left.named_parameters()
    )
    assert any(
        not torch.equal(
            before_right[name],
            parameter,
        )
        for name, parameter
        in model.latent.right.named_parameters()
    )
    assert any(
        not torch.equal(
            before_fusion[name],
            parameter,
        )
        for name, parameter
        in model.latent.fusion.named_parameters()
    )

    batch = next(iter(pipeline.dataloader(batch_size=4)))

    prediction = box.predict_batch(batch)

    assert prediction.shape == (4, 1)
    assert torch.all(torch.isfinite(prediction))
    assert torch.all(prediction >= 0.)
    assert torch.all(prediction <= 1.)
