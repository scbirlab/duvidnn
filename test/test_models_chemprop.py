from dataclasses import asdict

from chemprop.data import (
    MoleculeDatapoint,
    MoleculeDataset,
    build_dataloader,
)
import numpy as np
import torch

from duvidnn.models import ChempropEncoder, MultiTower, MLP


def _make_chemprop_batch():
    data = [
        MoleculeDatapoint.from_smi("CCO"),
        MoleculeDatapoint.from_smi("CCN"),
        MoleculeDatapoint.from_smi("CCC"),
    ]

    dataset = MoleculeDataset(data)
    batch = next(
        iter(
            build_dataloader(
                dataset,
                batch_size=3,
                shuffle=False,
            )
        )
    )
    bmg, V_d, X_d, *_ = batch
    return {
        "bmg": bmg,
        "V_d": V_d,
        "X_d": X_d,
    }


def test_chemprop_encoder():
    
    model = ChempropEncoder(
        output_dim=32,
    )

    x = _make_chemprop_batch()
    y = model(**x)

    assert y.shape == (3, 32)

    y.sum().backward()

    assert any(
        p.grad is not None
        for p in model.message_passing.parameters()
    )


def test_chemprop_multitower():
    model = MultiTower(
        towers={
            "compound": ChempropEncoder(
                output_dim=32,
            ),
            "species": MLP(
                input_dim=128,
                hidden_dims=[64],
                output_dim=32,
            ),
        },
        fusion=MLP(
            hidden_dims=[32],
            output_dim=1,
        ),
        merge="concat",
    )
    chemprop_batch = _make_chemprop_batch()
    prediction = model(
        compound=chemprop_batch,
        species=torch.randn(3, 128),
    )
    assert prediction.shape == (3, 1)

    loss = prediction.sum()
    loss.backward()

    assert any(
        parameter.grad is not None
        for parameter
        in model.towers["compound"].parameters()
    )

    assert any(
        parameter.grad is not None
        for parameter
        in model.towers["species"].parameters()
    )

    assert any(
        parameter.grad is not None
        for parameter
        in model.fusion.parameters()
    )
