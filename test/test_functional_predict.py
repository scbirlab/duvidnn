

import torch

from duvidnn.invoke import ModelInvoker
from duvidnn.invoke.functional import functional_predict
from duvidnn.mapping import ColumnMap


def test_functional_predict():

    from torch import nn

    torch.manual_seed(0)

    model = nn.Linear(2, 1)

    invoker = ModelInvoker(
        model=model,
        input_map=ColumnMap(
            inputs={"input": "x"},
        ),
    )

    batch = {
        "x": torch.tensor([
            [1., 2.],
            [3., 4.],
        ]),
    }

    expected = invoker.predict(batch)
    observed = functional_predict(
        model=model,
        invoker=invoker,
        batch=batch,
        params=dict(model.named_parameters()),
    )

    assert torch.allclose(observed, expected)


def test_functional_predict_named_composition():
    
    from duvidnn.models import (
        HillCurve,
        MLP,
        Readout,
        TwoTower,
    )

    torch.manual_seed(0)

    model = Readout(
        latent=TwoTower(
            left=MLP(
                input_dim=2,
                output_dim=3,
                hidden_dims=4,
            ),
            right=MLP(
                input_dim=1,
                output_dim=2,
                hidden_dims=4,
            ),
            fusion=MLP(
                input_dim=5,
                output_dim=1,
                hidden_dims=4,
            ),
        ),
        readout=HillCurve(
            slope=1.,
            trainable_slope=True,
        ),
    )

    invoker = ModelInvoker(
        model=model,
        input_map=ColumnMap(
            inputs={
                "left": "left_features",
                "right": "right_features",
                "context": "concentration",
            },
        ),
    )

    batch = {
        "left_features": torch.tensor([
            [1., 2.],
            [3., 4.],
        ]),
        "right_features": torch.tensor([
            [1.],
            [2.],
        ]),
        "concentration": torch.tensor([
            [.5],
            [2.],
        ]),
    }

    expected = invoker.predict(batch)

    observed = functional_predict(
        model=model,
        invoker=invoker,
        batch=batch,
    )

    assert torch.allclose(observed, expected)

    params = {
        name: parameter
        for name, parameter
        in model.named_parameters()
    }

    name = next(iter(params))

    modified = dict(params)
    modified[name] = modified[name] + 1.

    changed = functional_predict(
        model=model,
        invoker=invoker,
        batch=batch,
        params=modified,
    )

    assert not torch.allclose(changed, expected)
    assert torch.allclose(invoker.predict(batch), expected)

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    changed = functional_predict(
        model=model,
        invoker=invoker,
        batch=batch,
        params=modified,
    )

    assert not torch.allclose(changed, expected)
    assert torch.allclose(
        invoker.predict(batch),
        expected,
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
