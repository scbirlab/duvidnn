import torch
from torch import nn

from duvida import parameter_gradient

from duvidnn.invoke import ModelInvoker, make_stateless_model
from duvidnn.mapping import ColumnMap


def _parameter_gradient(
    model,
    invoker,
    batch
):
    stateless_model = make_stateless_model(model, invoker)

    params = dict(model.named_parameters())

    return parameter_gradient(stateless_model)(
        (params,),
        batch,
    )[0]


def test_duvida_parameter_gradient_linear():

    model = nn.Linear(2, 1)
    invoker = ModelInvoker(
        model=model,
        input_map={"inputs" :{"input": "x"}},
    )

    batch = {
        "x": torch.tensor([
            [1., 2.],
            [3., 4.],
        ]),
    }

    gradients = _parameter_gradient(
        model,
        invoker,
        batch,
    )

    assert gradients["weight"].shape == (2, 1, 2)
    assert gradients["bias"].shape == (2, 1)
    assert torch.allclose(
        gradients["weight"],
        batch["x"][:, None, :],
    )
    assert torch.allclose(
        gradients["bias"],
        torch.ones_like(gradients["bias"])
    )


def test_duvida_parameter_gradient_two_tower():

    from duvidnn.models import MLP, TwoTower

    torch.manual_seed(0)

    model = TwoTower(
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
    )

    invoker = ModelInvoker(
        model=model,
        input_map={"inputs": {
            "left": "left_features",
            "right": "right_features",
        }},
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
    }

    _ = invoker.predict(
        batch
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    gradients = _parameter_gradient(
        model,
        invoker,
        batch,
    )

    assert set(gradients) == set(before)
    for name, gradient in gradients.items():
        assert gradient.shape == (2, *before[name].shape)
        assert torch.all(torch.isfinite(gradient))
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_duvida_parameter_gradient_readout():

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
        input_map={"inputs": {
            "left": "left_features",
            "right": "right_features",
            "context": "concentration",
        }},
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

    _ = invoker.predict(batch)

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }
    gradients = _parameter_gradient(
        model,
        invoker,
        batch,
    )

    assert set(gradients) == set(before)

    for name, gradient in gradients.items():
        assert gradient.shape == (2, *before[name].shape)
        assert torch.all(torch.isfinite(gradient))

    readout_parameters = [
        name
        for name in gradients
        if name.startswith("readout.")
    ]

    assert readout_parameters
    assert any(
        torch.any(gradients[name] != 0.)
        for name in readout_parameters
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
