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
    inputs = invoker.inputs(batch)
    stateless_model = make_stateless_model(model)
    params = dict(model.named_parameters())

    return parameter_gradient(stateless_model)(
        (params,),
        inputs,
    )[0]


def test_duvida_parameter_gradient_linear():

    model = nn.Linear(2, 1)
    invoker = ModelInvoker(
        model=model,
        input_map={"inputs": {"input": "x"}},
    )

    batch = {
        "x": torch.tensor([
            [1., 2.],
            [3., 4.],
        ]),
    }
    prediction = invoker.predict(batch)

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

    for name, gradient in gradients.items():
        assert gradient.shape == (*prediction.shape, *before[name].shape)
        assert torch.all(torch.isfinite(gradient))
    assert torch.allclose(
        gradients["weight"],
        batch["x"][:, None, None, :],
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

    prediction = invoker.predict(batch)

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
        assert gradient.shape == (*prediction.shape, *before[name].shape)
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

    prediction = invoker.predict(batch)

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
        assert gradient.shape == (*prediction.shape, *before[name].shape)
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


def test_duvida_parameter_gradient_chemprop():

    import torch

    from aspect import DataPipeline
    from duvida import parameter_gradient

    from duvidnn.invoke import (
        ModelInvoker,
        make_stateless_model,
    )
    from duvidnn.mapping import ColumnMap
    from duvidnn.models import (
        ChempropEncoder,
        HillCurve,
        MLP,
        Readout,
        TwoTower,
    )

    torch.manual_seed(0)

    pipeline = DataPipeline({
        "molecule": ("smiles", "chemprop-mol"),
    })

    model = Readout(
        latent=TwoTower(
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
        ),
        readout=HillCurve(
            slope=1.,
            trainable_slope=True,
        ),
    )
    
    invoker = ModelInvoker(
        model=model,
        input_map={"inputs": {
            "left": "molecule",
            "right": "features",
            "context": "concentration",
        }},
    )

    data = {
        "smiles": ["CCO", "CCN"],
        "features": [
            [0., 0.],
            [0., 1.],
        ],
        "concentration": [
            [0.5],
            [1.0],
        ],
    }

    prepared = pipeline(data)

    batch = next(iter(pipeline.dataloader(batch_size=2)))

    # Prove ordinary invocation works before testing derivatives.
    prediction = invoker.predict(batch)

    assert prediction.shape == (2, 1)

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    gradients = _parameter_gradient(
        model,
        invoker,
        batch
    )

    assert set(gradients) == set(before)

    for name, gradient in gradients.items():
        assert gradient.shape[0] == 2
        assert torch.all(torch.isfinite(gradient))

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_duvida_parameter_gradient_chemprop():

    from aspect import DataPipeline

    from duvidnn.models import (
        ChempropEncoder,
        HillCurve,
        MLP,
        Readout,
        TwoTower,
    )

    torch.manual_seed(0)

    pipeline = DataPipeline({
        "molecule": ("smiles", "chemprop-mol"),
    })

    model = Readout(
        latent=TwoTower(
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
        ),
        readout=HillCurve(
            slope=1.,
            trainable_slope=True,
        ),
    )

    invoker = ModelInvoker(
        model=model,
        input_map={
            "inputs": {
                "left": "molecule",
                "right": "features",
                "context": "concentration",
            },
        },
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
    }

    pipeline(data)

    batch = next(iter(pipeline.dataloader(batch_size=2)))

    prediction = invoker.predict(batch)

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
        assert gradient.shape == (*prediction.shape, *before[name].shape)
        assert torch.all(torch.isfinite(gradient))

    nonzero_grads = []
    print(gradients)
    for name, gradient in gradients.items():
        if name.startswith("latent.towers.left."):
            nonzero_grads.append(torch.any(gradient != 0.))

    assert any(nonzero_grads)

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
