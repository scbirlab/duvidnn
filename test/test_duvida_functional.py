import torch
from torch import nn

from duvida import (
    doubtscore,
    fisher_information_diagonal,
    fisher_score,
    information_sensitivity,
    parameter_gradient,
)

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


def _fisher_score(
    model,
    invoker,
    batch,
    target,
):
    inputs = invoker.inputs(
        batch
    )

    stateless_model = make_stateless_model(
        model
    )

    params = dict(
        model.named_parameters()
    )

    loss = lambda prediction, observed: torch.sum(
        torch.square(
            prediction - observed
        )
    )

    return fisher_score(
        stateless_model,
        loss,
    )(
        (params,),
        inputs,
        target,
    )[0]


def _fisher_information(
    model,
    invoker,
    batch,
    target,
):
    inputs = invoker.inputs(
        batch
    )

    stateless_model = make_stateless_model(
        model
    )

    params = dict(
        model.named_parameters()
    )

    loss = lambda prediction, observed: torch.sum(
        torch.square(
            prediction - observed
        )
    )

    return fisher_information_diagonal(
        stateless_model,
        loss,
        approximator="squared_jacobian",
    )(
        (params,),
        inputs,
        target,
    )[0]


def _doubtscore(
    model,
    invoker,
    batch,
    candidate_batch,
    target,
):
    inputs = invoker.inputs(
        batch
    )

    candidate_inputs = invoker.inputs(
        candidate_batch
    )

    stateless_model = make_stateless_model(
        model
    )

    params = dict(
        model.named_parameters()
    )

    loss = lambda prediction, observed: torch.sum(
        torch.square(
            prediction - observed
        )
    )

    return doubtscore(
        stateless_model,
        loss,
    )(
        (params,),
        candidate_inputs,
        inputs,
        target,
    )[0]


def _information_sensitivity(
    model,
    invoker,
    batch,
    candidate_batch,
    target,
):
    inputs = invoker.inputs(
        batch
    )

    candidate_inputs = invoker.inputs(
        candidate_batch
    )

    stateless_model = make_stateless_model(
        model
    )

    params = dict(
        model.named_parameters()
    )

    loss = lambda prediction, observed: torch.sum(
        torch.square(
            prediction - observed
        )
    )

    return information_sensitivity(
        stateless_model,
        loss,
        approximator="squared_jacobian",
    )(
        (params,),
        candidate_inputs,
        inputs,
        target,
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


def test_duvida_fisher_preserves_parameter_structure():

    model = nn.Linear(2, 1)

    invoker = ModelInvoker(
        model=model,
        input_map={"inputs": {"input": "x"},
        },
    )

    batch = {
        "x": torch.tensor([
            [1., 2.],
            [3., 4.],
        ]),
    }

    target = torch.tensor([
        [0.],
        [1.],
    ])

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    score = _fisher_score(
        model,
        invoker,
        batch,
        target,
    )

    information = _fisher_information(
        model,
        invoker,
        batch,
        target,
    )

    assert set(score) == set(before)
    assert set(information) == set(before)

    for name in before:
        assert score[name].shape == before[name].shape
        assert information[name].shape == before[name].shape

        assert torch.all(torch.isfinite(score[name]))
        assert torch.all(torch.isfinite(information[name]))

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_duvida_fisher_chemprop():

    from aspect import DataPipeline

    from duvidnn.models import (
        ChempropEncoder,
        MLP,
        TwoTower,
    )

    torch.manual_seed(0)

    pipeline = DataPipeline({
        "molecule": ("smiles", "chemprop-mol"),
    })

    model = TwoTower(
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

    invoker = ModelInvoker(
        model=model,
        input_map={"inputs": {
            "left": "molecule",
            "right": "features",
        }},
    )

    pipeline({
        "smiles": ["CCO", "CCN"],
        "features": [
            [0., 0.],
            [1., 1.],
        ],
    })

    batch = next(iter(pipeline.dataloader(batch_size=2)))

    target = torch.tensor([
        [.25],
        [.75],
    ])

    _ = invoker.predict(batch)

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    score = _fisher_score(
        model,
        invoker,
        batch,
        target,
    )

    information = _fisher_information(
        model,
        invoker,
        batch,
        target,
    )

    assert set(score) == set(before)
    assert set(information) == set(before)

    for name in before:
        assert score[name].shape == before[name].shape
        assert information[name].shape == before[name].shape
        assert torch.all(torch.isfinite(score[name]))
        assert torch.all(torch.isfinite(information[name]))

    chemprop_parameters = [
        name
        for name in before
        if name.startswith("towers.left.")
    ]

    assert chemprop_parameters
    assert any(
        torch.any(score[name] != 0.)
        for name in chemprop_parameters
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_duvida_doubtscore_chemprop():

    from aspect import DataPipeline

    from duvidnn.models import (
        ChempropEncoder,
        MLP,
        TwoTower,
    )

    torch.manual_seed(0)

    pipeline = DataPipeline({
        "molecule": ("smiles", "chemprop-mol"),
    })

    model = TwoTower(
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

    invoker = ModelInvoker(
        model=model,
        input_map={
            "inputs": {
                "left": "molecule",
                "right": "features",
            },
        },
    )

    pipeline({
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
    })

    loader = pipeline.dataloader(batch_size=2)

    iterator = iter(loader)

    batch = next(iterator)
    candidate_batch = next(iterator)

    target = torch.tensor([
        [0.25],
        [0.75],
    ])

    _ = invoker.predict(batch)

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    observed = _doubtscore(
        model,
        invoker,
        batch,
        candidate_batch,
        target,
    )

    assert set(observed) == set(before)

    candidate_prediction = invoker.predict(candidate_batch)

    for name, score in observed.items():
        assert score.shape == (
            *candidate_prediction.shape,
            *before[name].shape,
        )

    chemprop_parameters = [
        name
        for name in before
        if name.startswith("towers.left.")
    ]

    assert chemprop_parameters
    assert any(
        torch.any(torch.isfinite(observed[name]))
        for name in chemprop_parameters
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])


def test_duvida_information_sensitivity_chemprop():

    from aspect import DataPipeline

    from duvidnn.models import (
        ChempropEncoder,
        MLP,
        TwoTower,
    )

    torch.manual_seed(0)

    pipeline = DataPipeline({
        "molecule": ("smiles", "chemprop-mol"),
    })

    model = TwoTower(
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

    invoker = ModelInvoker(
        model=model,
        input_map={
            "inputs": {
                "left": "molecule",
                "right": "features",
            },
        },
    )

    pipeline({
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
    })

    loader = pipeline.dataloader(batch_size=2)

    iterator = iter(loader)

    batch = next(iterator)
    candidate_batch = next(iterator)

    target = torch.tensor([
        [0.25],
        [0.75],
    ])

    _ = invoker.predict(batch)

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    observed = _information_sensitivity(
        model,
        invoker,
        batch,
        candidate_batch,
        target,
    )

    assert set(observed) == set(before)

    candidate_prediction = invoker.predict(candidate_batch)

    for name, sensitivity in observed.items():
        assert sensitivity.shape == (
            *candidate_prediction.shape,
            *before[name].shape,
        )

    chemprop_parameters = [
        name
        for name in before
        if name.startswith("towers.left.")
    ]

    assert chemprop_parameters
    assert any(
        torch.any(torch.isfinite(observed[name]))
        for name in chemprop_parameters
    )
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
