import torch
from torch import nn

from duvidnn.invoke import (
    DuvidaModel, 
    ModelInvoker,
)


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

    functional = DuvidaModel(model, invoker)
    gradients = functional.parameter_gradient(batch)

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

    functional = DuvidaModel(model, invoker)
    gradients = functional.parameter_gradient(batch)

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
    functional = DuvidaModel(model, invoker)
    gradients = functional.parameter_gradient(batch)

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


def test_duvida_fisher_preserves_parameter_structure():

    model = nn.Linear(2, 1)

    invoker = ModelInvoker(
        model=model,
        input_map={
            "inputs": {"input": "x"},
            "target": "y",
        },
    )

    batch = {
        "x": torch.tensor([
            [1., 2.],
            [3., 4.],
        ]),
        "y": torch.tensor([
            [0.],
            [1.],
        ])
    }

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }
    loss = lambda prediction, observed: torch.sum(
        torch.square(
            prediction - observed
        )
    )
    functional = DuvidaModel(model, invoker)
    score = functional.fisher_score(batch, loss)
    information = functional.fisher_information(batch, loss)

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
        input_map={
            "inputs": {
                "left": "molecule",
                "right": "features",
            },
            "target": "y",
        },
    )

    pipeline({
        "smiles": ["CCO", "CCN"],
        "features": [
            [0., 0.],
            [1., 1.],
        ],
        "y": [
            [.25],
            [.75],
        ],
    })

    batch = next(iter(pipeline.dataloader(batch_size=2)))

    _ = invoker.predict(batch)

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    loss = lambda prediction, observed: torch.sum(
        torch.square(
            prediction - observed
        )
    )
    functional = DuvidaModel(model, invoker)
    score = functional.fisher_score(batch, loss)
    information = functional.fisher_information(batch, loss)

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
            "target": "y",
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
        "y": [
            [0.25],
            [0.75],
            [0.25],
            [0.75],
        ]
    })

    loader = pipeline.dataloader(batch_size=2)

    iterator = iter(loader)

    batch = next(iterator)
    candidate_batch = next(iterator)

    _ = invoker.predict(batch)

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    loss = lambda prediction, observed: torch.sum(
        torch.square(
            prediction - observed
        )
    )
    functional = DuvidaModel(model, invoker)
    observed = functional.doubtscore(candidate_batch, batch, loss)

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
            "target": "y",
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
        "y": [
            [0.25],
            [0.75],
            [0.25],
            [0.75],
        ],
    })

    loader = pipeline.dataloader(batch_size=2)

    iterator = iter(loader)

    batch = next(iterator)
    candidate_batch = next(iterator)

    _ = invoker.predict(batch)

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    loss = lambda prediction, observed: torch.sum(
        torch.square(
            prediction - observed
        )
    )
    functional = DuvidaModel(model, invoker)
    observed = functional.information_sensitivity(candidate_batch, batch, loss)

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


def test_accumulated_fisher_score_matches_full_batch():

    model = nn.Linear(2, 1)
    invoker = ModelInvoker(
        model=model,
        input_map={
            "inputs": {
                "input": "x",
            },
            "target": "y",
        },
    )

    functional = DuvidaModel(model, invoker)
    loss = nn.MSELoss(reduction="mean")

    full_batch = {
        "x": torch.tensor([
            [1., 2.],
            [3., 4.],
            [5., 6.],
        ]),
        "y": torch.tensor([
            [1.],
            [2.],
            [3.],
        ]),
    }

    batches = [
        {
            key: value[:2]
            for key, value
            in full_batch.items()
        },
        {
            key: value[2:]
            for key, value
            in full_batch.items()
        },
    ]

    observed, _ = (
        functional.accumulate_fisher_score(
            batches,
            loss,
            loss_reduction="mean",
        )
    )
    expected = functional.fisher_score(
        full_batch,
        loss,
        loss_reduction="mean",
    )

    for name in expected:
        assert torch.allclose(observed[name], expected[name])
