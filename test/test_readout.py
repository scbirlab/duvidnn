
from duvidnn.models import Readout

def test_readout_model():
    import torch
    from torch import nn

    class Predictor(nn.Module):
        def forward(self, x):
            return x * 2

    class ReadoutFn(nn.Module):
        def forward(self, latent, context):
            return latent + context

    model = Readout(
        latent=Predictor(),
        readout=ReadoutFn(),
    )

    x = torch.tensor([
        [1.],
        [2.],
    ])

    context = torch.tensor([
        [10.],
        [20.],
    ])

    observed = model(
        x=x,
        context=context,
    )

    expected = torch.tensor([
        [12.],
        [24.],
    ])

    torch.testing.assert_close(observed, expected)


def test_readout_model_without_context():
    import torch
    from torch import nn

    class Predictor(nn.Module):
        def forward(self, x):
            return x * 2

    class ReadoutFn(nn.Module):
        def forward(self, latent):
            return latent + 1.

    model = Readout(
        latent=Predictor(),
        readout=ReadoutFn(),
    )

    x = torch.tensor([
        [1.],
        [2.],
    ])

    observed = model(
        x=x,
        context=None,
    )

    expected = torch.tensor([
        [3.],
        [5.],
    ])

    torch.testing.assert_close(observed, expected)


MODEL_CONFIG = {
    "class_path": "duvidnn.models.composition.Readout",
    "init_args": {
        "latent": {
            "class_path": "duvidnn.models.mlp.MLP",
            "init_args": {"input_dim": 2},
        },
        "readout": {
            "class_path": "torch.nn.Sigmoid",
            "init_args": {},
        },
    },
}


def test_instantiate_readout_from_config():
    from duvidnn.config import instantiate_model
    from duvidnn.models import Readout

    model = instantiate_model(MODEL_CONFIG)

    assert isinstance(model, Readout)


def test_readout_model_with_two_tower_latent():
    import torch
    from torch import nn

    from duvidnn.models import Readout, TwoTower

    class Left(nn.Module):
        def forward(self, x):
            return x * 2.

    class Right(nn.Module):
        def forward(self, x):
            return x * 3.

    class Fusion(nn.Module):
        def forward(self, x):
            return x.sum(dim=-1, keepdim=True)

    class ReadoutFn(nn.Module):
        def forward(self, latent, context):
            return latent + context

    model = Readout(
        latent=TwoTower(
            left=Left(),
            right=Right(),
            fusion=Fusion(),
            merge="concat",
        ),
        readout=ReadoutFn(),
    )

    left = torch.tensor([
        [1.],
        [2.],
    ])

    right = torch.tensor([
        [4.],
        [5.],
    ])

    context = torch.tensor([
        [10.],
        [20.],
    ])

    observed = model(
        left=left,
        right=right,
        context=context,
    )

    expected = torch.tensor([
        [24.],
        [39.],
    ])

    torch.testing.assert_close(
        observed,
        expected,
    )


def test_hill_readout_at_ic50():
    import torch
    from duvidnn.models import HillCurve

    readout = HillCurve(slope=1.)

    assert readout.latent_params == ("log_ic50",)
    assert readout.context_params == ("conc",)

    ic50 = torch.tensor([
        [2.],
        [10.],
    ])
    observed = readout(
        latent=torch.log(ic50),
        context=ic50,
    )
    expected = torch.full_like(ic50, .5)

    torch.testing.assert_close(observed, expected)


def test_hill_curve_rejects_nonpositive_concentration():

    import pytest
    import torch

    from duvidnn.models.physical.dose import hill_curve

    with pytest.raises(
        ValueError,
        match="Concentration must be positive",
    ):
        hill_curve(
            conc=torch.zeros(1),
            log_ic50=torch.zeros(1),
        )
