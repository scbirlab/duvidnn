
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
