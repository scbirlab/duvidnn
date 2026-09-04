
from torch import nn
from duvidnn.models.composition import Ensemble


def test_ensemble():

    ensemble = Ensemble([
        nn.Linear(3, 2) for _ in range(3)
    ])
    observed = ensemble(
        torch.randn(4, 3),
    )

    assert observed.shape == (4, 2, 3)
 