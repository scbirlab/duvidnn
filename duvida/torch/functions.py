"""Functions in pytorch."""

import torch 

from ..stateless.typing import ArrayLike, Array


def mse_loss(
    y_pred: ArrayLike, 
    y_true: ArrayLike
) -> Array:
    return torch.mean(torch.square(y_pred - y_true))
