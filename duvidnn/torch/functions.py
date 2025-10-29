"""Functions in pytorch."""

from typing import Callable, Tuple

from duvida.types import ArrayLike, Array
import torch 


def _normalize_dims(
    x: ArrayLike, 
    y: ArrayLike,
    broadcast: bool = False
) -> Tuple[Array, Array]:
    """Normalize prediction dimensions.

    If `x` has one extra dimension compared to `y` and their leading
    dimensions match exactly, `x` is assumed to be an ensemble of
    predictions and the mean across its last axis is used. Otherwise the
    function ensures that `x` and `y` are broadcastable. and raises a
    ``ValueError`` if they are not.

    Parameters
    ==========
    x : torch.Tensor
        Prediction array.
    y : torch.Tensor
        Ground truth array.

    Returns
    =======
    Tuple[torch.Tensor, torch.Tensor]
        Possibly averaged `x` and original `y`.

    Raises
    ======
    ValueError
        If `x` and `y` are not broadcastable.
    
    """

    # x, y = np.asarray(x), np.asarray(y)

    try:
        yshape = y.shape
    except AttributeError as e:
        from carabiner import print_err
        # y is a list!
        print_err(f"[WARN] y is a {type(y)}! {y=}")
        print_err(f"{x=}")
        raise e
    if x.shape[:-1] != yshape[:(x.ndim - 1)]:
        raise ValueError(
            f"Leading dimensions of x and y (shapes {x.shape} and {y.shape}) are not compatible!"
        )   

    if x.ndim == y.ndim + 1:
        y = y.unsqueeze(-1)
    elif x.ndim == y.ndim and y.shape[-1] not in (x.shape[-1], 1) and x.shape[-1] != 1:
        raise ValueError(
            f"Trailing dimensions of x and y (shapes {x.shape} and {y.shape}) are not compatible!"
        )   

    try:
        b = torch.broadcast_tensors(x, y)
    except ValueError:
        raise ValueError(
            f"Shapes {x.shape} and {y.shape} are not broadcastable!"
        )

    if broadcast:
        bshape = b.shape
        x, y = (torch.broadcast_to(v, bshape).ravel() for v in (x, y))

    return x, y


def mse_loss(
    y_pred: ArrayLike, 
    y_true: ArrayLike
) -> Array:
    y_pred, y_true = _normalize_dims(y_pred, y_true)
    return torch.mean(torch.square(y_pred - y_true))


def mae_loss(
    y_pred: ArrayLike, 
    y_true: ArrayLike
) -> Array:
    y_pred, y_true = _normalize_dims(y_pred, y_true)
    return torch.mean(torch.abs(y_pred - y_true))


def cosine_loss(
    y_pred: ArrayLike, 
    y_true: ArrayLike
) -> Array:
    y_pred, y_true = _normalize_dims(y_pred, y_true)
    return torch.mean(torch.cos(y_pred, y_true))

# def rmse(x, y):
#     """Root mean square error.

#     Parameters
#     ==========
#     x : torch.Tensor
#         Predicted values. If predictions are provided as an ensemble, the mean
#         across the last axis is used.
#     y : torch.Tensor
#         Ground truth values.

#     Returns
#     =======
#     float
#         The RMSE between ``x`` and ``y``.

#     """

#     x, y = _normalize_dims(x, y)
#     return torch.mean(torch.square(x - y)))



