"""Utilities for evaluating predictions."""

from typing import Callable, Tuple

import numpy as np
from numpy.typing import ArrayLike
import scipy

def _normalize_dims(
    x: ArrayLike, 
    y: ArrayLike,
    broadcast: bool = False,
    aggfun: Callable = np.mean
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize prediction dimensions.

    If `x` has one extra dimension compared to `y` and their leading
    dimensions match exactly, `x` is assumed to be an ensemble of
    predictions and the mean across its last axis is used. Otherwise the
    function ensures that `x` and `y` are broadcastable. and raises a
    ``ValueError`` if they are not.

    Parameters
    ==========
    x : ArrayLike
        Prediction array.
    y : ArrayLike
        Ground truth array.

    Returns
    =======
    Tuple[numpy.ndarray, numpy.ndarray]
        Possibly averaged `x` and original `y`.

    Raises
    ======
    ValueError
        If `x` and `y` are not broadcastable.
    
    """

    x, y = np.asarray(x), np.asarray(y)

    if x.shape[:-1] != y.shape[:(x.ndim - 1)]:
        raise ValueError(
            f"Leading dimensions of x and y (shapes {x.shape} and {y.shape}) are not compatible!"
        )   

    if x.ndim == y.ndim + 1:
        x = aggfun(x, axis=-1)
    elif x.ndim == y.ndim and y.shape[-1] == 1:
        x = aggfun(x, axis=-1, keepdims=True)

    try:
        b = np.broadcast(x, y)
    except ValueError:
        raise ValueError(
            f"Shapes {x.shape} and {y.shape} are not broadcastable!"
        )

    if broadcast:
        bshape = b.shape
        x, y = (np.broadcast_to(v, bshape).ravel() for v in (x, y))

    return x, y


def rmse(x, y):
    """Root mean square error.

    Parameters
    ==========
    x : numpy.ndarray
        Predicted values. If predictions are provided as an ensemble, the mean
        across the last axis is used.
    y : numpy.ndarray
        Ground truth values.

    Returns
    =======
    float
        The RMSE between ``x`` and ``y``.
    
    Examples
    ========
    >>> import numpy as np
    >>> pred = np.array([1.0, 2.0, 3.0])
    >>> truth = np.array([1.0, 2.0, 4.0])
    >>> np.allclose(rmse(pred, truth), np.sqrt(((pred - truth) ** 2).mean()))
    True

    """

    x, y = _normalize_dims(x, y)
    return np.sqrt(np.mean(np.square(x - y)))


def mae(x, y):
    x, y = _normalize_dims(x, y)
    return np.mean(np.abs(x - y))


def pearson_r(x, y):
    x, y = _normalize_dims(x, y, broadcast=True)
    return  np.corrcoef(x, y)[0, 1]


def spearman_r(x, y):
    x, y = _normalize_dims(x, y, broadcast=True)
    return scipy.stats.spearmanr(
        x, 
        y, 
        nan_policy="omit",
    ).statistic
