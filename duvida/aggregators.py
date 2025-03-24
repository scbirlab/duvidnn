"""Functions for collapsing dimensions."""

from typing import Callable, Union
from functools import partial

from numpy import nanmean, nanmedian, nansum, nanvar, ndarray
from numpy.linalg import norm
from numpy.typing import ArrayLike

AggFunction = Callable[[ArrayLike], ndarray]

AGGREGATORS = {
    "mean": nanmean,
    "median": nanmedian,
    "sum": nansum,
    "var": nanvar,
    "rms": partial(norm, ord=2, keepdims=False),
}

def slice_at_axis(a, i, axis=-1):
    n_dims = len(a.shape)
    if axis < 0:
        axis = n_dims + axis
    slicer = tuple([] if j == axis else [i] for j in range(n_dims))
    return a[slicer]
    

def get_aggregator(
    aggregator: Union[str, AggFunction],
    axis: int = -1,
    **kwargs
) -> AggFunction:
    
    """
    
    """

    if isinstance(aggregator, int):
        return partial(slice_at_axis, i=aggregator, axis=axis)
    elif isinstance(aggregator, str):
        if aggregator.endswith("-norm") and aggregator.split("-").isdigit():
            return partial(norm, ord=int(aggregator.split("-")), axis=axis)
        elif aggregator in AGGREGATORS:
            return partial(AGGREGATORS[aggregator], axis=axis, **kwargs)
        else:
            raise ValueError(f"Invalid aggregator name '{aggregator}'!")
    elif isinstance(aggregator, Callable):
        return aggregator
    else:
        raise ValueError(f"Invalid aggregator type '{type(aggregator)}'! Must be a str or callable.")