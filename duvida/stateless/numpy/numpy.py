"""Very rough backend-agnostic NumPy API."""

from ..config import config
from ..typing import Array, ArrayLike

if config.backend == 'jax':
    from jax.numpy import (
        float64, int64, 
        allclose, 
        arange as jax_arange, 
        asarray, array, concatenate, diag, 
        expand_dims, ones, ones_like, roll, sum, sqrt, square, take, zeros,
        zeros_like
    )
    from jax.nn import one_hot as jax_one_hot

    def arange(start, stop=None, step=None, device: str = "cpu") -> Array:
        return jax_arange(start, stop, step)

    def get_array_size(a: ArrayLike) -> int:
        return a.size

    def set_array_element(a: ArrayLike, i: int, x: ArrayLike) -> Array:
        return a.at[i].set(x)

    def unsqueeze(a: ArrayLike, axis: int) -> Array:
        return expand_dims(a, axis)

    def dtype_like(a: ArrayLike, b: ArrayLike) -> Array:
        return a.astype(asarray(b).dtype)

    def one_hot(tensor: ArrayLike, num_classes: int = -1, device: str = 'cpu') -> Array:
        return dtype_like(jax_one_hot(tensor, num_classes), 1.)

else:
    from torch import (
        float64, int64, 
        as_tensor as asarray, 
        Tensor as array, 
        arange, 
        allclose, 
        concat as concatenate, 
        diag, numel, 
        ones, ones_like, 
        roll, sum, sqrt, square, take, 
        zeros, zeros_like
    )
    from torch.nn.functional import one_hot as torch_one_hot

    def get_array_size(a: ArrayLike):
        return numel(a)

    def set_array_element(a: ArrayLike, i: int, x: ArrayLike) -> Array:
        a_copy = a.detach().clone()
        a_copy[i] = x
        return a_copy

    def unsqueeze(a: ArrayLike, axis: int) -> Array:
        return a.unsqueeze(axis)
    
    def dtype_like(a: ArrayLike, b: ArrayLike) -> Array:
        return a.to(asarray(b).dtype)

    def one_hot(tensor: ArrayLike, num_classes: int = -1, device: str = 'cpu') -> Array:
        return dtype_like(torch_one_hot(tensor, num_classes), 1.).to(device)