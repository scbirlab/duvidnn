"""Generic utilities for JAX and PyTorch."""

from typing import Callable, Iterable, Union
from functools import partial

from .config import config
from .typing import Array, ArrayLike

if config.backend == 'jax':
    from jax import jit, jvp, grad, hessian, random, vmap

    def random_normal(seed: int, device=None):
        key = random.key(seed)
        def _normal(shape, *args, **kwargs):
            return random.normal(key, shape=shape, *args, **kwargs)
        return _normal

else:
    from torch import compile, normal, zeros, Generator
    from torch.random import manual_seed
    from torch.func import jvp, grad, hessian, vmap as vmap_torch

    def vmap(
        f: Callable, 
        in_axes: Union[int, Iterable[int]] = 0, 
        out_axes: Union[int, Iterable[int]] = 0, 
        *args, **kwargs
    ) -> Callable:
        return vmap_torch(f, in_dims=in_axes, out_dims=out_axes, 
                          randomness='different',
                          *args, **kwargs)

    jit = partial(compile, fullgraph=True)

    def random_normal(seed: int, device='cpu'):
        generator = Generator(device=device).manual_seed(seed)
        def _normal(shape, *args, **kwargs):
            return normal(generator=generator, 
                          mean=zeros(shape, device=device), 
                          std=1.,
                          *args, **kwargs)
        return _normal

def reciprocal(f: Callable[[ArrayLike], Array]) -> Callable[[ArrayLike], Array]:

    """Convert a function returning a numeric value to return its reciprocal instead.
    
    Parameters
    ----------
    f : Callable
        Function to convert.
    
    Returns
    -------
    Callable
        Function returning the reciprocal of `f`.

    Examples
    --------
    >>> f = lambda x: x + 1.
    >>> f(3.)
    4.
    >>> reciprocal(f)(2.)
    .25

    """

    def _inverted(*args, **kwargs):
        return 1. / f(*args, **kwargs)

    return _inverted


def get_eps():

    """Get machine precision.
    
    """
    
    if config.backend == 'jax':
        from jax.numpy import asarray, finfo
        def converter(x): return asarray(x)
    else:
        from torch import as_tensor, finfo, float64
        def converter(x): return as_tensor(x)
    return converter(finfo(converter(1.).dtype).eps)
