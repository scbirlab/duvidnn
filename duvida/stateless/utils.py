"""Generic utilities for JAX and PyTorch."""

from typing import Callable, Iterable, Union
from functools import partial

from .config import Config
from .typing import Array, ArrayLike

config = Config()
__backend__ = config.backend

if config.backend == 'jax':
    from jax import jit, jvp, grad, hessian, random, vmap
    from jax.flatten_util import ravel_pytree

    def random_normal(seed: int, device=None) -> Callable:
        key = random.key(seed)
        def _normal(shape, *args, **kwargs):
            return random.normal(
                key, 
                shape=shape, 
                *args, **kwargs
            )
        return _normal

elif config.backend == 'torch':
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

    def random_normal(seed: int, device='cpu') -> Callable:
        generator = Generator(device=device).manual_seed(seed)
        def _normal(shape, *args, **kwargs):
            return normal(
                generator=generator, 
                mean=zeros(shape, device=device), 
                std=1.,
                *args, **kwargs
            )
        return _normal

    def ravel_pytree(params):
        """Torch pytree flattener.

        Returns
        -------
        flat : 1-D ndarray / Tensor
        unflatten : Callable[[1-D ndarray], original-shaped pytree]

        Examples
        --------
        >>> import duvida.numpy as dnp
        >>> flat, unravel = _flatten([dnp.ones((2,)), dnp.zeros((3,))])
        >>> flat.shape
        (5,)
        >>> (unravel(flat)[0] == 1.).all()
        True
        """
        leaves, spec = torch.utils._pytree.tree_flatten(params)  # 
        sizes = [p.numel() for p in leaves]
        flat = torch.cat([p.flatten() for p in leaves])

        def unravel(vec):
            chunks = torch.split(vec, sizes)
            rebuilt = [
                chunk.reshape_as(p) 
                for chunk, p in zip(chunks, leaves)
            ]
            return torch.utils._pytree.tree_unflatten(spec, rebuilt)

        return flat, unravel
else:
    raise ValueError(f"Invalid backed: {config.backend}")

    

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
    4.0
    >>> reciprocal(f)(1.)
    0.5

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
