"""Generic utilities for JAX and PyTorch."""

from typing import Callable, Iterable, Union
from functools import partial

from carabiner import print_err

from .config import config
from .typing import Array, ArrayLike

__backend__ = config.backend

if config.backend == 'jax':
    from jax import jit, jvp, grad, hessian, random, vmap
    from jax.flatten_util import ravel_pytree

    def random_normal(seed: int, device=None) -> Callable:
        """Generate a sample from the Normal distribution.

        Examples
        ========
        >>> gen = random_normal(seed=0)
        >>> a, b = gen((3,)), gen((3,))
        >>> (a == b).all()
        Array(True, dtype=bool)

        """
        key = random.key(seed)
        def _normal(shape, *args, **kwargs):
            return random.normal(
                key, 
                shape=shape, 
                *args, **kwargs
            )
        return _normal

elif config.backend == 'torch':
    from functools import wraps
    from torch import concat as concatenate, compile, normal, zeros, Generator, split
    from torch.random import manual_seed
    from torch.func import jvp, grad, hessian, vmap as vmap_torch
    from torch.utils._pytree import tree_flatten, tree_unflatten
    from torch._dynamo import config as dynamo_config
    dynamo_config.suppress_errors = True
    dynamo_config.capture_scalar_outputs = True

    _COMPILE_WARNINGS = set()

    def vmap(
        f: Callable, 
        in_axes: Union[int, Iterable[int]] = 0, 
        out_axes: Union[int, Iterable[int]] = 0, 
        *args, **kwargs
    ) -> Callable:
        """Vectorizes function over axis of its arguments.

        Examples
        ========
        >>> import duvida.stateless.numpy as dnp
        >>> double = lambda z: z * 2
        >>> vmap(double)(dnp.array([1., 2., 3.])).tolist()
        [2.0, 4.0, 6.0]

        """
        return vmap_torch(
            f, 
            in_dims=in_axes, 
            out_dims=out_axes, 
            randomness='different',
            *args, **kwargs
        )
        

    def jit(fn):
        try:
            compiled = compile(
                fn,
                fullgraph=True,
                mode="max-autotune",
            )
        except RuntimeError as e:
            warning = f"[torch.compile] Compiling `{fn}` failed; running eagerly"
            if warning not in _COMPILE_WARNINGS:
                _COMPILE_WARNINGS.add(warning)
                print_err(warning + "\n" + str(e))
            return fn

        @wraps(fn)
        def wrapped(*args, **kwargs):
            try:
                return compiled(*args, **kwargs)
            except Exception as e:
                warning = f"[torch.compile] Running compiled `{fn}` failed; falling back"
                if warning not in _COMPILE_WARNINGS:
                    _COMPILE_WARNINGS.add(warning)
                    print_err(warning + "\n" + str(e))
                return fn(*args, **kwargs)
        return wrapped


    def random_normal(seed: int, device='cpu') -> Callable:
        """Generate a sample from the Normal distribution.

        Examples
        ========
        >>> gen = random_normal(seed=0)
        >>> a, b = gen((3,)), gen((3,))
        >>> a, b
        [1], [2]
        >>> (a == b).all()
        tensor(True)

        """
        generator = Generator(device=device)
        generator.manual_seed(seed)

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
        >>> import duvida.stateless.numpy as dnp
        >>> flat, unravel = ravel_pytree([dnp.ones((2,)), dnp.zeros((3,))])
        >>> flat.shape
        (5,)
        >>> (unravel(flat)[0] == 1.).all()
        True
        """
        leaves, spec = tree_flatten(params)  # 
        sizes = [p.numel() for p in leaves]
        flat = concatenate([p.flatten() for p in leaves])

        def unravel(vec):
            chunks = split(vec, sizes)
            rebuilt = [
                chunk.reshape_as(p) 
                for chunk, p in zip(chunks, leaves)
            ]
            return tree_unflatten(spec, rebuilt)

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

        def converter(x): 
            return asarray(x)

    else:
        from torch import as_tensor, finfo

        def converter(x): 
            return as_tensor(x)
            
    return converter(finfo(converter(1.).dtype).eps)
