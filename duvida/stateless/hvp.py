"""Functional transform for Hessian vector product."""

from typing import Callable
from functools import partial

from .numpy import numpy as dnp
from .typing import Array, ArrayLike
from .utils import grad, jvp


def hvp(
    f: Callable, 
    argnums: int = 0, 
    *args, **kwargs
) -> Callable:

    """Forward-over-reverse Hessian vector product transform for scalar-output functions.

    This does not behave exactly like other functional transforms. The 
    resulting function takes as its first argument a vector, and the 
    following argments are the original function's positional and 
    keyword arguments.

    Additional aguments are passed to `jax.grad()`.

    Parameters
    ----------
    f : Callable
        Function with scalar-output to transform.
    argnums : int, optional
        Which argument number to take the second derivative for. Default: 0.
    
    Returns
    -------
    Callable
        Function which takes a vector `v` as its first argument, and then `f`'s 
        original arguments. Returns the vector product between the Hessian
        of `f` and `v`.

    Examples
    --------
    >>> from duvida.stateless.config import config
    >>> config.set_backend("jax", precision="double")
    >>> from duvida.stateless.utils import grad, hessian
    >>> import duvida.stateless.numpy as dnp 
    >>> f = lambda x: dnp.sum(x ** 3. + x ** 2. + 4.)
    >>> a = dnp.array([1., 2.])
    >>> f(a)
    Array(22., dtype=float64)
    >>> hvp(f)(dnp.ones_like(a), a) == hessian(f)(a) @ dnp.ones_like(a)
    Array([ True,  True], dtype=bool)
    >>> g = lambda x, y: dnp.sum(x ** 2. + x ** 2. + 4. + y ** 3.)
    >>> b = dnp.array([3., 4.])
    >>> hvp(g)(dnp.ones_like(a), a, b) == hessian(g)(a, b) @ dnp.ones_like(a)
    Array([ True,  True], dtype=bool)
    >>> hvp(g, argnums=1)(dnp.ones_like(a), a, b) == hessian(g, argnums=1)(a, b) @ dnp.ones_like(a)
    Array([ True,  True], dtype=bool)

    """

    grad_fn = partial(grad, *args, **kwargs)

    def _hvp(v: ArrayLike, *f_args, **f_kwargs) -> Array:
        d_args = dnp.asarray(f_args[argnums])
        pre_d_args = [arg for i, arg in enumerate(f_args) if i < argnums]
        post_d_args = [arg for i, arg in enumerate(f_args) if i > argnums]
        
        def _jacobian(f_d0_args):
            return f(
                *pre_d_args, 
                f_d0_args, 
                *post_d_args, 
                **f_kwargs,
            )

        return jvp(grad_fn(_jacobian), (d_args,), (v,))[1]

    return _hvp
