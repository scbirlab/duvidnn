"""JAX implementation of Hessian vector product."""

from typing import Callable
from functools import partial

try:
    from jax import config, jvp, grad
except ImportError as e:
    from carabiner import print_err
    print_err("JAX is not installed. Try re-installing duvida with `pip install duvida[jax]`.")
    raise e
else:
    import jax.numpy as jnp
    config.update('jax_enable_x64', True)

def hvp(f: Callable, 
        argnums: int = 0, 
        *args, **kwargs) -> Callable:

    """Forward-over-reverse Hessian vector product transform for scalar-output functions.

    This does not behave exactly like other JAX transforms. The 
    resulting function takes as its first argument a vector, and the 
    following argments are the original function's positional and 
    keyword arguments.

    Additional aguments are passed to `jax.grad()`.

    Parameters
    ----------
    f : Callable
        JAX-compatible function with scalar-output to transform.
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
    >>> from jax import grad, hessian
    >>> import jax.numpy as jnp
    >>> f = lambda x: jnp.sum(x ** 3. + x ** 2. + 4.)
    >>> a = jnp.array([1., 2.])
    >>> f(a)
    Array(16., dtype=float64)
    >>> grad(f)(a)
    Array([3., 5.], dtype=float64)
    >>> hvp(f)(jnp.ones_like(a), a)
    Array([2., 2.], dtype=float64)
    >>> g = lambda x, y: jnp.sum(x ** 2. + x ** 2. + 4. + y ** 3.)
    >>> b = jnp.array([3., 4.])
    >>> grad(f)(a) == grad(g)(a, b)
    Array([ True,  True], dtype=bool)
    >>> hvp(f)(jnp.ones_like(a), a) == hvp(g)(jnp.ones_like(a), a, b)
    Array([ True,  True], dtype=bool)
    >>> hvp(g, argnums=1)(jnp.ones_like(a), a, b) == hessian(g, argnums=1)(a, b) @ jnp.ones_like(a)
    Array([ True,  True], dtype=bool)

    """

    grad_fn = partial(grad, *args, **kwargs)

    def _hvp(v, *f_args, **f_kwargs):
        d_args = f_args[argnums]
        pre_d_args = [arg for i, arg in enumerate(f_args) if i < argnums]
        post_d_args = [arg for i, arg in enumerate(f_args) if i > argnums]
        
        @grad_fn
        def _jacobian(f_d0_args):
            return f(*pre_d_args, f_d0_args, *post_d_args, 
                     **f_kwargs)

        return jvp(_jacobian, (d_args,), (v,))[1]

    return _hvp