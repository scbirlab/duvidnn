"""JAX transforms for Hessians diagonals and their approximations."""

from typing import Any, Callable, Optional, Tuple, Union

from .hvp import hvp

# local imports will check for presence of JAX
from jax import jvp, grad, random, vmap, Array
from jax.typing import ArrayLike

Approximator = Callable[[Callable], Callable]

_EPS = jnp.sqrt(jnp.finfo(dtype=jnp.float64).eps)
_DEFAULT_APPROXIMATOR = 'exact_diagonal'

def get_approximators(key: Optional[str] = None) -> Union[Tuple[str], Dict[str, Approximator]]:

    """Return a list of available Hessian diagonal approximators, or one of the
    named approximator functions.

    Parameters
    ----------
    key : str, optional
        If provided, return the function with the name `key`.

    Returns
    -------
    Union[Tuple[str], Dict[str, Approximator]]
        If key is provided, returns the named approximator function. Otherwise
        returns a tuple of named approximators.

    Raises
    ------
    NotImplementedError
        When `key` is provided but is not in the list of approximators.

    """

    approximators = {
        'squared_jacobian': squared_jacobian,
        'exact_diagonal': exact_diagonal,
        'bekas': bekas,
        'rough_finite_difference': rough_finite_difference,
    }

    if key is None:
        return tuple(approximators)
    else:
        try:
            return approximators[key]
        except KeyError as e:
            NotImplementedError(f"Approximator called '{key}' is not implemented. Choose from {', '.sorted(approximators)}")


def squared_jacobian(f: Callable[[Any], Array], 
                     *args, **kwargs) -> Callable[[Any], Array]:
    """JAX transform to the square of the Jacobian of a scalar-output function.
    
    Additional aguments are passed to `jax.grad()`. Set `argnums` to
    differentiate `f` with respect to parameters other than the first.

    Parameters
    ----------
    f : Callable
        JAX-compatible function with scalar-output to transform.
    
    Returns
    -------
    Callable
        Function which takes `f`'s original arguments. Returns the 
        square of the gradient of `f` with respect to its first parameter.

    Examples
    --------
    >>> from jax import grad, hessian
    >>> import jax.numpy as jnp
    >>> f = lambda x: jnp.sum(x ** 3. + x ** 2. + 4.)
    >>> a = jnp.array([1., 2.])
    >>> f(a)
    Array(22., dtype=float64)
    >>> jnp.diag(hessian(f)(a))
    Array([ 8., 14.], dtype=float64)
    >>> squared_jacobian(f)(a) == jnp.square(grad(f)(a))
    Array([ True,  True], dtype=bool)
    >>> squared_jacobian(f)(a)  # Not very accurate!
    Array([ 25., 256.], dtype=float64)
    
    """
    _jacobian = grad(f, *args, **kwargs)

    def _sq_jacobian(*args, **kwargs) -> Array:
        return jnp.square(_jacobian(*args, **kwargs))

    return _sq_jacobian


def exact_diagonal(f: Callable[[Any], Array], 
                   argnums: int = 0,
                   *args, **kwargs) -> Callable[[Any], Array]:

    """JAX transform to the Hessian diagonal of a scalar-output function.
    
    Additional aguments are passed to `hvp()`. Set `argnums` to
    differentiate `f` with respect to parameters other than the first.

    Parameters
    ----------
    f : Callable
        JAX-compatible function with scalar-output to transform.
    argnums : int, optional
        Which argument number to take the second derivative for. Default: 0.
    
    Returns
    -------
    Callable
        Function which takes `f`'s original arguments. Returns the 
        diagonal of the Hessian of `f`. 

    Examples
    --------
    >>> from jax import hessian
    >>> import jax.numpy as jnp
    >>> f = lambda x: jnp.sum(x ** 3. + x ** 2. + 4.)
    >>> a = jnp.array([1., 2.])
    >>> exact_diagonal(f)(a) == jnp.diag(hessian(f)(a))
    Array([ True,  True], dtype=bool)
    >>> g = lambda x: jnp.sum(jnp.sum(x) ** 3. + x ** 2. + 4.)
    >>> exact_diagonal(g)(a) == jnp.diag(hessian(g)(a))
    Array([ True,  True], dtype=bool)
    
    """
    hvp_f = hvp(f, argnums=argnums, *args, **kwargs)

    def get_hessian_element(i: int, 
                            zeros: ArrayLike, 
                            *args, **kwargs) -> float:
        zeros = zeros.at[i].set(1.)
        return hvp_f(zeros, *args, **kwargs)[i]

    def _hessian_diagonal(*args, **kwargs) -> Array:
        d_args = args[argnums]
        v_hvp_f = vmap(get_hessian_element, 
                       in_axes=(0, None) + (None, ) * len(args))
        zeros = jnp.zeros((d_args.size, ), dtype=jnp.float64)
        idx = jnp.arange(d_args.size, dtype=jnp.int64)  # p
        return v_hvp_f(idx, zeros, *args, **kwargs) 

    return _hessian_diagonal


def bekas(f: Callable[[Any], Array], 
          n: int = 1, 
          seed: int = 0,
          argnums: int = 0,
          *args, **kwargs) -> Callable[[Any], Array]:

    """JAX transform to the Bekas estimator of the Hessian diagonal of a scalar-output function.
    
    Additional aguments are passed to `hvp()`. Set `argnums` to
    differentiate `f` with respect to parameters other than the first.

    Bekas estimator is described at https://doi.org/10.1016/j.apnum.2007.01.003.

    Parameters
    ----------
    f : Callable
        JAX-compatible function with scalar-output to transform.
    n : int, optional
        Number of Gaussian random samples to take. Default: 1.
    seed : int, optional
        Controls the random behavior. Set a different seed to get a 
        different result. Default: 0.
    argnums : int, optional
        Which argument number to take the second derivative for. Default: 0.
    
    Returns
    -------
    Callable
        Function which takes `f`'s original arguments. Returns the 
        Bekas approximation of the diagonal of the Hessian of `f`. 

    Examples
    --------
    >>> from jax import hessian
    >>> import jax.numpy as jnp
    >>> f = lambda x: jnp.sum(x ** 3. + x ** 2. + 4.)
    >>> a = jnp.array([1., 2.])
    >>> f(a)
    Array(22., dtype=float64)
    >>> jnp.diag(hessian(f)(a))
    Array([ 8., 14.], dtype=float64)
    >>> bekas(f)(a)  # Apparently very accurate
    Array([ 8., 14.], dtype=float64)
    >>> jnp.allclose(bekas(f)(a), jnp.diag(hessian(f)(a)))
    Array(True, dtype=bool)
    >>> bekas(f)(a) == jnp.diag(hessian(f)(a))  # But not quite exact
    Array([ True, False], dtype=bool)
    >>> bekas(f, n=1000)(a) == jnp.diag(hessian(f)(a))  # Increase samples for better accuracy
    Array([ True,  True], dtype=bool)
    >>> bekas(f, seed=1)(a) == bekas(f, seed=0)(a)  # Change the seed to alter the outcome
    Array([ True, False], dtype=bool)
    >>> g = lambda x: jnp.sum(jnp.sum(x) ** 3. + x ** 2. + 4.)
    >>> jnp.diag(hessian(g)(a))
    Array([38., 38.], dtype=float64)
     >>> bekas(g, n=1000)(a)  # Less accurate when parameters interact
    Array([38.52438307, 38.49679655], dtype=float64)
    >>> bekas(g, n=1000, seed=1)(a)  # Change the seed to alter the outcome
    Array([39.07878869, 38.97796601], dtype=float64)
    
    """
    key = random.key(seed)
    hvp_f = hvp(f, argnums=argnums, *args, **kwargs)

    def _approx_hessian_diagonal(*args, **kwargs) -> Array:
        d_args = args[argnums]
        v_hvp_f = vmap(hvp_f, 
                       in_axes=(1, ) + (None, ) * len(args), 
                       out_axes=1)
        v = random.normal(key, shape=(d_args.size, n))  # p, n
        samples = v * v_hvp_f(v, *args, **kwargs)   # p, n
        return jnp.sum(samples, axis=-1) / jnp.sum(jnp.square(v), axis=-1)

    return _approx_hessian_diagonal


def rough_finite_difference(f: Callable[[Any], Array], 
                            argnums: int = 0, 
                            eps: float = _EPS,
                            *args, **kwargs) -> Callable[[Any], Array]:
    
    """JAX transform to a rough Hessian diagonal of a scalar-output function.

    This uses the usual finite-difference algorithm from `scipy.optimize.fprime()`
    to calculate the Jacobian of the autodiff gradients of `f`, but changes 
    all values at once, instead of one-by-one. 
    
    This will work best when parameters don't have strong interactions with
    each other becuase it's essentially a column sum of the Hessian matrix.
    
    Additional aguments are passed to `grad()`. Set `argnums` to
    differentiate `f` with respect to parameters other than the first.

    Parameters
    ----------
    f : Callable
        JAX-compatible function with scalar-output to transform.
    argnums : int, optional
        Which argument number to take the second derivative for. Default: 0.
    eps : float, optional
        The
    
    Returns
    -------
    Callable
        Function which takes `f`'s original arguments. Returns the 
        rough finite-difference approximation of the diagonal of the 
        Hessian of `f`. 

    Examples
    --------
    >>> from jax import hessian
    >>> import jax.numpy as jnp
    >>> f = lambda x: jnp.sum(x ** 3. + x ** 2. + 4.)
    >>> a = jnp.array([1., 2.])
    >>> f(a)
    Array(22., dtype=float64)
    >>> jnp.diag(hessian(f)(a))
    Array([ 8., 14.], dtype=float64)
    >>> rough_finite_difference(f)(a)  # Relatively accurate
    Array([ 8.00000006, 14.        ], dtype=float64)
    >>> rough_finite_difference(f, eps=.01)(a)
    Array([ 8.03, 14.03], dtype=float64)
    >>> g = lambda x: jnp.sum(jnp.sum(x) ** 3. + x ** 2. + 4.)
    >>> jnp.diag(hessian(g)(a))
    Array([38., 38.], dtype=float64)
    >>> rough_finite_difference(g)(a)  # Less accurate when parameters interact
    Array([74., 74.], dtype=float64)
    
    """
    _jacobian = grad(f, argnums=argnums, *args, **kwargs)

    def _approx_hessian_diagonal(*args, **kwargs) -> Array:
        d_args = [arg + eps if i == argnums else arg for i, arg in enumerate(args)]
        return (_jacobian(*d_args, **kwargs) - _jacobian(*args, **kwargs)) / eps

    return _approx_hessian_diagonal
