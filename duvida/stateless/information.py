"""Implementation of information sensitivity metrics for stateless model frameworks."""

from typing import Callable, Union
from functools import partial

from .hessians import _DEFAULT_APPROXIMATOR, get_approximators
from .numpy import numpy as dnp
from .typing import Array, ArrayLike, LossFunction, StatelessModel
from .utils import reciprocal, grad, jit, vmap


def parameter_gradient(model: StatelessModel) -> Callable[[ArrayLike, ArrayLike], Array]:

    """Gradient of a scalar function with respect to its parameters for each 
    point in x.
    
    The input function must take x as its first argument, and unpacked parameters
    as its other arguments. The output has x as its second argument and an 
    iterable of parameters as its first arguments.

    Parameters
    ----------
    model : Callable
        Function to calcuate the gradient of with respect to the parameters
        in its second argument.

    Returns
    -------
    Callable
        Function returning parameter gradient of `f`.

    Examples
    --------    
    >>> import duvida.stateless.numpy as dnp
    >>> f = lambda x, p1, p2: x ** p1 + p2
    >>> x = dnp.array([1., 2.])
    >>> p = dnp.array([2., 1.])
    >>> f(x, *p)
    Array([2., 5.], dtype=float64)
    >>> parameter_gradient(f)(p, x)  # doctest: +NORMALIZE_WHITESPACE
    Array([[0.        , 1.        ],
           [2.77258872, 1.        ]], dtype=float64)

    """
    
    @partial(vmap, in_axes=(None, 0))
    @grad
    def _parameter_gradient(
        params: ArrayLike, 
        x: ArrayLike
    ) -> Array:
        return dnp.sum(model(x, *params))

    return _parameter_gradient


def parameter_hessian_diagonal(
    f: StatelessModel, 
    approximator: str = _DEFAULT_APPROXIMATOR, 
    *args, **kwargs
) -> Callable[[ArrayLike, ArrayLike], Array]:

    """Diagonal of the Hessian of a scalar function with respect to its parameters for each 
    point in x.
    
    The input function must take an iterable of parameters as its second
    argument, and take x values as its first argument. The output has 
    x as its first argument and unpacked parameters as its other arguments.

    Additional arguments are passed to `approximator` function.

    Parameters
    ----------
    f : Callable
        Function to calcuate the gradient of with respect to the parameters
        in its second argument.
    approximator : str, optional
        Named approximator for diagonal of Hessian. Default: use exact Hessian diagonal.
    
    Returns
    -------
    Callable
        Function returning diagonal of the Hessian of `f`.

    Examples
    --------    
    >>> import duvida.stateless.numpy as dnp 
    >>> f = lambda x, p1, p2: x ** p1 + p2
    >>> x = dnp.array([1., 2.])
    >>> p = dnp.array([2., 1.])
    >>> f(x, *p)
    Array([2., 5.], dtype=float64)
    >>> parameter_hessian_diagonal(f)(p, x)  # doctest: +NORMALIZE_WHITESPACE
    Array([[0.        , 0.        ],
           [1.92181206, 0.        ]], dtype=float64)
    >>> parameter_hessian_diagonal(f, approximator='squared_jacobian')(p, x)  # doctest: +NORMALIZE_WHITESPACE
    Array([[0.        , 1.        ],
           [7.68724822, 1.        ]], dtype=float64)
    >>> parameter_hessian_diagonal(f, approximator='rough_finite_difference')(p, x)  # doctest: +NORMALIZE_WHITESPACE
    Array([[0.        , 0.        ],
           [1.92204204, 0.        ]], dtype=float64)
    >>> parameter_hessian_diagonal(f, approximator='bekas', n=3, seed=0)(p, x)  # doctest: +NORMALIZE_WHITESPACE
    Array([[0.        , 0.        ],
           [1.92181206, 0.        ]], dtype=float64)

    """
    
    @partial(vmap, in_axes=(None, 0))
    @get_approximators(approximator, *args, **kwargs)
    def _scalar_f(
        params: ArrayLike, 
        x: ArrayLike
    ) -> Array:
        return dnp.sum(f(x, *params))      

    return _scalar_f


def parameter_gradient_unrolled(
    model: StatelessModel
) -> Callable[[ArrayLike, ArrayLike], Array]:

    @grad
    def _f0(
        params: ArrayLike, 
        x: ArrayLike
    ) -> Array:
        return dnp.sum(model(x, *params))

    def _f(
        params: ArrayLike, 
        x: ArrayLike
    ) -> Array:
        return [_f0(params, [d]) for d in x]

    return _f


def parameter_hessian_diagonal_unrolled(
    model: StatelessModel, 
    approximator: str = _DEFAULT_APPROXIMATOR, 
    *args, **kwargs
) -> Callable[[ArrayLike, ArrayLike], Array]:

    @get_approximators(approximator, *args, **kwargs)
    def _f0(
        params: ArrayLike, 
        x: ArrayLike
    ) -> Array:
        return dnp.sum(model(x, *params))

    def _f(
        params: ArrayLike, 
        x: ArrayLike
    ) -> Array:
        return [_f0(params, [d]) for d in x]

    return _f


def _model_loss(
    model: StatelessModel, 
    loss: LossFunction
) -> Callable[[ArrayLike, ArrayLike, ArrayLike], float]:
    
    def _loss(
        params: ArrayLike, 
        x_true: ArrayLike, 
        y_true: ArrayLike
    ) -> Array:
        return loss(model(x_true, *params), y_true)

    return _loss


def fisher_score(
    model: StatelessModel, 
    loss: LossFunction
) -> Callable[[ArrayLike, ArrayLike, ArrayLike], Array]:
    
    """Returns a function for the Fisher score of a model's loss with respect to
    its parameters.
    
    The input function model takes an iterable of parameters as its second
    argument, and take x values as its first argument. The output has 
    parameter iterable as its first argument, observed x-values as its second 
    argument, and observed y-values as its third argument.

    Parameters
    ----------
    model : Callable
        Function to convert.
    loss : Callable
        Function taking predictions and observations, and giving a loss value.
    
    Returns
    -------
    Callable
        Function returning the Fisher score of `model`'s `loss` with respect to
        its parameters.

    Examples
    --------   
    >>> import duvida.stateless.numpy as dnp  
    >>> model = lambda x, p1, p2: x ** p1 + p2
    >>> mse_fn = lambda ypred, ytrue: dnp.sum(dnp.square(ypred - ytrue))
    >>> x = dnp.array([1., 2.])
    >>> p = dnp.array([0., 2.])
    >>> model(x, *p)
    Array([3., 3.], dtype=float64)
    >>> fisher_score(model, mse_fn)(p, x, model(x, *p) + .1)
    Array([-0.13862944, -0.4       ], dtype=float64)

    """

    return grad(_model_loss(model, loss))


def fisher_information_diagonal(
    model: StatelessModel, 
    loss: LossFunction, 
    approximator: Union[str, Callable] = _DEFAULT_APPROXIMATOR,
    *args, **kwargs
) -> Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Array]:

    """Returns a function for the diagonal of the Fisher information of 
    a model's loss with respect to its parameters.
    
    The input function model takes an iterable of parameters as its second
    argument, and take x values as its first argument. The output has 
    parameter iterable as its first argument, observed x-values as its second 
    argument, and observed y-values as its third argument.

    Additional arguments are passed to `approximator` function.

    Parameters
    ----------
    model : Callable
        Function to convert.
    loss : Callable
        Function taking predictions and observations, and giving a loss value.
    approximator : str, optional
        Named approximator for diagonal of Hessian. Default: use exact Hessian diagonal.
    
    Returns
    -------
    Callable
        Function returning the Fisher score of `model`'s `loss` with respect to
        its parameters.

    Examples
    --------    
    >>> import duvida.stateless.numpy as dnp  
    >>> model = lambda x, p1, p2: x ** p1 + p2
    >>> mse_fn = lambda ypred, ytrue: dnp.sum(dnp.square(ypred - ytrue))
    >>> x = dnp.array([1., 2.])
    >>> p = dnp.array([0., 2.])
    >>> model(x, *p)
    Array([3., 3.], dtype=float64)
    >>> fisher_information_diagonal(model, mse_fn)(p, x, model(x, *p) + .1)
    Array([0.86481543, 4.        ], dtype=float64)

    """

    return get_approximators(approximator)(
        _model_loss(model, loss), 
        *args, **kwargs,
    )


def doubtscore(
    model: StatelessModel, 
    loss: LossFunction, 
    use_reciprocal: bool = False
) -> Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Array]:

    """Given a model and loss function, returns a function for the 
    doubtscore of points in the domain.
    
    The input function model takes an iterable of parameters as its second
    argument, and takes domain x values as its first argument. 
    
    The output function has a parameter iterable as its first argument, 
    domain x values as its second argument, observed x-values as its third 
    argument, and observed y-values as its fourth argument.

    Parameters
    ----------
    model : Callable
        Function to convert.
    loss : Callable
        Function taking predictions and observations, and giving a loss value.
    reciprocal : bool, optional
        Whether the returned function should instead give the reciprocal
        doubtscore. Default: `False`.
    
    Returns
    -------
    Callable
        Function returning the doubtscore for values of the domain and 
        observed data.

    Examples
    --------   
    >>> import duvida.stateless.numpy as dnp   
    >>> model = lambda x, p1, p2: x ** p1 + p2
    >>> mse_fn = lambda ypred, ytrue: dnp.sum(dnp.square(ypred - ytrue))
    >>> x = dnp.array([1., 2.])
    >>> p = dnp.array([0., 2.])
    >>> model(x, *p)
    Array([3., 3.], dtype=float64)
    >>> doubtscore(model, mse_fn)(p, x + .1, x, model(x, *p) + .1)
    Array([[-1.45450818, -0.4       ],
           [-0.1868479 , -0.4       ]], dtype=float64)

    """

    param_grad_fn = parameter_gradient(model)
    fisher_score_fn = fisher_score(model, loss)

    def _doubtscore(params: ArrayLike, x: ArrayLike, 
                    x_true: ArrayLike, y_true: ArrayLike) -> Array:
        return (
            fisher_score_fn(params, x_true, y_true) 
            / param_grad_fn(params, x)
        )

    if use_reciprocal:
        return jit(reciprocal(_doubtscore))
    else:
        return jit(_doubtscore)
    

def _information_sensitivity_term1(
    model: StatelessModel, 
    loss: LossFunction, 
    approximator: Union[str, Callable] = _DEFAULT_APPROXIMATOR,
    *args, **kwargs
) -> Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Array]:
    param_grad_fn = parameter_gradient(model)
    fisher_info_fn = fisher_information_diagonal(model, loss, approximator, 
                                                 *args, **kwargs)

    def _term1(params: ArrayLike, x: ArrayLike, 
               x_true: ArrayLike, y_true: ArrayLike) -> Array:
        return fisher_info_fn(params, x_true, y_true) / param_grad_fn(params, x)

    return _term1


def _information_sensitivity_term2(
    model: StatelessModel, 
    loss: LossFunction, 
    approximator: Union[str, Callable] = _DEFAULT_APPROXIMATOR,
    *args, **kwargs
) -> Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Array]:
    fisher_score_fn = fisher_score(model, loss)
    param_grad_fn = parameter_gradient(model)
    param_hessian_fn = parameter_hessian_diagonal(model, approximator, 
                                                  *args, **kwargs)

    def _term2(
        params: ArrayLike, 
        x: ArrayLike, 
        x_true: ArrayLike, 
        y_true: ArrayLike
    ) -> Array:
        return (fisher_score_fn(params, x_true, y_true) * param_hessian_fn(params, x)
                / dnp.square(param_grad_fn(params, x)))

    return _term2


def information_sensitivity(
    model: StatelessModel, 
    loss: LossFunction, 
    approximator: Union[str, Callable] = _DEFAULT_APPROXIMATOR,
    use_reciprocal: bool = False,
    *args, **kwargs
) -> Callable[[ArrayLike, ArrayLike, ArrayLike, ArrayLike], Array]:

    """Given a model and loss function, returns a function for the 
    inforamtion sensitivity of points in the domain.
    
    The input function model takes an iterable of parameters as its second
    argument, and takes domain x values as its first argument. 
    
    The output function has a parameter iterable as its first argument, 
    domain x values as its second argument, observed x-values as its third 
    argument, and observed y-values as its fourth argument.

    Additional arguments are passed to `approximator` function.

    Parameters
    ----------
    model : Callable
        Function to convert.
    loss : Callable
        Function taking predictions and observations, and giving a loss value.
    pproximator : str, optional
        Named approximator for diagonal of Hessian. Default: use exact Hessian diagonal.
    reciprocal : bool, optional
        Whether the returned function should instead give the reciprocal
        information sensitivity. Default: `False`.
    
    Returns
    -------
    Callable
        Function returning the information sensitivity for values of the domain and 
        observed data.

    Examples
    --------    
    >>> from duvida.stateless.config import config
    >>> model = lambda x, p1, p2: x ** p1 + p2
    >>> mse_fn = lambda ypred, ytrue: dnp.sum(dnp.square(ypred - ytrue))
    >>> x = dnp.array([1., 2.])
    >>> p = dnp.array([0., 2.])
    >>> model(x, *p)
    Array([3., 3.], dtype=float64)
    >>> information_sensitivity(model, mse_fn)(p, x + .1, x, model(x, *p) + .1)
    Array([[9.21232363, 4.        ],
           [1.3042473 , 4.        ]], dtype=float64)

    """

    term1_fn, term2_fn = (
        f(model, loss, approximator, *args, **kwargs) 
        for f in (
            _information_sensitivity_term1, 
            _information_sensitivity_term2,
        )
    )

    def _information_sensitivity(
        params: ArrayLike, 
        x: ArrayLike, 
        x_true: ArrayLike, 
        y_true: ArrayLike
    ) -> Array:
        return term1_fn(params, x, x_true, y_true) - term2_fn(params, x, x_true, y_true)

    if use_reciprocal:
        return jit(reciprocal(_information_sensitivity))
    else:
        return jit(_information_sensitivity)
