import numpy as np
import pytest
from duvidnn.base.evaluation import rmse, pearson_r, spearman_r


def test_rmse_simple():
    pred = [1., 2., 3.]
    truth = [1., 2., 4.]
    assert rmse(pred, truth) == pytest.approx(np.sqrt(((np.asarray(pred) - np.asarray(truth)) ** 2).mean()))


def test_rmse_ensemble():
    pred = [[1., 2.], [2., 3.], [3., 4.]]
    truth = [1., 2., 4.]
    expected = np.sqrt(((np.asarray(pred).mean(axis=-1) - np.asarray(truth)) ** 2).mean())
    assert rmse(pred, truth) == pytest.approx(expected)


def test_rmse_broadcast():
    pred = [1., 2., 3.]
    truth = [[1.], [2.], [4.]]
    expected = np.sqrt(((np.asarray(pred) - np.asarray(truth)) ** 2).mean())
    assert rmse(pred, truth) == pytest.approx(expected)


def test_rmse_fail_shape():
    pred = np.ones((2, 2))
    truth = np.ones(3)
    with pytest.raises(ValueError):
        rmse(pred, truth)


def test_pearson_and_spearman():
    x = [[1., 2.], [2., 4.]]
    y = [1., 3.]
    r = pearson_r(x, y)
    rho = spearman_r(x, y)
    assert r == pytest.approx(1.)
    assert rho == pytest.approx(1.)
