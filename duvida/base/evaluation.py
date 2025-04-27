"""Utilities for evaluating predictions."""

import numpy as np
import scipy


def rmse(x, y):
    return np.mean(np.abs(np.mean(x, axis=-1, keepdims=True) - y))


def pearson_r(x, y):
    corr = np.corrcoef(np.mean(x, axis=-1, keepdims=True), y, rowvar=False)
    return np.diag(corr, k=1).flatten()[0]


def spearman_r(x, y):
    return scipy.stats.spearmanr(
        np.mean(x, axis=-1).flatten(), 
        y.flatten(), 
        nan_policy="omit",
    ).statistic
