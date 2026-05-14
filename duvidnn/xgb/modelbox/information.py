# xgb/information.py
from __future__ import annotations

from typing import Callable, Dict, Optional

import json
import sys

import numpy as np
from numpy.typing import ArrayLike
try:
    import xgboost as xgb
except ImportError:
    from carabiner import print_err
    print_err(
        """
        [ERROR] XGBoost not installed! Try:
            $ pip install duvidnn[xgb]
        """
    )
    sys.exit(1)

from ...base.information import DoubtMixinBase

_EPS = 1e-12


class XGBDoubtMixin(DoubtMixinBase):
    """
    Implements the info metrics for XGBoost regressors by treating each leaf value
    as a "parameter". Works with base/information.py unchanged.

    Expected:
      self.model: xgboost.Booster (or sklearn XGBRegressor.get_booster()).
    """

    device = "cpu"

    @staticmethod
    def _get_shape(x: ArrayLike):
        return np.asarray(x).shape

    @staticmethod
    def release_memory() -> None:
        # no-op; XGBoost manages its own caches
        return None

    # ---------- leaf index bookkeeping ----------

    def _ensure_leaf_index(self, booster: xgb.Booster) -> None:
        if getattr(self, "_n_leaf_params", None) is not None:
            return

        dump = booster.get_dump(dump_format="json")  # list[str], per tree

        tree_leaf_to_pos = []
        tree_offsets = []
        offset = 0

        def collect_leaf_nodeids(node: dict, acc: list[int]) -> None:
            if "leaf" in node:
                acc.append(int(node["nodeid"]))
                return
            for ch in node.get("children", []):
                collect_leaf_nodeids(ch, acc)

        for t_json in dump:
            t = json.loads(t_json)
            leaf_ids: list[int] = []
            collect_leaf_nodeids(t, leaf_ids)

            leaf_ids = sorted(set(leaf_ids))
            mapping = {nodeid: j for j, nodeid in enumerate(leaf_ids)}
            tree_leaf_to_pos.append(mapping)
            tree_offsets.append(offset)
            offset += len(mapping)

        self._tree_leaf_to_pos = tree_leaf_to_pos
        self._tree_offsets = tree_offsets
        self._n_leaf_params = offset

    def _leaf_indices_to_global(self, leaf_nodeids: np.ndarray) -> np.ndarray:
        """
        leaf_nodeids: (B, T) int nodeid per tree from pred_leaf=True
        returns: (B, T) global parameter indices
        """
        B, T = leaf_nodeids.shape
        out = np.empty((B, T), dtype=np.int64)
        for t in range(T):
            mapping = self._tree_leaf_to_pos[t]
            base = self._tree_offsets[t]
            for i in range(B):
                out[i, t] = base + mapping[int(leaf_nodeids[i, t])]
        return out

    # ---------- required API ----------

    def parameter_gradient(
        self,
        model: Optional[xgb.Booster] = None,
        last_layer_only: bool = False,  # not meaningful here
    ) -> Callable[[ArrayLike], Dict[str, np.ndarray]]:
        booster = self.model if model is None else model
        self._ensure_leaf_index(booster)

        def fn(x: ArrayLike) -> Dict[str, np.ndarray]:
            X = np.asarray(x)
            d = xgb.DMatrix(X)
            leaf = np.asarray(booster.predict(d, pred_leaf=True))  # (B,T)
            idx = self._leaf_indices_to_global(leaf)

            B, T = idx.shape
            G = np.zeros((B, self._n_leaf_params), dtype=np.float32)
            for i in range(B):
                G[i, idx[i, :]] = 1.0
            return {"leaf_values": G}

        return fn

    def fisher_score(
        self,
        model: Optional[xgb.Booster] = None,
        loss=None,  # ignored for MSE; kept for signature compat
        last_layer_only: bool = False,
    ) -> Callable[[ArrayLike, ArrayLike], Dict[str, np.ndarray]]:
        booster = self.model if model is None else model
        self._ensure_leaf_index(booster)

        def fn(x_true: ArrayLike, y_true: ArrayLike) -> Dict[str, np.ndarray]:
            X = np.asarray(x_true)
            y = np.asarray(y_true).reshape(-1)
            d = xgb.DMatrix(X)

            yhat = np.asarray(booster.predict(d)).reshape(-1)
            # assuming 0.5*(yhat-y)^2
            g = (yhat - y).astype(np.float32)

            leaf = np.asarray(booster.predict(d, pred_leaf=True))
            idx = self._leaf_indices_to_global(leaf)

            B, T = idx.shape
            S = np.zeros((B, self._n_leaf_params), dtype=np.float32)
            # per-sample contribution in visited leaf slots
            for i in range(B):
                S[i, idx[i, :]] = np.abs(g[i])
            return {"leaf_values": S}

        return fn

    def fisher_information_diagonal(
        self,
        model: Optional[xgb.Booster] = None,
        loss=None,  # ignored for MSE
        approximator: str = "exact",  # ignored
        last_layer_only: bool = False,
        *args, **kwargs,
    ) -> Callable[[ArrayLike, ArrayLike], Dict[str, np.ndarray]]:
        booster = self.model if model is None else model
        self._ensure_leaf_index(booster)

        def fn(x_true: ArrayLike, y_true: ArrayLike) -> Dict[str, np.ndarray]:
            X = np.asarray(x_true)
            d = xgb.DMatrix(X)

            # For 0.5*(yhat-y)^2, d2L/dyhat2 = 1 per sample
            h = np.ones((X.shape[0],), dtype=np.float32)

            leaf = np.asarray(booster.predict(d, pred_leaf=True))
            idx = self._leaf_indices_to_global(leaf)

            B, T = idx.shape
            H = np.zeros((B, self._n_leaf_params), dtype=np.float32)
            for i in range(B):
                H[i, idx[i, :]] = h[i]
            return {"leaf_values": H}

        return fn

    def parameter_hessian_diagonal(
        self,
        model: Optional[xgb.Booster] = None,
        approximator: str = "exact",
        last_layer_only: bool = False,
        *args, **kwargs,
    ) -> Callable[[ArrayLike], Dict[str, np.ndarray]]:
        booster = self.model if model is None else model
        self._ensure_leaf_index(booster)

        def fn(x: ArrayLike) -> Dict[str, np.ndarray]:
            X = np.asarray(x)
            Z = np.zeros((X.shape[0], self._n_leaf_params), dtype=np.float32)
            return {"leaf_values": Z}

        return fn

    # ---------- cores (NumPy) ----------

    @staticmethod
    def doubtscore_core(fisher_score, parameter_gradient, eps=_EPS, device="cpu"):
        fs = np.asarray(fisher_score, dtype=np.float32)         # (P,)
        pg = np.asarray(parameter_gradient, dtype=np.float32)   # (B,P)
        return pg / (fs.reshape(1, -1) + eps)

    @staticmethod
    def information_sensitivity_core(
        fisher_score,
        fisher_information_diagonal,
        parameter_gradient,
        parameter_hessian_diagonal=None,
        eps=_EPS,
        optimality_approximation: bool = False,
        device="cpu",
    ):
        fi = np.asarray(fisher_information_diagonal, dtype=np.float32)  # (P,)
        pg = np.asarray(parameter_gradient, dtype=np.float32)           # (B,P)
        # term2 is zero here; prediction is linear in leaf values
        return (pg * pg) / (pg * fi.reshape(1, -1) + eps)
