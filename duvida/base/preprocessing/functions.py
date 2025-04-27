"""Data preprocessing functions."""

from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Union
from functools import partial

from chemprop.data import (
    MoleculeDatapoint, 
    MoleculeDataset, 
    MolGraph
)
import numpy as np
from schemist.features import calculate_feature

from .registry import register_function


@register_function("identity")
def Identity() -> Callable:
    """Simple pass-through.
    
    """
    def _identity(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        return np.asarray(data[input_column])
    return _identity


@register_function("one-hot")
def OneHot(
    categories: Iterable[str],
    intercept: bool = False
) -> Callable:
    """Convert string labels into one-hot encodings.
    
    """
    prepend = [1] if intercept else []

    def _one_hot(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        return np.asarray([
            prepend + [1 if x == cat else 0 for cat in categories] 
            for x in data[input_column]
        ])

    return _one_hot
    

@register_function("morgan-fingerprint")
def MorganFingerprint(**kwargs) -> Callable:
    """Get Morgan fingerprint from SMILES.
    
    """
    feature_calculator = partial(
        calculate_feature,
        feature_type="fp",
        return_dataframe=False,
        on_bits=False,
        **kwargs,
    )

    def _morgan_fingerprint(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        fingerprints, _ = feature_calculator(strings=data[input_column])
        return fingerprints
        
    return _morgan_fingerprint


@register_function("descriptors-2d")
def Descriptors2D(
    normalized: bool = True,
    histogram_normalized: bool = True
) -> Callable:
    """Get 2D descriptors from SMILES, optionally normalized.
    
    """
    feature_calculator = partial(
        calculate_feature,
        feature_type="2d",
        return_dataframe=False,
        normalized=normalized,
        histogram_normalized=histogram_normalized,
    )
    
    def _descriptors_2d(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> np.ndarray:
        desc_2d, _ = feature_calculator(strings=data[input_column])
        return desc_2d

    return _descriptors_2d


@register_function("chemprop-mol")
def ChempropData(
    label_column: Optional[Union[str, Iterable[str]]] = None,
    extra_featurizers: Optional[Union[Mapping[str, Any], Iterable[Mapping[str, Any]]]] = None
) -> Callable:
    """Convert SMILES to iterable of Chemprop datum.
    
    """
    if isinstance(extra_featurizers, str):
        extra_featurizers = [extra_featurizers]
    if isinstance(label_column, str):
        label_column = [label_column]

    def _stack_columns(
        data: Mapping[str, Iterable],
        nrows: int,
        columns: Optional[Iterable[str]] = None
    ) -> np.ndarray:
        if columns is None:
            array = [None] * nrows
        else: 
            array = [np.asarray(data[col]) for col in columns]
            array = [a if a.ndim > 1 else a[..., np.newaxis] for a in array]
            if len(array) > 0:
                array = np.concatenate(array, axis=-1).astype(np.float32) 
            else:
                array = [None] * nrows
        return array
    
    def _chemprop_data(
        data: Mapping[str, Iterable],
        input_column: str
    ) -> List[Dict[str, np.ndarray]]:
        nrows = len(data[input_column])
        y_vals = _stack_columns(data, nrows, label_column)
        extra_features = _stack_columns(data, nrows, extra_featurizers)

        mol_datapoints = [
            MoleculeDatapoint.from_smi(smi=x, y=y, x_d=xd) 
            for x, y, xd in zip(data[input_column], y_vals, extra_features)
        ]
    
        datums = []
        for datum in MoleculeDataset(mol_datapoints):
            new_datum = {}
            for key, val in datum._asdict().items():
                if isinstance(val, MolGraph):
                    new_val = {
                        key2: val2.astype(np.float32) if isinstance(val2, np.ndarray) else np.float32(val2) 
                        for key2, val2 in val._asdict().items()
                    }
                elif isinstance(val, float):
                    new_val = np.float32(val)
                elif isinstance(val, np.ndarray):
                    new_val = val.astype(np.float32)
                elif val is not None:
                    new_val = val
                else:
                    new_val = None
                new_datum[key] = new_val
            datums.append(new_datum)
        return datums

    return _chemprop_data
