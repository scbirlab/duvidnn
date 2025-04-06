"""Custom types."""

from typing import Any, Iterable, Mapping, Union

from datasets import Dataset, IterableDataset
from pandas import DataFrame
from numpy.typing import ArrayLike

DataLike = Union[
    str, 
    DataFrame, 
    Mapping[str, ArrayLike], 
    Dataset, 
    IterableDataset
]
FeatureLike = Union[
    str, 
    Mapping[str, 'Preprocessor'], 
    Mapping[str, Mapping[str, Any]], 
    Iterable[Union[
        str, 
        Mapping[str, 'Preprocessor'], 
        Mapping[str, Mapping[str, Any]]
    ]]
]
StrOrIterableOfStr = Union[str, Iterable[str]]