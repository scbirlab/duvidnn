"""Modelbox data aspects."""

from typing import Callable, Dict, Mapping, Union
from functools import partial

from datasets import Dataset, IterableDataset
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

from ...base.data import ChemMixinBase, DataMixinBase, _DEFAULT_BATCH_SIZE
from ...utils.package_data import CACHE

_DEVICE = "cpu"

class XGBIterator(xgboost.DataIter):
    """A custom iterator for loading files in batches."""

    def __init__(
        self, 
        ds, 
        in_key: str, 
        out_key: str,
        batch_size: int = _DEFAULT_BATCH_SIZE, 
        cache: Optional[str] = None
    ) -> None:
        self._it = 0
        self.ds = ds
        self.batch_size = batch_size
        # XGBoost will generate some cache files under the current directory with the
        # prefix "cache"
        super().__init__(cache_prefix=cache or CACHE)

    def next(self, input_data: Callable) -> bool:
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function
        is called by XGBoost during the construction of ``DMatrix``

        """
        if self._it >= np.ceil(self.ds.num_rows / batch_size):
            # return False to let XGBoost know this is the end of iteration
            return False

        # input_data is a keyword-only function passed in by XGBoost and has the similar
        # signature to the ``DMatrix`` constructor.
        item = self.ds[self._it]
        input_data(data=item[self._in_key], label=item[self._out_key])
        self._it += 1
        return True

    def reset(self) -> None:
        """Reset the iterator to its beginning"""
        self._it = 0


class XGBDataMixin(DataMixinBase):

    _format = 'numpy'
    _format_kwargs = {
        'dtype': np.float32, 
        'device': _DEVICE,
    }

    @staticmethod
    def make_dataloader(
        dataset: Union[Dataset, IterableDataset],
        batch_size: int = _DEFAULT_BATCH_SIZE, 
        cache: Optional[str] = None,
        **kwargs
    ):
        return xgb.ExtMemQuantileDMatrix(
            XGBIterator(
                dataset,
                DataMixinBase._in_key,
                DataMixinBase._out_key,
                cache=cache,
            )
        )
