"""Utilities for generating hyperparameter plans."""

from typing import Any, Dict, Iterator, Optional, Union
from dataclasses import dataclass
import pickle
from io import TextIOWrapper
from itertools import product
import json

from carabiner import cast, print_err
import numpy as np


@dataclass
class HyperRange:

    """Produce sequences evenly spaced in log or linear space.

    Simply a convenience wrapper around `np.linspace` and `np.geomspace`
    to accept boolean types, ensure types are maintained, and behave 
    like builtin `range`.

    Parameters
    ----------
    min : float | int | bool
        Minimum value when `max` provided, else the maximum value of the range. 
    max : float | int | bool, optional
        Maximum value of the range if minimum is provided.
    n_step : int
        Number of steps in the sequence. Default: 2 if 
        `min` and `max` are different, else 1.
    log_scale : bool
        Whether to have log spacing in the sequence. Default: `False`.

    Returns
    -------
    HyperRange

    Examples
    --------
    >>> HyperRange(1.)
    HyperRange: [ 1.0 ]
    >>> HyperRange(1., 10.)
    HyperRange: [ 1.0, 10.0 ]
    >>> HyperRange(1., 100., 20)
    HyperRange: [ 1.0, 6.2105263157894735, ..., 94.78947368421052, 100.0 ]
    >>> HyperRange(False, True)
    HyperRange: [ False, True ]
    >>> HyperRange(False, True) == HyperRange(True, False)
    True
    >>> HyperRange(False, True).min
    False
    >>> HyperRange(False, True).max
    True
    >>> HyperRange(1, 100).max
    100
    >>> HyperRange(1, 100).min
    1
    >>> HyperRange(1, 100.1).max  # Output type is the same as `start`
    100
    >>> list(HyperRange(1, 100))
    [1, 100]
    
    """
    
    start: Union[float, int, bool]
    stop: Optional[Union[float, int, bool]] = None
    n_step: Optional[int] = None
    log_scale: bool = False

    def __post_init__(self):
        range_fn = np.linspace if not self.log_scale else np.geomspace
        if self.stop is None:
            self.stop = self.start
        if self.n_step is None:
            if self.stop == self.start:
                self.n_step = 1
            else:
                 self.n_step = 2
        self.range = sorted(set(
            range_fn(
                self.start, 
                self.stop, 
                num=self.n_step,
            )
            .astype(type(self.start))
            .tolist()
        ))

    def __iter__(self) -> Iterator[Union[float, int, bool]]:
        yield from self.range

    def __len__(self) -> int:
        return len(self.range)

    def __eq__(self, b) -> bool:
        if isinstance(b, type(self)):
            return self.range == b.range
        else:
            return self.range == b

    def __repr__(self) -> str:
        prefix = "HyperRange"

        if len(self) > 4:
            lead = ', '.join(map(str, self.range[:2]))
            trail = ', '.join(map(str, self.range[-2:]))
            return f"{prefix}: [ {lead}, ..., {trail} ]"
        else:
            return f"{prefix}: [ {', '.join(map(str, self.range))} ]"

    def __str__(self) -> str:
        return str(self)
        
    @property
    def min(self) -> Union[float, int, bool]:
        """Get minimum value.

        Returns
        -------
        float | int | bool
            Minimum value of HyperRange.

        """
        return min(self.range)

    @property
    def max(self) -> Union[float, int, bool]:
        """Get maximum value.

        Returns
        -------
        float | int | bool
            Maximum value of HyperRange.

        """
        return max(self.range)
          

class HyperOpt(dict):

    """Construct a parameter grid based on the provided ranges.

    Examples
    --------
    >>> HyperOpt(a=1.)
    {'a': HyperRange: [ 1.0 ]}
    >>> HyperOpt(a=1., b=True)
    {'a': HyperRange: [ 1.0 ], 'b': HyperRange: [ True ]}
    >>> configs = HyperOpt(a=1., b=True, d=HyperRange(50, 53, 4))
    >>> len(configs)
    4
    >>> configs
    {'a': HyperRange: [ 1.0 ], 'b': HyperRange: [ True ], 'd': HyperRange: [ 50, 51, 52, 53 ]}
    >>> for conf in configs: print(conf)
    ... 
    {'a': 1.0, 'b': True, 'd': 50}
    {'a': 1.0, 'b': True, 'd': 51}
    {'a': 1.0, 'b': True, 'd': 52}
    {'a': 1.0, 'b': True, 'd': 53}

    """

    def __init__(self, silent: bool = False, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        for key in self.keys():
            val = self[key]
            if not isinstance(val, HyperRange):
                val = cast(val, to=list)
                if isinstance(val[0], (float, int, bool)) and len(val) <= 2:
                    self[key] = HyperRange(*val)
                else:
                    self[key] = val
        iterator = product(*(self.get(key) for key in self.keys()))
        configs = tuple(dict(zip(self.keys(), args)) for args in iterator)
        if not silent:
            print_err(f"There are {len(configs)} configurations to test.")
        self._ranges = configs
        self._keys = tuple(self._ranges[0])

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from self._ranges

    def __len__(self) -> int:
        return len(self._ranges)

    @classmethod
    def from_file(
        cls, 
        file: Union[str, TextIOWrapper], 
        serialized: bool = False,
        **kwargs
    ):
        """Create a HyperOpt object from a JSON or serialized file.

        """
        if not serialized:
            f = cast(file, to=TextIOWrapper)
            d = json.load(f)
            return cls(**kwargs, **d)
        else:
            with open(file, "rb") as f:
                obj = pickle.load(f)
            return obj
            
    def write(
        self, 
        file: Union[str, TextIOWrapper],
        serialize: bool = False
    ) -> None:
        """Save a HyperOpt object in a reloadable format.

        """
        file = cast(file, str)
        open_mode = "wb" if serialize else "w"
        with open(file, open_mode) as f:
            if serialize:
                pickle.dump(obj=self, file=f)
            else:
                json.dump(
                    {key: list(val) for key, val in self.items()}, 
                    f,
                    sort_keys=True,
                    indent=4,
                )
        return None
