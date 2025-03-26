"""ModelBoxes specifically for chemistry."""

from typing import Callable, Dict, Iterable, Mapping, Optional, Union
from abc import abstractmethod, ABC

from functools import partial

from carabiner import cast, print_err
from chemprop.data import Datum, MoleculeDatapoint, MoleculeDataset, MolGraph
from datasets import Dataset
import numpy as np
from numpy import asarray
from numpy.typing import ArrayLike
from schemist.features import calculate_feature

from .base_classes import ModelBoxBase

class FPModelBoxMixinBase(ABC):
        
    @staticmethod
    def preprocess_data(
        data: Mapping[str, ArrayLike],
        _in_key: str, 
        _out_key: str,
        use_fp: bool = True,
        use_2d: bool = False,
        extra_featurizers: Optional[Union[Iterable[Callable], Callable]] = None,
        _allow_no_features: bool = False
    ) -> Dict[str, np.ndarray]:
        """Convert to binary radius-2 Morgan fingerprints with 2048 bits.

        Optionally use normalized 2D descriptors.
        """
        if all([
            not use_fp, 
            not use_2d, 
            extra_featurizers is None, 
            not _allow_no_features,
        ]):  # i.e. no features specified
            print_err("WARNING : FPModelBoxMixinBase : Specified both `use_fp=False` and `use_2d=False`, so defaulting to `use_fp=True`.")
            use_fp = True

        extra_features = []
        if use_fp:
            fingerprints, _ = calculate_feature(
                'fp', 
                strings=(x[0] for x in data[_in_key]), 
                on_bits=False,
            )
            extra_features.append(fingerprints)
        if use_2d:
            desc_2d, _ = calculate_feature(
                '2d', 
                strings=(x[0] for x in data[_in_key]), 
            )
            extra_features.append(desc_2d)
        if extra_featurizers is not None:
            extra_featurizers = cast(extra_featurizers, to=list)
            for featurizer in extra_featurizers:
                extra_features.append(featurizer(data[_in_key]))
        if len(extra_features) > 0:
            extra_features = np.concatenate(extra_features, axis=-1)

        return {
            _in_key: asarray(extra_features).astype("float"), 
            _out_key: asarray(data[_out_key])
        }

    def _ingest_data(
        self,
        *args, **kwargs
    ) -> None:

        """Load dataset used for training.
        
        """
        ingest_kwargs = {
            "use_fp": self.use_fp,
            "use_2d": self.use_2d,
            "extra_featurizers": self.extra_featurizers,
        }
        ingest_kwargs.update(kwargs)
        return super()._ingest_data(*args, **ingest_kwargs)

    @staticmethod
    @abstractmethod
    def _get_max_sim(
        q: ArrayLike, 
        references: ArrayLike
    ):
        pass

    @staticmethod
    @abstractmethod
    def _get_nn_tanimoto(
        queries: Mapping[str, ArrayLike],
        refs_data: Mapping[str, ArrayLike],
        _in_key: str,
        _out_key: str,
        _sim_fn: Callable
    ):
        pass

    def tanimoto_nn(
        self, 
        candidates: Union[ArrayLike, Dataset],
        batch_size: int = 16,
        _in_key: Optional[str] = None,
        **kwargs
    ) -> Dataset:
        """Get Tanimoto similarity of nearest training set data.
    
        """
        if _in_key is None:
            _in_key = self._in_key
        if isinstance(self, ModelBoxBase):
            queries = self._prepare_data(
                data=candidates, 
                use_fp=True,
                use_2d=False,
                extra_featurizers=None,
                **kwargs
            )
            refs = (
                self._input_training_data
                .map(
                    partial(
                        self.preprocess_data,
                        _in_key=self._in_key, 
                        _out_key=self._out_key,
                        use_fp=True,
                        use_2d=False,
                        extra_featurizers=None,
                        **kwargs
                    ),
                    batched=True,
                    batch_size=batch_size,
                    desc="Calculating fingerprints",
                )
                .with_format(self._format, **self._format_kwargs)
            )
            # refs = self._prepare_data(
            #     data=self._input_training_data, 
            #     features=self._in_key,
            #     labels=self._out_key,
            #     use_fp=True,
            #     use_2d=False,
            #     extra_featurizers=None,
            #     **kwargs
            # )
            _nn_tanimoto = queries.map(
                partial(
                    self._get_nn_tanimoto, 
                    refs_data=refs,
                    _in_key=_in_key, 
                    _out_key=self._out_key,
                    _sim_fn=self._get_max_sim,
                ),
                batched=True,
                batch_size=batch_size,
                desc="Calculating Tanimoto similarity to nearest training neighbor",
            )
            return _nn_tanimoto
        else:
            raise ValueError("FPModelBoxMixin.tanimoto_nn() can only be used with ModelBox!") 

    # def tanimoto_nn(
    #     self, 
    #     candidates: Union[ArrayLike, Dataset],
    #     batch_size: int = 16,
    #     **kwargs
    # ) -> Dataset:
    #     """Get Tanimoto simialrity of nearest training set data.
    
    #     """
    #     if isinstance(self, ModelBoxBase):
    #         queries = self._prepare_data(
    #             data=candidates, 
    #             **kwargs
    #         )
    #         _nn_tanimoto =  queries.map(
    #             partial(
    #                 self._get_nn_tanimoto, 
    #                 refs_data=self.training_data,
    #                 _in_key=self._in_key, 
    #                 _out_key=self._out_key,
    #                 _sim_fn=self._get_max_sim,
    #             ),
    #             batched=True,
    #             batch_size=batch_size,
    #             desc="Calculating Tanimoto similarity to nearest training neighbor",
    #         )
    #         return _nn_tanimoto
    #     else:
    #         raise ValueError("FPModelBoxMixin.tanimoto_nn() can only be used with ModelBox!")


class ChempropModelBoxMixinBase(FPModelBoxMixinBase):
        
    @staticmethod
    def preprocess_data(
        data: Mapping[str, ArrayLike],
        _in_key: str, 
        _out_key: str,
        use_fp: bool = False,
        use_2d: bool = False,
        extra_featurizers: Optional[Union[Iterable[Callable], Callable]] = None
    ) -> Dict[str, np.ndarray]:

        """Convert to Chemprop data structure.
        """

        vectors = FPModelBoxMixinBase.preprocess_data(
            data=data,
            _in_key=_in_key, 
            _out_key=_out_key,
            use_fp=use_fp,
            use_2d=use_2d,
            extra_featurizers=extra_featurizers,
            _allow_no_features=True,
        )
        extra_features = vectors[_in_key]

        # extra_features = []
        # if use_fp:
        #     fingerprints, _ = calculate_feature(
        #         'fp', 
        #         strings=(x[0] for x in data[_in_key]), 
        #         on_bits=False,
        #     )
        #     extra_features.append(fingerprints.astype(np.float32))
        # if use_2d:
        #     desc_2d, _ = calculate_feature(
        #         '2d', 
        #         strings=(x[0] for x in data[_in_key]), 
        #     )
        #     extra_features.append(desc_2d.astype(np.float32))
        if len(extra_features) > 0:
            # extra_features = np.concatenate(extra_features, axis=-1)
            extra_features = extra_features.astype(np.float32)
            mol_datapoints = [
                MoleculeDatapoint.from_smi(smi[0], y, x_d=xd) 
                for smi, y, xd in zip(data[_in_key], asarray(data[_out_key]), extra_features)
            ]
        else:
            extra_features = [None for _ in data[_in_key]]
            mol_datapoints = [
                MoleculeDatapoint.from_smi(smi[0], y) 
                for smi, y in zip(data[_in_key], asarray(data[_out_key]))
            ]
        
        mol_dataset = MoleculeDataset(mol_datapoints)
        datums = []
        for datum in mol_dataset:
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

        return {
            _in_key: datums, 
            _out_key: asarray(data[_out_key]),
            "extra_features": extra_features,
            "smiles": data[_in_key]
        }

    def tanimoto_nn(
        self, 
        candidates: Union[ArrayLike, Dataset],
        batch_size: int = 16,
        **kwargs
    ) -> Dataset:
        return super().tanimoto_nn(
            candidates, 
            batch_size, 
            _in_key="extra_features", 
            **kwargs
        )

    # def _ingest_data(
    #     self,
    #     *args, **kwargs
    # ) -> None:

    #     """Load dataset used for training.
        
    #     """
    #     ingest_kwargs = {
    #         "use_fp": self.use_fp,
    #         "use_2d": self.use_2d,
    #     }
    #     ingest_kwargs.update(kwargs)
    #     return super()._ingest_data(*args, **ingest_kwargs)

    # def tanimoto_nn(
    #     self, 
    #     candidates: Union[ArrayLike, Dataset],
    #     batch_size: int = 16,
    #     **kwargs
    # ) -> Dataset:
    #     """Get Tanimoto similarity of nearest training set data.
    
    #     """
    #     if isinstance(self, ModelBoxBase):
    #         queries = self._prepare_data(
    #             data=candidates, 
    #             use_fp=True,
    #             use_2d=False,
    #             **kwargs
    #         )
    #         refs = self._prepare_data(
    #             data=self._input_training_data, 
    #             features=self._in_key,
    #             labels=self._out_key,
    #             use_fp=True,
    #             use_2d=False,
    #             **kwargs
    #         )
    #         _nn_tanimoto = queries.map(
    #             partial(
    #                 self._get_nn_tanimoto, 
    #                 refs_data=refs,
    #                 _in_key="extra_features", 
    #                 _out_key=self._out_key,
    #                 _sim_fn=self._get_max_sim,
    #             ),
    #             batched=True,
    #             batch_size=batch_size,
    #             desc="Calculating Tanimoto Tanimoto similarity to nearest training neighbor",
    #         )
    #         return _nn_tanimoto
    #     else:
    #         raise ValueError("ChempropModelBoxMixin.tanimoto_nn() can only be used with ModelBox!") 
