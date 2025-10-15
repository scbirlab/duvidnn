"""Base mixins for data."""

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Mapping, Tuple, Optional, Union

from abc import abstractmethod, ABC
from functools import partial
import os

from carabiner import cast, print_err

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset
    from pandas import DataFrame
else:
    Dataset, DatasetDict, IterableDataset, DataFrame = Any, Any, Any, Any

import numpy as np
from numpy.typing import ArrayLike
from schemist.converting import convert_string_representation

from ..checkpoint_utils import load_checkpoint_file, save_json
from .preprocessing import Preprocessor
from .typing import DataLike, FeatureLike, StrOrIterableOfStr

_DEFAULT_BATCH_SIZE: int = 128


class DataMixinBase(ABC):

    """Add data-loading capability to models.
    
    """

    _default_cache: str = "cache/duvida/data"
    _default_preprocessing_args: dict = {}
    _in_key: str = 'inputs'
    _out_key: str = 'labels'
    _input_training_data = None
    _input_featurizers = None
    _input_cols = None
    _label_cols = None
    _format: str = 'numpy'
    _format_kwargs: Optional[Mapping[str, Any]] = None
    training_data = None
    training_example = None
    input_shape = None 
    output_shape = None

    def save_data_checkpoint(
        self, 
        checkpoint_dir: str
    ):
        keys = (
            "_in_key",
            "_out_key",
            "_input_cols",
            "_label_cols",
            "_input_featurizers",
            "input_shape",
            "output_shape",
            "_default_cache",
            "_default_preprocessing_args",
        )
        data_config = {
            key: getattr(self, key) for key in keys
        }
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if self._input_training_data is not None:
            self._input_training_data.save_to_disk(
                os.path.join(os.path.join(checkpoint_dir, "input-data.hf")),
            )
            if self.training_data is not None:
                (
                    self.training_data
                    .with_format("numpy", dtype="float")
                    .save_to_disk(os.path.join(checkpoint_dir, "training-data.hf")),
                )
        save_json(data_config, os.path.join(checkpoint_dir, "data-config.json"))
        return None

    def load_data_checkpoint(
        self, 
        checkpoint: str,
        cache_dir: Optional[str] = None
    ):
        data_config = load_checkpoint_file(
            checkpoint, 
            filename="data-config.json",
            callback="json",
            none_on_error=False,
            cache_dir=cache_dir,
        )
        for key, val in data_config.items():
            setattr(self, key, val)
        self._input_training_data = load_checkpoint_file(
            checkpoint, 
            filename="input-data.hf",
            callback="hf-dataset",
            none_on_error=True,
            cache_dir=cache_dir,
        )
        training_data = load_checkpoint_file(
            checkpoint, 
            filename="training-data.hf",
            callback="hf-dataset",
            none_on_error=True,
            cache_dir=cache_dir,
        )
        if training_data is not None:
            self.training_data = training_data.with_format(
                self._format, 
                **self._format_kwargs,
            )
            self.training_example = (
                self.training_data
                .take(1)
                .with_format('numpy')
            )
        return self

    @staticmethod
    def _concat_features(
        x: Mapping[str, ArrayLike],
        inputs: StrOrIterableOfStr,
        labels: StrOrIterableOfStr,
        _in_key: str = "inputs",
        _out_key: str = "labels"
    ) -> Dict[str, np.ndarray]:
        
        if isinstance(inputs, str):
            x = {
                _in_key: x[inputs],
                _out_key: [np.asarray(x[col]) for col in cast(labels, to=list)]
            }
            x[_out_key] = np.concatenate([
                col if col.ndim > 1 else col[..., np.newaxis] 
                for col in x[_out_key]
            ], axis=-1)
            return x
        else:
            cols_to_concat = {
                _in_key: cast(inputs, to=list),
                _out_key: cast(labels, to=list),
            }
            x = {
                key: [np.asarray(x[col]) for col in columns]
                for key, columns in cols_to_concat.items()
            }
            return {
                key: np.concatenate([
                    col if col.ndim > 1 else col[..., np.newaxis] 
                    for col in columns
                ], axis=-1)
                for key, columns in x.items()
            }

    @staticmethod
    def _check_is_calculated(
        x: Dataset,  
        preprocessor: Preprocessor
    ) -> Tuple[str, bool]:
        out_column = preprocessor.output_column
        return out_column, out_column in x.column_names

    @staticmethod
    def _featurize(
        x: Mapping[str, ArrayLike],
        featurizers: Iterable[Mapping[str, Any]]
    ) -> Dict[str, np.ndarray]:

        for featurizer in featurizers:
            if isinstance(featurizer, Mapping):
                featurizer = Preprocessor.from_dict(featurizer)
            elif not isinstance(featurizer, Preprocessor):
                raise ValueError(f"Featurizer is neither a dict nor a `Preprocessor`: {featurizer}")
            if featurizer.output_column not in x:
                x = featurizer(x)
            else:
                print_err(f"INFO: {featurizer.output_column} already present, skipping.")
        return x

    @staticmethod
    def _check_column_presence(
        features: StrOrIterableOfStr, 
        labels: StrOrIterableOfStr,
        data: Dataset
    ) -> Iterable[str]:
        columns = cast(features, to=list) + cast(labels, to=list)
        data_cols = data.column_names
        absent_columns = [col for col in columns if col not in data_cols]
        if len(absent_columns) > 0:
            raise ValueError(
                f"""
                Requested columns ({', '.join(columns)}) not present in 
                {type(data)}: {', '.join(absent_columns)}.
                """
            )
        return columns
    
    @staticmethod
    def _load_from_csv(
        filename: str,
        cache: Optional[str] = None
    ) -> Dataset:
        from datasets import Dataset

        if filename.endswith((".csv", ".tsv", ".txt", ".csv.gz", ".tsv.gz", ".txt.gz")):
            read_f = partial(
                Dataset.from_csv,
                cache_dir=cache,
                sep="," if filename.endswith((".csv", ".csv.gz")) else "\t",
            )
        elif filename.endswith(".parquet"):
            read_f = partial(Dataset.from_parquet, cache_dir=cache)
        elif filename.endswith(".hf"):
            read_f = Dataset.load_from_disk
        elif filename.endswith(".arrow"):
            read_f = Dataset.from_file
        else:
            raise IOError(f"Could not infer how to open '{filename}' from its extension.")
        return read_f(filename)

    @classmethod
    def _load_from_dataframe(
        cls,
        dataframe: Union[DataFrame, Mapping[str, ArrayLike]],
        cache: Optional[str] = None
    ) -> Dataset:
        from datasets.fingerprint import Hasher
        from pandas import DataFrame

        if cache is None:
            cache = cls._default_cache
            print_err(f"Defaulting to cache: {cache}")
        if not isinstance(dataframe, DataFrame) and isinstance(dataframe, Mapping):
            dataframe = DataFrame(dataframe)

        hash_name = Hasher.hash(dataframe)
        df_temp_file = os.path.join(cache, "df-load", f"{hash_name}.csv")
        df_temp_dir = os.path.dirname(df_temp_file)
        if not os.path.exists(df_temp_dir):
            os.makedirs(df_temp_dir)

        if not os.path.exists(df_temp_file):
            print_err(f"Caching dataframe at {df_temp_file}")
            dataframe.to_csv(df_temp_file, index=False)

        print_err(f"Reloading dataframe from {df_temp_file}")
        return cls._load_from_csv(
            df_temp_file, 
            cache=cache,
        )

    @staticmethod
    def _resolve_featurizers(
        features: FeatureLike
    ):
        transformer_prefix = "transformer://"
        if isinstance(features, (str, Mapping)):
            features = [features]
        resolved_featurizers = []
        for featurizer in features:
            if isinstance(featurizer, str):
                if featurizer.endswith(".json"):
                    featurizer = Preprocessor.from_file(featurizer)
                elif featurizer.startswith(transformer_prefix):
                    ref = featurizer.split(transformer_prefix)[-1]
                    try:
                        ref, col = ref.split(":")
                    except ValueError:
                        raise ValueError(
                            f"""
                            Transformers models should be provided in the format 
                            {transformer_prefix}<ref>:<input-column>[~agg1,agg2]

                            But got: "{featurizer}"
                            """
                        )

                    try:
                        col, aggs = col.split("~")
                    except ValueError:
                        col, aggs = ref, ["mean"]
                    else:
                        aggs = aggs.split(",")

                    featurizer = Preprocessor(
                        name="hf-bart", 
                        input_column=col,
                        kwargs={
                            "ref": ref,
                            "aggregator": aggs,
                        }
                    )
                elif ":" in featurizer and featurizer.split(":")[-1] in Preprocessor.show():
                    try:
                        name, col = featurizer.split(":")
                    except ValueError:
                        raise ValueError(
                            f"""
                            Column mapped to featurizer name should be in the format
                            <column-name>:<featurizer-name>, with only one colon
                            character.

                            But got "{featurizer}"
                            """
                        )
                    else:
                        featurizer = Preprocessor(name=name, input_column=col)
                else:
                    featurizer = Preprocessor(name="identity", input_column=featurizer)
            elif isinstance(featurizer, Mapping):
                featurizer = Preprocessor.from_dict(featurizer)
            elif isinstance(featurizer, Preprocessor):
                pass
            else:
                raise ValueError(
                    f"""
                    Featurizer must be a column name, HF transformers reference,
                    JSON filename, a dict, or `Preprocessor`, but it was a 
                    {type(featurizer)}: {featurizer}
                    """
                )
            resolved_featurizers.append(featurizer)
        return resolved_featurizers

    @staticmethod
    def _resolve_hf_hub_dataset(
        ref: str, 
        cache: str,
    ) -> Dataset:
        from datasets import concatenate_datasets, load_dataset, DatasetDict

        hf_ref_full = ref.split("hf://")[-1]
        hf_ref = hf_ref_full.split("@")[0] if "@" in ref else hf_ref_full
        if ":" in hf_ref_full:
            ds_config, ds_split = hf_ref_full.split("@")[-1].split(":")[:2]
        else:
            ds_config, ds_split = hf_ref_full.split("@")[-1], None
        
        dataset = load_dataset(path=hf_ref, name=ds_config, split=ds_split, cache_dir=cache)
        if isinstance(dataset, DatasetDict):
            dataset = concatenate_datasets([v for key, v in dataset.items()])
        return dataset

    @classmethod
    def _resolve_data(
        cls, 
        data: DataLike, 
        cache: Optional[str] = None
    ) -> Union[Dataset, IterableDataset]:
        from datasets import Dataset, IterableDataset
        from pandas import DataFrame

        if isinstance(data, (Dataset, IterableDataset)):
            dataset = data
        elif isinstance(data, (DataFrame, Mapping)):
            dataset = cls._load_from_dataframe(
                data, 
                cache=cache,
            )
        elif isinstance(data, str):
            if data.startswith("hf://"):
                dataset = cls._resolve_hf_hub_dataset(
                    data,
                    cache=cache,
                )
            else:
                dataset = cls._load_from_csv(
                    data, 
                    cache=cache,
                )
        else:
            raise ValueError(
                """
                Data must be a string, Dataset, dictionary, or Pandas DataFrame.
                """
            )
        return dataset

    def _ingest_data(
        self, 
        data: DataLike,
        features: Optional[FeatureLike] = None, 
        labels: Optional[StrOrIterableOfStr] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        cache: Optional[str] = None,
        one_column_input: Optional[str] = None,
        _extra_cols_to_keep: Optional[StrOrIterableOfStr] = None,
        **preprocessing_args
    ) -> Tuple[
        List[str], 
        List[str], 
        Dict[str, Dict[str, Union[str, Preprocessor]]], 
        Dataset, 
        Dataset
    ]:

        """Process data to be consistent with training data.
        
        """
        if cache is None:
            cache = self._default_cache
        if features is None:
            features = self._input_featurizers
        if labels is None:
            labels = self._label_cols
        if features is None:
            raise AttributeError(
                """
                You cannot process new data before loading the training data.
                Try running .load_training_data() first.
                """
            )
        elif not isinstance(features, (Mapping, Iterable, str)):
            raise ValueError(
                """
                Features must be a dict, str, or list of str.
                """
            )
        dataset = self._resolve_data(data)
        featurizers = self._resolve_featurizers(features)
        featurizers_dicts = [f.to_dict() for f in featurizers]
        input_columns = sorted(set([
            featurizer.input_column for featurizer in featurizers
        ]))
        if _extra_cols_to_keep is not None:
            input_columns += cast(_extra_cols_to_keep, to=list)
        if len(input_columns) == 0:
            raise AttributeError("No input columns generated for model.")
        labels = cast(labels, to=list)
        preprocessing_args = {
            key: val or self._default_preprocessing_args.get(key)
            for key, val in preprocessing_args.items()
        }
        
        input_dataset = (
            dataset
            .map(
                self.preprocess_data,
                fn_kwargs=preprocessing_args,
                batched=True,
                batch_size=batch_size,
                desc="Preprocessing",
            )
        )
        self._check_column_presence(input_columns, labels, input_dataset)
        input_dataset = (
            input_dataset
            .select_columns(input_columns + labels)
            .map(
                self._featurize,
                fn_kwargs={"featurizers": featurizers_dicts},
                batched=True,
                batch_size=batch_size,
                desc="Featurizing",
            )
        )
        if one_column_input is not None:
            concat_label = one_column_input
        else:
            concat_label = [f.output_column for f in featurizers]
        processed_dataset = (
            input_dataset
            .with_format(None)  # guard against tensors
            .map(
                self._concat_features,
                fn_kwargs={
                    "inputs": concat_label, 
                    "labels": labels,
                    "_in_key": self._in_key,
                    "_out_key": self._out_key,
                },
                batched=True,
                batch_size=batch_size,
                desc="Collating features and labels",
            )
            .select_columns(
                input_columns 
                + labels
                + [self._in_key, self._out_key]
            )
        )

        if self._format_kwargs is None:
            self._format_kwargs = {}

        return (
            input_columns,
            labels,
            featurizers_dicts,
            input_dataset, 
            processed_dataset.with_format(
                self._format, # do not put before concat_features - breaks Arrow with chemprop due to multidim arrays
                **self._format_kwargs,
            )
        )

    def load_training_data(
        self,
        features: FeatureLike, 
        labels: Union[StrOrIterableOfStr, ArrayLike],
        data: DataLike,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        cache: Optional[str] = None,
        **preprocessing_args
    ) -> None:

        """Load dataset used for training.
        
        """

        self._input_cols = cast(features, to=list)
        self._label_cols = cast(labels, to=list)
        (
            self._input_cols, 
            self._label_cols, 
            self._input_featurizers, 
            self._input_training_data, 
            self.training_data,
        ) = self._ingest_data(
            features=features, 
            labels=labels,
            data=data,
            batch_size=batch_size,
            cache=cache,
            **preprocessing_args,
        )
        self.training_example = self.training_data.take(1).with_format('numpy')
        self.input_shape = self.training_example[self._in_key].shape[1:]
        self.output_shape = self.training_example[self._out_key].shape[1:]
        return None

    @staticmethod
    def preprocess_data(data: Mapping[str, ArrayLike]) -> Dict[str, np.ndarray]:
        return data

    @staticmethod
    @abstractmethod
    def make_dataloader(dataset: Iterable, batch_size: int, shuffle: bool):
        pass


class ChemMixinBase(DataMixinBase):

    smiles_column = "clean_smiles"
    common_fp_column = "tanimoto_nn_fp"
    tanimoto_column = "tanimoto_nn"
    
    @staticmethod
    def _featurizer_constructor(
        smiles_column: str,
        use_fp: bool = True,
        use_2d: bool = True,
        extra_featurizers: Optional[FeatureLike] = None,
        _allow_no_features: bool = False
    ) -> Iterable[Preprocessor]:
        featurizer = []
        if all([
            not use_fp, 
            not use_2d,
            extra_featurizers is None or len(extra_featurizers) == 0,
            not _allow_no_features,
        ]):
            print_err("No featurizers defined for fingerprint. Setting `use_fp=True`.")
            use_fp = True
        if use_fp:
            featurizer.append(Preprocessor(
                name="morgan-fingerprint",
                input_column=smiles_column,
            ))
        if use_2d:
            featurizer.append(Preprocessor(
                name="descriptors-2d",
                input_column=smiles_column,
            ))
        if extra_featurizers is not None:
            featurizer += extra_featurizers
        if len(featurizer) == 0 and not _allow_no_features:
            # Should never happen
            raise ValueError("No features defined for fingerprint.")
        else:
            return featurizer

    @staticmethod
    def preprocess_data(
        data: Mapping[str, ArrayLike],
        structure_column: str,
        smiles_column: str,
        input_representation: str = "smiles"
    ) -> Dict[str, np.ndarray]:
        """Generate clean, canonical SMILES.

        Examples
        ========
        >>> d  = {"struc": ["CCC", "CCO"]}
        >>> out = ChemMixinBase.preprocess_data(d, "struc", ChemMixinBase.smiles_column)
        >>> out[ChemMixinBase.smiles_column]
        ['CCC', 'CCO']
        >>> d1 = {"struc": ["CCC"]}  # singleton batch
        >>> out1 = ChemMixinBase.preprocess_data(d1, "struc", ChemMixinBase.smiles_column)
        >>> out1[ChemMixinBase.smiles_column]  # returns list
        ['CCC']
        >>> d2 = {"struc": "CCC"}  # non-batched map
        >>> out2 = ChemMixinBase.preprocess_data(d2, "struc", ChemMixinBase.smiles_column)
        >>> out2[ChemMixinBase.smiles_column]  # still returns list
        ['CCC']
        
        """
        converted = convert_string_representation(
            strings=data[structure_column],
            input_representation=input_representation,
            output_representation="smiles",
        )
        if len(data[structure_column]) > 1 and not isinstance(data[structure_column], str):
            data[smiles_column] = list(converted)
        else:
            data[smiles_column] = [converted]
        return data

    @staticmethod
    def _get_max_sim(
        query: ArrayLike, 
        references: ArrayLike,
        aggregator: Callable[[ArrayLike], float] = np.max
    ):
        a_n_b = np.sum(query[np.newaxis] * references, axis=-1, keepdims=True)
        sum_q = np.sum(query)
        similarities = a_n_b / (sum_q + np.sum(references, axis=-1) - np.sum(a_n_b, axis=-1))[..., np.newaxis]
        return aggregator(similarities)

    @staticmethod
    def _get_nn_tanimoto(
        x: Mapping[str, ArrayLike],
        refs_data: Mapping[str, ArrayLike],
        results_column: str,
        _in_key: str,
        _sim_fn: Callable[[ArrayLike, ArrayLike], float],
    ) -> Dict[str, np.ndarray]:
        query_fps = x[_in_key]
        refs = refs_data[_in_key]
        results = [_sim_fn(q, refs) for q in query_fps]
        results = np.stack(results, axis=0)
        x[results_column] = results
        return x

    def tanimoto_nn(
        self, 
        data: DataLike,
        query_structure_column: Optional[str] = None,
        query_input_representation: Optional[str] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        cache: Optional[str] = None,
        **kwargs
    ) -> Dataset:
        """Get Tanimoto similarity of nearest training set data.
    
        """
        if query_structure_column is None:
            query_structure_column = self._default_preprocessing_args["structure_column"]
        if query_input_representation is None:
            query_input_representation = self._default_preprocessing_args["input_representation"]
        fp_preprocessor = Preprocessor(
            name="morgan-fingerprint",
            input_column=self.__class__.smiles_column,
            **kwargs
        )
        query_dataset = self._resolve_data(data, cache=cache)
        common_map_opts = {
            "batched": True,
            "batch_size": batch_size
        }
        fp_map_opts = {
            "function": self._featurize,
            "fn_kwargs": {
                "featurizers": [fp_preprocessor]
            }
        }
        query_fp_col = fp_preprocessor.output_column
        queries = (
            query_dataset
            .map(
                self.preprocess_data,
                fn_kwargs={
                    "smiles_column": self.__class__.smiles_column,
                    "structure_column": query_structure_column,
                    "input_representation": query_input_representation,
                },
                **common_map_opts,
                desc="Converting to clean SMILES",
            )
            .map(
                **fp_map_opts,
                **common_map_opts,
                desc="Calculating query fingerprints",
            )
            .rename_column(query_fp_col, self.common_fp_column)
            .with_format(
                self._format, 
                **self._format_kwargs,
            )
        )

        query_fp_col, query_is_calculated = self._check_is_calculated(
            queries,
            fp_preprocessor,
        )

        ref_fp_col, ref_is_calculated = self._check_is_calculated(
            self._input_training_data,
            fp_preprocessor,
        )
        if not ref_is_calculated:
            refs = (
                self._input_training_data
                .map(
                    **fp_map_opts,
                    **common_map_opts,
                    desc="Calculating reference fingerprints",
                )
            )
        else:
            refs = self._input_training_data
        refs = (
            refs
            .rename_column(ref_fp_col, self.common_fp_column)
            .select_columns([self.smiles_column, self.common_fp_column])
            .with_format(
                self._format, 
                **self._format_kwargs,
            )
        )
        return queries.map(
            self._get_nn_tanimoto,
            fn_kwargs={
                "refs_data": refs,
                "results_column": self.tanimoto_column,
                "_in_key": self.common_fp_column, 
                "_sim_fn": self._get_max_sim,
            },
            **common_map_opts,
            desc="Calculating Tanimoto similarity to nearest training neighbor",
        ).remove_columns(self.common_fp_column)
