"""Base abstract classes for duvida models."""

from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple, Optional, Union
from abc import abstractmethod, ABC
from functools import partial, reduce
from io import TextIOWrapper
import json
import os

from carabiner import cast
from datasets import Dataset, IterableDataset, load_dataset
from datasets.fingerprint import Hasher
from numpy import asarray, concatenate, mean, newaxis, ndarray, stack
from numpy.typing import ArrayLike
from pandas import DataFrame
from torch.utils.data import DataLoader

from .aggregators import get_aggregator, AggFunction
from .checkpoint_utils import load_checkpoint_file

class DataMixinBase(ABC):

    """Add data-loading capability to models.
    
    """

    _in_key: str = 'inputs'
    _out_key: str = 'labels'
    _input_training_data = None
    _input_cols = None
    _label_cols = None
    training_data = None
    training_example = None
    input_shape = None 
    output_shape = None
    _format: str = 'numpy'
    _format_kwargs: Optional[Mapping[str, Any]] = None
    _default_cache = "~/.cache/huggingface"

    def save_data_checkpoint(
        self, 
        checkpoint_dir: str
    ):
        keys = (
            "_in_key",
            "_out_key",
            "_input_cols",
            "_label_cols",
            "input_shape",
            "output_shape",
            "_default_cache",
        )
        data_config = {
            key: getattr(self, key) for key in keys
        }
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if not self._input_training_data is None:
            self._input_training_data.save_to_disk(
                os.path.join(os.path.join(checkpoint_dir, "input-data.hf")),
            )
            if not self.training_data is None:
                (
                    self.training_data
                    .with_format("numpy", dtype="float")
                    .save_to_disk(os.path.join(checkpoint_dir, "training-data.hf")),
                )
        with open(os.path.join(checkpoint_dir, "data-config.json"), "w") as f:
            json.dump(data_config, f)
        return None

    def load_data_checkpoint(
        self, 
        checkpoint: str
    ):
        data_config = load_checkpoint_file(
            checkpoint, 
            filename="data-config.json",
            callback="json",
            none_on_error=False,
        )
        for key, val in data_config.items():
            setattr(self, key, val)
        self._input_training_data = load_checkpoint_file(
            checkpoint, 
            filename="input-data.hf",
            callback="hf-dataset",
            none_on_error=True,
        )
        training_data = load_checkpoint_file(
            checkpoint, 
            filename="training-data.hf",
            callback="hf-dataset",
            none_on_error=True,
        )
        if not training_data is None:
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
        x: Mapping[str, List],
        features: Union[str, Iterable[str]], 
        labels: Union[str, Iterable[str]],
        _in_key: str = "inputs",
        _out_key: str = "labels"
    ) -> Mapping[str, ndarray]:
        coltypes = {
            _in_key: cast(features, to=list), 
            _out_key: cast(labels, to=list),
        }
        x = {
            coltype: [asarray(x[col]) for col in columns]
            for coltype, columns in coltypes.items()
        }
        x = {
            coltype: concatenate([col if col.ndim > 1 else col[..., newaxis] for col in columns], axis=-1)
            for coltype, columns in x.items()
        }
        return x

    @staticmethod
    def _check_data_types(features, labels, data, datatype) -> None:
        if not isinstance(features, (str, Iterable)) or not isinstance(labels, (str, Iterable)):
            raise ValueError("Dataset provided to .load_training_data() but "
                             "features and labels are not strings for "
                             "columns indexing.")
        if not isinstance(data, datatype):
            raise ValueError(f"Input data must be {datatype}.")

        return None

    @staticmethod
    def _check_column_presence(
        features: Union[str, Iterable[str]], 
        labels: Union[str, Iterable[str]],
        data: Union[Dataset, DataFrame]
    ) -> Iterable[str]:
        columns = cast(features, to=list) + cast(labels, to=list)
        if isinstance(data, Dataset):
            data_cols = data.column_names
        else:
            data_cols = data
        absent_columns = [col for col in columns if not col in data_cols]
        if len(absent_columns) > 0:
            raise ValueError(f"Requested columns ({', '.join(columns)}) not present in {type(data)}: {', '.join(absent_columns)}.")
        return columns

    def _load_from_dataset(
        self,
        dataset: Dataset,
        features: Union[str, Iterable[str]], 
        labels: Union[str, Iterable[str]],
    ) -> Dataset:
        self._check_data_types(features, labels, dataset, (Dataset, IterableDataset))
        columns = self._check_column_presence(features, labels, dataset)
        return dataset.select_columns(columns)
    
    def _load_from_dataframe(
        self,
        dataframe: Union[DataFrame, Mapping[str, ArrayLike]],
        features: Union[str, Iterable[str]], 
        labels: Union[str, Iterable[str]],
        cache: Optional[str] = None
    ) -> Dataset:
        if cache is None:
            cache = self._default_cache
        random_name = Hasher.hash(dataframe)
        df_temp_file = os.path.join(self._default_cache, "duvida", f"{random_name}.csv")
        df_temp_dir = os.path.dirname(df_temp_file)
        if not os.path.exists(df_temp_dir):
            os.mkdir(df_temp_dir)
        
        self._check_data_types(features, labels, dataframe, (DataFrame, Mapping))
        columns = self._check_column_presence(features, labels, dataframe)
        if isinstance(dataframe, DataFrame):
            dataframe.to_csv(df_temp_file, index=False)
        elif isinstance(dataframe, Mapping):
            dataframe = DataFrame({
                col: dataframe[col] for col in columns
            })
            dataframe.to_csv(df_temp_file, index=False)
        dataset = load_dataset(df_temp_file, cache_dir=cache)
        return dataset

    def _ingest_data(
        self, 
        features: Optional[Union[str, Iterable[str], ArrayLike]] = None, 
        labels: Optional[Union[str, Iterable[str], ArrayLike]] = None,
        filename:  Optional[str] = None,
        data: Optional[Union[DataFrame, Mapping[str, ArrayLike], Dataset, IterableDataset]] = None,
        batch_size: int = 128,
        cache: Optional[str] = None,
        ds_config: Optional[str] = None,
        ds_split: Optional[str] = None,
        **pp_args
    ) -> Tuple[Dataset, Dataset]:

        """Process data to be consistent with training data.
        
        """
        if cache is None:
            cache = self._default_cache
        if features is None:
            features = self._input_cols
        if labels is None:
            labels = self._label_cols
        
        if features is None:
            raise AttributeError(
                "You cannot process new data before loading the training data."
                "Try running .load_training_data() first."
            )

        if data is not None:
            if isinstance(data, (Dataset, IterableDataset)):
                dataset = self._load_from_dataset(data, features, labels)
            elif isinstance(data, (DataFrame, Mapping)):
                dataset = self._load_from_dataframe(data, features, labels, cache)
            else:
                raise ValueError(
                    """
                    If provided, data must be a Dataset object, a dictionary, or a Pandas DataFrame.
                    """
                )
        elif isinstance(filename, str):
            if filename.startswith("hf://"):
                hf_ref_full = filename.split("hf://")[-1]
                hf_ref = hf_ref_full.split("@")[0] if "@" in filename else hf_ref_full
                if ":" in hf_ref_full:
                    ds_config, ds_split = hf_ref_full.split("@")[-1].split(":")[:2]
                else:
                    ds_config, ds_split = None, None
                dataset = self._load_from_dataset(
                    load_dataset(hf_ref, ds_config, split=ds_split, cache_dir=cache), 
                    features, 
                    labels,
                )
            else:
                sep = "," if filename.endswith(".csv") else "\t"
                dataset = self._load_from_dataset(
                    Dataset.from_csv(filename, cache_dir=cache, sep=sep), 
                    features, 
                    labels,
                )
        elif isinstance(features, ArrayLike) and isinstance(labels, ArrayLike):
            df = DataFrame()
            dataset = self._load_from_dataframe({
                    self._in_key: features, 
                    self._out_key: labels,
                }, 
                self._in_key, 
                self._out_key, 
                cache,
            )
        else:
            raise ValueError(
                """
                Inputs can be either:
                - dataframe: pd.DataFrame or dict, with features and labels str or iterable of str
                - features and labels both lists or numpy arrays
                - filename is a str
                """
            )
        
        input_dataset = (
            dataset
            .map(
                self._concat_features,
                fn_kwargs={
                    "features": features, 
                    "labels": labels,
                    "_in_key": self._in_key,
                    "_out_key": self._out_key,
                },
                batched=True,
                batch_size=batch_size,
                desc="Collating features and labels",
            )
            .select_columns(
                cast(features, to=list)
                + [self._in_key, self._out_key]
            )
        )
        processed_dataset = input_dataset.map(
            partial(
                self.preprocess_data, 
                _in_key=self._in_key, 
                _out_key=self._out_key, 
                **pp_args,
            ),
            batched=True,
            batch_size=batch_size,
            desc="Preprocessing",
        )

        if self._format_kwargs is None:
            self._format_kwargs = {}

        return (
            input_dataset, 
            processed_dataset.with_format(self._format, **self._format_kwargs)
        )

    def load_training_data(
        self,
        features: Union[str, Iterable[str], ArrayLike], 
        labels: Optional[Union[str, Iterable[str], ArrayLike]] = None,
        filename:  Optional[str] = None,
        data: Optional[Union[DataFrame, Dataset, Mapping[str, ArrayLike]]] = None,
        batch_size: int = 128,
        cache: Optional[str] = None,
        ds_config: Optional[str] = None,
        ds_split: Optional[str] = None,
        **pp_args
    ) -> None:

        """Load dataset used for training.
        
        """

        self._input_cols = cast(features, to=list)
        self._label_cols = cast(labels, to=list)
        self._input_training_data, self.training_data = self._ingest_data(
            features=self._input_cols, 
            labels=self._label_cols,
            filename=filename,
            data=data,
            batch_size=batch_size,
            cache=cache,
            ds_config=ds_config,
            ds_split=ds_split,
            **pp_args
        )
        self.training_example = self.training_data.take(1).with_format('numpy')
        self.input_shape = self.training_example[self._in_key].shape[1:]
        self.output_shape = self.training_example[self._out_key].shape[1:]
        return None

    @staticmethod
    def preprocess_data(data: Mapping[str, ArrayLike], _in_key: str, _out_key: str, **kwargs) -> Dict[str, ndarray]:
        return data

    @staticmethod
    @abstractmethod
    def make_dataloader(dataset: Iterable, batch_size: int, shuffle: bool):
        pass


class ModelTrainerBase(ABC):

    def __init__(
        self, 
        epochs: int, 
        *args, **kwargs,
    ):
        self.epochs = epochs
        self._args = args
        self._kwargs = kwargs
        self._trainer = None

    @abstractmethod
    def create_trainer(self):
        pass

    @abstractmethod
    def train(self, model, train_dataloader, val_dataloader):
        pass


class ModelBoxBase(ABC):

    """Container class for models and their training datasets.
    
    """
    
    _in_key = DataMixinBase._in_key
    _out_key = DataMixinBase._out_key
    _prediction_key = "__prediction__"

    def __init__(self):
        self.model = None
        self._trainer = None

    @abstractmethod
    def save_checkpoint(
        self,
        checkpoint_dir: str
    ):
        pass

    def _check_training_data(self):
        if isinstance(self, DataMixinBase):
            if self.training_data is None:
                raise ValueError("Training data is not provided. Run .load_training_data() first!")
            else:
                return self.training_data
        else:
            raise ValueError("Training data is not provided!")

    def _prepare_data(
        self, 
        data: Optional = None,
        filename: Optional[str] = None,
        features: Optional = None,
        labels: Optional = None,
        batch_size: int = 16,
        dataloader: bool = False,
        shuffle: bool = False,
        cache: Optional[str] = None,
        _dataloader_kwargs: Optional[Mapping] = None,
        **kwargs
    ) -> Union[Dataset, Iterable[Mapping[str, Any]]]:  
        if data is None and filename is None:
            dataset = self._check_training_data()
        elif isinstance(self, DataMixinBase):
            _, dataset = self._ingest_data(
                features=features,
                labels=labels,
                filename=filename,
                data=data,
                batch_size=batch_size,
                cache=cache,
                **kwargs
            )
        elif data is not None:
            if isinstance(data, Iterable):
                dataset = data
            else:
                raise TypeError(f"Provided data must be iterable, but was {type(data)}.")
        elif filename is not None:
            raise ValueError("Cannot process files if not subclassed from `DataMixinBase`")

        essential_keys = set([self._in_key, self._out_key])
        missing_keys = sorted(essential_keys - set(dataset.column_names))
        if len(missing_keys) > 0:
            raise KeyError(f"Dataset is missing essential columns: {', '.join(missing_keys)}.")
            
        if isinstance(self, DataMixinBase) and dataloader:
            return self.make_dataloader(
                dataset=dataset, 
                batch_size=batch_size,
                shuffle=shuffle,
            )
        elif dataloader:
            if _dataloader_kwargs is None:
                _dataloader_kwargs = {}
            return DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                **_dataloader_kwargs
            )
        else:
            return dataset

    @abstractmethod
    def create_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_trainer(
        self, 
        callbacks: Optional[Iterable] = None,
        epochs: int = 1, 
        *args, **kwargs,
    ) -> ModelTrainerBase:
        pass
    
    def train(
        self, 
        training_data: Optional = None,
        training_filename: Optional[str] = None,
        val_data: Optional = None,
        val_filename: Optional[str] = None,
        features: Optional = None,
        labels: Optional = None,
        val_features: Optional = None,
        val_labels: Optional = None,
        epochs: int = 1, 
        batch_size: int = 16,
        # learning_rate: float = .01,
        callbacks: Optional[Iterable] = None,
        trainer_opts: Mapping[str, Union[str, int, bool]] = None,
        *args, **kwargs
    ) -> None:
        
        """Train model with training and validation data.
    
        """

        training_data = self._prepare_data(
            features=features,
            labels=labels,
            filename=training_filename,
            data=training_data,
            batch_size=batch_size,
            shuffle=True,
            dataloader=True,
        )
        if val_data is None and val_filename is None:
            raise ValueError("No validation data provided!")
        if val_features is None:
            val_features = features
        if val_labels is None:
            val_labels = labels
        val_data = self._prepare_data(
            features=val_features,
            labels=val_labels,
            filename=val_filename,
            data=val_data,
            batch_size=batch_size,
            dataloader=True,
        )

        if trainer_opts is None:
            trainer_opts = {}
        if self.model is None:
            self.model = self.create_model(
                *args, **kwargs,
            )
        if self._trainer is None:
            self._trainer = self.create_trainer(
                epochs=epochs, # number of epochs to train for
                callbacks=callbacks,
                **trainer_opts,
            )
        self.model = self._trainer.train(
            model=self.model, 
            train_dataloader=training_data, 
            val_dataloader=val_data,
        )
        return None

    @abstractmethod
    def eval_mode(self):
        pass

    @staticmethod
    @abstractmethod
    def detach_tensor():
        pass

    def _predict(
        self,
        x: Mapping[str, Any]
    ) -> Dict[str, Any]:
        self.eval_mode()
        return {self._prediction_key: self.model(x[self._in_key])}

    def predict(
        self, 
        data: Optional = None,
        filename: Optional[str] = None,
        features: Optional = None,
        labels: Optional = None,
        batch_size: int = 16,
        aggregator: Optional[Union[str, AggFunction]] = None,
        cache: Optional[str] = None,
        **kwargs
    ) -> Dataset:

        """Make predictions on new data.
    
        """
        data = self._prepare_data(
            features=features,
            labels=labels,
            filename=filename,
            data=data,
            batch_size=batch_size,
            cache=cache,
        )
        if aggregator is not None:
            aggregator = get_aggregator(aggregator, **kwargs)
            def _predict(x):
                x = self._predict(x)
                x[self._prediction_key] = aggregator(
                    self.detach_tensor(x[self._prediction_key]), 
                )
                return x
        else:
            _predict = self._predict

        return data.map(
            _predict, 
            batched=True, 
            batch_size=batch_size, 
            desc="Predicting",
        )

    def evaluate(
        self, 
        data: Optional = None,
        filename: Optional[str] = None,
        features: Optional = None,
        labels: Optional = None,
        metrics: Optional[Iterable] = None,
        batch_size: int = 16,
        aggregator: Optional[Union[str, AggFunction]] = None,
        cache: Optional[str] = None,
    ) -> Tuple[ndarray, Union[Tuple[float], Dict[str, float]]]:

        """Calculate metrics on training data or new data.
    
        """

        # data_prepped = self._prepare_data(
        #     features=features,
        #     labels=labels,
        #     filename=filename,
        #     data=data,
        #     batch_size=batch_size,
        # )
        predictions = self.predict(
            features=features,
            labels=labels, 
            data=data, 
            filename=filename,
            batch_size=batch_size,
            aggregator=aggregator,
            cache=cache,
        )

        if metrics is None:
            metrics = []
        else:
            y_vals = predictions.with_format("numpy")[self._out_key]
        predictions = predictions.with_format("numpy")

        if isinstance(metrics, Mapping):
            metrics = {
                name: metric(predictions[self._prediction_key], y_vals).tolist()
                for name, metric in dict(metrics).items()
            }
        else:
            metrics = tuple(
                metric(predictions[self._prediction_key], y_vals).tolist()
                for metric in metrics
            )
        
        predictions = {
            key: predictions[key].flatten().tolist()
            if len(predictions[key].shape) > 1 and predictions[key].shape[-1] == 1 
            else predictions[key].tolist()
            for key in predictions.column_names
        }
        return DataFrame(predictions), metrics


class DoubtMixinBase(ABC):

    optimizer = None

    @staticmethod
    def _pack_params_like(
        p: ArrayLike, 
        d: Mapping[str, ArrayLike]
    ) -> Dict[str, ndarray]:
        index = 0
        output = {}
        for name, param in d.items():
            end_index = index + param.numel()
            output[name] = p[index:end_index].reshape(*param.shape)
            index = end_index
        return output

    @staticmethod
    @abstractmethod
    def _get_shape(x: ArrayLike):
        pass

    def _add_size(
        self, 
        x: ArrayLike, 
        y: Mapping[str, ArrayLike], 
        axis: int = 0
    ) -> Dict[str, ndarray]:
        return {
            key: self._get_shape(_x)[axis] + self._get_shape(y[key])[axis] 
            for key, _x in y.items()
        }

    @staticmethod
    def _add_values(x: ArrayLike, 
                    y: Mapping[str, ArrayLike]) -> Dict[str, ndarray]:
        return {key: x[key] + _y for key, _y in y.items()}

    @staticmethod
    def _reduce_dataset(
        x: Dataset, 
        reducer: Callable,
        columns: Optional[Union[str, Iterable[str]]] = None,
        batch_size: int = 1000,
    ) -> Dataset:
        if columns is not None:
            x = x.select_columns(
                [col for col in cast(columns, to=list) 
                 if col in x.column_names]
            )
        return {col: reducer(x[col]) for col in x.column_names}

    def _mean_of_dictionaries(
        self,
        x: Iterable[Mapping[str, ArrayLike]], 
        axis: int = 0
    ) -> Union[None, Dict[str, ndarray]]:
        n = len(x)
        if n > 0:
            total_size = reduce(self._add_size, x)
            total_value = reduce(self._add_values, x)
            return {key: (val / total_size[key]) for key, val in total_value.items()}
        else:
            return None

    @staticmethod
    @abstractmethod
    def release_memory() -> None:
        pass

    @abstractmethod
    def parameter_gradient(self, *args, **kwargs) -> Callable:
        pass

    @abstractmethod
    def fisher_score(self, *args, **kwargs) -> Callable:
        pass
    
    @abstractmethod
    def parameter_hessian_diagonal(self, *args, **kwargs) -> Callable:
        pass
    
    @abstractmethod
    def fisher_information_diagonal(self, *args, **kwargs) -> Callable:
        pass
    
    @staticmethod
    @abstractmethod
    def doubtscore_core(fisher_score, parameter_gradient):
        pass
    
    @staticmethod
    @abstractmethod
    def information_sensitivity_core(fisher_score, fisher_information_diagonal, parameter_gradient, parameter_hessian_diagonal):
        pass
    
    @staticmethod
    def _process_information(
        data: Mapping[str, ArrayLike],
        grad_fns: Iterable[Callable],
        _in_key: str = "inputs",
        _out_key: str = "labels"
    ) -> Dict[str, List[Dict]]:
        grads_hessians = dict(zip(
            ('fisher_score', 'fisher_information_diagonal'),
            [[]] * len(grad_fns),
        ))
        for key, f in zip(grads_hessians, grad_fns):
            f_result = f(x_true=data[_in_key], y_true=data[_out_key])
            grads_hessians[key].append(f_result)
        return grads_hessians

    def _information_scores(
        self, 
        model: Optional[Callable] = None,
        dataset: Optional[Dataset] = None,
        hessian: bool = False, 
        batch_size: int = 16,
        **kwargs
    ) -> Dataset:
        if dataset is None:  
            if isinstance(self, ModelBoxBase):
                dataset = self._check_training_data()
            else:
                raise ValueError(f"Training data not provided!")
        
        results = []
        fn = (self.fisher_score,)
        if hessian:
            fn += (partial(
                self.fisher_information_diagonal, 
                **kwargs,
            ),)
        
        grad_fns = tuple(f(model) for f in fn)
        dataset = (
            dataset
            .map(
                partial(
                    self._process_information, 
                    grad_fns=grad_fns, 
                    _in_key=self._in_key, 
                    _out_key=self._out_key,
                ),
                batch_size=batch_size,
                batched=True,
                remove_columns=dataset.column_names,
                desc=f"Calculating Fisher score{' and information' if hessian else ''}",
            )
        )
        dataset = dataset.select_columns([
            col for col in ['fisher_score', 'fisher_information_diagonal'] 
            if col in dataset.column_names
        ])
        return (
            self._reduce_dataset(dataset, self._mean_of_dictionaries, col)[col]
            for col in dataset.column_names
        )

    def _doubtscore(
        self, 
        data: Mapping[str, ArrayLike], 
        param_grad_fn: Callable, 
        param_hessian_fn: Callable,
        fisher_score: Mapping[str, ArrayLike], 
        loss_hessian: Optional[Mapping[str, ArrayLike]] = None,
        aggregator: Union[str, AggFunction] = 'rms'
    ) -> Dict[str, ndarray]:
        
        aggregator = get_aggregator(aggregator, axis=-1)
        param_gradient = param_grad_fn(data[self._in_key])
        doubtscore = aggregator(
            concatenate([
                self.doubtscore_core(fisher_score[name], param_gradient[name]) 
                for name in fisher_score
            ], axis=-1), 
        )
        return dict(score=doubtscore)

    def _information_sensitivity(
        self, 
        data: Mapping[str, ArrayLike], 
        param_grad_fn: Callable, 
        param_hessian_fn: Callable,
        fisher_score: Mapping[str, ArrayLike], 
        loss_hessian: Mapping[str, ArrayLike],
        optimality_approximation: bool = False,
        aggregator: Union[str, AggFunction] = 'rms'
    ) -> Dict[str, ndarray]:

        aggregator = get_aggregator(aggregator, axis=-1)
        if optimality_approximation:
            param_gradient, param_hessian = (
                param_grad_fn(data[self._in_key]), 
                {name: None for name in fisher_score}
            )
        else:
            param_gradient, param_hessian = (
                fn(data[self._in_key]) for fn in (param_grad_fn, param_hessian_fn)
            )
        infosens =  aggregator(
            concatenate([
                self.information_sensitivity_core(
                    fisher_score[name], 
                    loss_hessian[name], 
                    param_gradient[name], 
                    param_hessian[name],
                    optimality_approximation=optimality_approximation,
                ) for name in fisher_score
            ], axis=-1), 
        )
        return dict(score=infosens)

    def _get_info_score(
        self, 
        score_type: str,
        dataset: Dataset,
        candidates: Dataset,
        batch_size: int = 16,
        model: Optional[Callable] = None,
        optimality_approximation: bool = False,
        **kwargs,
    ) -> Dataset:

        if score_type == "information sensitivity":
            inner_fn = partial(
                self._information_sensitivity, 
                optimality_approximation=optimality_approximation,
            )
        elif score_type == "doubtscore":
            inner_fn = self._doubtscore
        else:
            raise NotImplementedError(f"Score '{score_type}' not yet implemented.")
            
        use_hessian = (score_type == "information sensitivity")
        use_param_hessian = (use_hessian and not optimality_approximation)
        fisher_score_and_info = self._information_scores(
            model=model,
            hessian=use_hessian,
            batch_size=batch_size, 
            **kwargs,
        )
        if use_hessian:
            fisher_score, fisher_info_diag = fisher_score_and_info
        else:
            fisher_score, fisher_info_diag = next(fisher_score_and_info), None

        if use_param_hessian:
            param_hessian_fn = self.parameter_hessian_diagonal(
                model=model,
                **kwargs,
            )
        else:
            param_hessian_fn = None            

        score = candidates.map(
            partial(
                inner_fn, 
                param_grad_fn=self.parameter_gradient(model=model), 
                param_hessian_fn=param_hessian_fn,
                fisher_score=fisher_score, 
                loss_hessian=fisher_info_diag,
            ),
            batched=True, 
            batch_size=batch_size,
            desc=f"Calculating parameter gradients{' and Hessians' if use_hessian and not optimality_approximation else ''}",
        ).select_columns(['score'])
        return score.rename_column('score', score_type)

    def _info_score_entrypoint(
        self, 
        score_type: str,
        candidates: Union[ArrayLike, Dataset],
        candidate_features: Optional[str] = None,
        candidate_labels: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        *args, **kwargs
    ) -> Dataset:
        self.release_memory()
        if dataset is None:  
            if isinstance(self, ModelBoxBase):
                dataset = self._check_training_data()
            else:
                raise ValueError(f"Training data not provided!")

        if isinstance(self, ModelBoxBase):
            candidates = self._prepare_data(
                features=candidate_features,
                labels=candidate_labels,
                data=candidates,
                batch_size=batch_size,
            )
            
        return self._get_info_score(
            score_type=score_type,
            candidates=candidates,
            dataset=dataset, 
            batch_size=batch_size,
            *args, **kwargs,
        )

    def doubtscore(self, *args, **kwargs) -> Dataset:

        """Calculate doubtscore.
        
        """

        return self._info_score_entrypoint(
            score_type="doubtscore",
            *args, **kwargs
        )

    def information_sensitivity(self, *args, **kwargs) -> Dataset:

        """Calculate information sensitivity.
        
        """

        return self._info_score_entrypoint(
            score_type="information sensitivity",
            *args, **kwargs
        )


class VarianceMixin:

    _prediction_key: str = ModelBoxBase._prediction_key
    _variance_key: str  = "prediction_variance"
    
    def prediction_variance(
        self, 
        candidates: Union[ArrayLike, Dataset],
        batch_size: int = 16,
        **kwargs
    ) -> ndarray:

        """Make predictions on new data.
    
        """
        if isinstance(self, ModelBoxBase):
            predictions = self.predict(
                data=candidates, 
                aggregator="var", 
                keepdims=False,
                **kwargs
            ).select_columns([self._prediction_key])
            return predictions.rename_column(self._prediction_key, self._variance_key)
        else:
            raise ValueError("VarianceMixin can only be used with ModelBox!")