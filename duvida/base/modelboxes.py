"""Base containers for models and their training data."""

from typing import Any, Callable, Dict, Iterable, Mapping, Tuple, Optional, Union
from abc import abstractmethod, ABC
import os

from carabiner import print_err
from datasets import Dataset
from numpy import ndarray
from pandas import DataFrame

from .aggregators import get_aggregator, AggFunction
from .data import DataMixinBase, ChemMixinBase
from .evaluation import rmse, pearson_r, spearman_r
from .information import DoubtMixinBase
from .preprocessing import Preprocessor
from .training import ModelTrainerBase
from .typing import DataLike, FeatureLike, StrOrIterableOfStr
from ..checkpoint_utils import save_json, _load_json


class ModelBoxBase(DataMixinBase, DoubtMixinBase, ABC):

    """Container class for models and their training datasets.
    
    """
    
    _prediction_key = "prediction"
    class_name = "base"
    _init_kwargs_filename = "modelbox-init-config.json"
    _special_args_filename = "modelbox-special-args.json"
    _model_config_filename = "model-config.json"

    def __init__(self, **kwargs):
        self.model = None
        self._trainer = None
        self._init_kwargs = kwargs
        self._model_config = kwargs
        self._special_args = None

    def save_checkpoint(
        self,
        checkpoint: str
    ) -> None:
        print_err(f"Saving checkpoint at {checkpoint}")
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.save_data_checkpoint(checkpoint)
        init_kwargs = {
            "class_name": self.class_name,
        }
        init_kwargs.update(self._init_kwargs)
        save_json(init_kwargs, os.path.join(checkpoint, self.__class__._init_kwargs_filename))
        save_json(self._model_config, os.path.join(checkpoint, self.__class__._model_config_filename))
        save_json(self._special_args, os.path.join(checkpoint, self.__class__._special_args_filename))
        self.save_weights(checkpoint)
        return None

    def load_checkpoint(
        self,
        checkpoint: str,
        cache_dir: Optional[str] = None
    ) -> None:
        print_err(f"Loading checkpoint from {checkpoint}")
        self.load_data_checkpoint(checkpoint, cache_dir=cache_dir)
        self._model_config = _load_json(checkpoint, self.__class__._model_config_filename)
        self._special_args = _load_json(checkpoint, self.__class__._special_args_filename)
        if self.training_data is not None:
            self.model = self.create_model()
            self.load_weights(checkpoint, cache_dir=cache_dir)
        return None

    @abstractmethod
    def save_weights(self, checkpoint: str):
        pass

    @abstractmethod
    def load_weights(self, checkpoint: str):
        pass

    def _prepare_data(
        self, 
        data: Optional[DataLike] = None,
        features: Optional[FeatureLike] = None,
        labels: Optional[StrOrIterableOfStr] = None,
        batch_size: int = 16,
        dataloader: bool = False,
        shuffle: bool = False,
        cache: Optional[str] = None,
        dataloader_kwargs: Optional[Mapping] = None,
        **preprocessing_args
    ) -> Union[Dataset, Iterable[Mapping[str, Any]]]:  
        if data is None:
            dataset = self._check_training_data()  # from DataMixinBase
        else:
            *_, dataset = self._ingest_data(
                features=features,
                labels=labels,
                data=data,
                batch_size=batch_size,
                cache=cache,
                **preprocessing_args,
            )
        essential_keys = set([self._in_key, self._out_key])
        missing_keys = sorted(essential_keys - set(dataset.column_names))
        if len(missing_keys) > 0:
            raise KeyError(f"Dataset is missing essential columns: {', '.join(missing_keys)}.")
        if dataloader_kwargs is None:
            dataloader_kwargs = {}
        if dataloader:
            return self.make_dataloader(
                dataset=dataset, 
                batch_size=batch_size,
                shuffle=shuffle,
                **dataloader_kwargs,
            )
        else:
            return dataset

    @abstractmethod
    def create_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def create_trainer(
        self, 
        callbacks: Optional[Iterable[Callable]] = None,
        epochs: int = 1, 
        *args, **kwargs,
    ) -> ModelTrainerBase:
        pass
    
    def train(
        self, 
        training_data: Optional[DataLike] = None,
        val_data: Optional[DataLike] = None,
        features: Optional[FeatureLike] = None,
        labels: Optional[StrOrIterableOfStr] = None,
        val_features: Optional[FeatureLike] = None,
        val_labels: Optional[StrOrIterableOfStr] = None,
        epochs: int = 1, 
        batch_size: int = 16,
        # learning_rate: float = .01,
        callbacks: Optional[Iterable[Callable]] = None,
        trainer_opts: Mapping[str, Union[str, int, bool]] = None,
        reinitialize: bool = False,
        **preprocessing_args
    ) -> None:
        
        """Train model with training and validation data.
    
        """
        preparation_kwargs = {
            "batch_size": batch_size,
            "dataloader": True,
        } | preprocessing_args
        training_data = self._prepare_data(
            features=features,
            labels=labels,
            data=training_data,
            shuffle=True,
            **preparation_kwargs,
        )
        if val_data is None:
            raise ValueError("No validation data provided!")
        if val_features is None:
            val_features = features
        if val_labels is None:
            val_labels = labels
        val_data = self._prepare_data(
            features=val_features,
            labels=val_labels,
            data=val_data,
            **preparation_kwargs,
        )

        if trainer_opts is None:
            trainer_opts = {}
        if self.model is None or reinitialize:
            self.model = self.create_model()
        if self._trainer is None or reinitialize:
            self._trainer = self.create_trainer(
                epochs=epochs,  # number of epochs to train for
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

    @staticmethod
    def _predict(
        x: Mapping[str, Any],
        model: Callable,
        _in_key: str,
        _prediction_column: str,
        detacher_fn: Callable,
        aggregator: Optional[Callable] = None
    ) -> Dict[str, Any]:
        prediction = model(x[_in_key])
        x[_prediction_column] = detacher_fn(prediction)
        return x

    @staticmethod
    def _aggregate(
        x: Mapping[str, Any],
        _prediction_column: str,
        detacher_fn: Callable,
        aggregator: Optional[Callable] = None
    ) -> Dict[str, Any]:
        x[_prediction_column] = aggregator(detacher_fn(x[_prediction_column]))
        return x

    def predict(
        self, 
        data: Optional[DataLike] = None,
        features: Optional[FeatureLike] = None,
        labels: Optional[StrOrIterableOfStr] = None,
        batch_size: int = 16,
        aggregator: Optional[Union[str, AggFunction]] = None,
        cache: Optional[str] = None,
        agg_kwargs: Optional[Mapping] = None,
        _prediction_column: Optional[str] = None,
        **kwargs
    ) -> Dataset:

        """Make predictions on new data.
    
        """
        
        _prediction_column = (
            _prediction_column 
            or self.__class__._prediction_key
        )
        data = self._prepare_data(
            features=features,
            labels=labels,
            data=data,
            batch_size=batch_size,
            cache=cache,
            **kwargs
        )
            
        self.eval_mode()
        predictions = data.map(
            self._predict,
            fn_kwargs={
                "model": self.model,
                "detacher_fn": self.detach_tensor,
                "_in_key": self.__class__._in_key,
                "_prediction_column": _prediction_column,
            },
            batched=True, 
            batch_size=batch_size, 
            desc="Predicting",
        )
        if aggregator is not None:
            if agg_kwargs is None:
                agg_kwargs = {}
            aggregator_fn = get_aggregator(
                aggregator, 
                **agg_kwargs,
            )
            predictions = predictions.map(
                self._aggregate,
                fn_kwargs={
                    "_prediction_column": _prediction_column,
                    "aggregator": aggregator_fn,
                    "detacher_fn": self.detach_tensor,
                },
                batched=True, 
                batch_size=batch_size, 
                desc=f"Aggregating predictions by {aggregator}",
            )
        return predictions

    def evaluate(
        self, 
        data: Optional[DataLike] = None,
        features: Optional[FeatureLike] = None,
        labels: Optional[StrOrIterableOfStr] = None,
        metrics: Optional[Union[Callable, Iterable[Callable], Mapping[str, Callable]]] = None,
        batch_size: int = 16,
        aggregator: Optional[Union[str, AggFunction]] = None,
        agg_kwargs: Optional[Mapping] = None,
        cache: Optional[str] = None,
        **kwargs
    ) -> Tuple[DataFrame, Union[Tuple[float], Dict[str, float]]]:

        """Calculate metrics on training data or new data.
    
        """
        eval_prediction_col = self.__class__._prediction_key
        predictions = self.predict(
            features=features,
            labels=labels, 
            data=data, 
            batch_size=batch_size,
            aggregator=aggregator,
            agg_kwargs=agg_kwargs,
            cache=cache,
            _prediction_column=eval_prediction_col,
            **kwargs,
        )

        if metrics is None:
            metrics = {
                "rmse": rmse, 
                "pearson_r": pearson_r, 
                "spearman_rho": spearman_r,
            }
        predictions = predictions.with_format("numpy")
        y_vals = predictions[self._out_key]
        preds = predictions[eval_prediction_col]
        # if len(y_vals.shape) == 1 and len(preds.shape) == 2:
        #     y_vals = y_vals[...,None]
        print(y_vals.shape, preds.shape)
        if isinstance(metrics, Mapping):
            metrics = {
                name: metric(preds, y_vals).tolist()
                for name, metric in dict(metrics).items()
            }
        elif isinstance(metrics, (Iterable, Callable)):
            if isinstance(metrics, Callable):
                metrics = [metrics]
            metrics = tuple(
                metric(preds, y_vals).tolist()
                for metric in metrics
            )
        
        predictions = {
            key: predictions[key].flatten().tolist()
            if len(predictions[key].shape) > 1 and predictions[key].shape[-1] == 1 
            else predictions[key].tolist()
            for key in predictions.column_names
        }
        predictions = {
            key: val
            for key, val in predictions.items()
            if key != self._in_key and not key.startswith("__f__")
        }
        return DataFrame(predictions), metrics


class ModelBoxWithVarianceBase(ModelBoxBase):

    _variance_key: str = "prediction variance"
    
    def prediction_variance(
        self, 
        candidates: Optional[DataLike] = None,
        batch_size: int = 16,
        cache: Optional[str] = None,
        **kwargs
    ) -> ndarray:

        """Make predictions on new data.
    
        """
        predictions = self.predict(
            data=candidates, 
            aggregator="var", 
            agg_kwargs={"keepdims": False},
            batch_size=batch_size,
            cache=cache,
            _prediction_column=self.__class__._variance_key,
            **kwargs,
        )
        return predictions


class FingerprintModelBoxBase(ChemMixinBase, ModelBoxWithVarianceBase):

    def __init__(
        self, 
        use_fp: bool = False,
        use_2d: bool = False,
        extra_featurizers: Optional[FeatureLike] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.use_fp = use_fp
        self.use_2d = use_2d
        if extra_featurizers is not None:
            if isinstance(extra_featurizers, (str, Mapping)):
                extra_featurizers = [extra_featurizers]
        self.extra_featurizers = extra_featurizers
        self._default_preprocessing_args = {
            "structure_column": None,
            "input_representation": None,
        }
        self._model_config = kwargs

    def load_training_data(
        self,
        structure_column,
        input_representation: str = "smiles",
        features: Optional[FeatureLike] = None,
        _run_featurizer_constructor_first: bool = True,
        **kwargs
    ) -> None:
        self._default_preprocessing_args["structure_column"] = structure_column
        self._default_preprocessing_args["input_representation"] = input_representation
        if _run_featurizer_constructor_first:
            featurizer = self._featurizer_constructor(
                smiles_column=self.smiles_column,
                use_fp=self.use_fp,
                use_2d=self.use_2d,
                extra_featurizers=self.extra_featurizers,
            )
        else:
            featurizer = []
        if features is not None:
            featurizer += self._resolve_featurizers(features)
        return super().load_training_data(
            features=self._resolve_featurizers(featurizer),
            **kwargs,
            **self._default_preprocessing_args,
            smiles_column=self.smiles_column,
        )

    def train(
        self, 
        structure_column: Optional[str] = None,
        input_representation: Optional[str] = None,
        *args, **kwargs
    ) -> None:
        if structure_column is None:
            structure_column = self._default_preprocessing_args["structure_column"]
        if input_representation is None:
            input_representation = self._default_preprocessing_args["input_representation"]
        return super().train(
            *args, **kwargs, 
            structure_column=structure_column,
            input_representation=input_representation,
            smiles_column=self.smiles_column,
        )

    def predict(
        self, 
        structure_column: Optional[str] = None,
        input_representation: Optional[str] = None,
        **kwargs
    ) -> Dataset:

        """Make predictions on new data.
    
        """
        if structure_column is None:
            structure_column = self._default_preprocessing_args["structure_column"]
        if input_representation is None:
            input_representation = self._default_preprocessing_args["input_representation"]
        _extra_cols_to_keep=(
                [structure_column] + kwargs.pop("_extra_cols_to_keep", [])
            )
        return super().predict(
            **kwargs, 
            structure_column=structure_column,
            input_representation=input_representation,
            smiles_column=self.smiles_column,
            _extra_cols_to_keep=_extra_cols_to_keep,
        )

    def _info_score_entrypoint(
        self, 
        score_type: str,
        candidates,
        preprocessing_args: Optional[Mapping] = None,
        **kwargs
    ) -> Dataset:
        base_pp_args = self._default_preprocessing_args | {
            "smiles_column": self.smiles_column,
        }
        if preprocessing_args is None:
            preprocessing_args = base_pp_args
        elif isinstance(preprocessing_args, Mapping):
            preprocessing_args = base_pp_args | dict(preprocessing_args)
        else:
            raise ValueError(
                f"If provided, preprocessing_args must be a mapping, but was {type(preprocessing_args)}: {preprocessing_args}"
            )
        try:
            structure_column = preprocessing_args["structure_column"] or self._default_preprocessing_args["structure_column"]
        except KeyError:
            raise ValueError(f"You didn't supply a `structure_column`!")

        preprocessing_args["_extra_cols_to_keep"] = (
            [structure_column] + preprocessing_args.get("_extra_cols_to_keep", [])
        )
        return super()._info_score_entrypoint(
            score_type=score_type,
            candidates=candidates,
            preprocessing_args=preprocessing_args,
            **kwargs
        )


class ChempropModelBoxBase(FingerprintModelBoxBase):

    def __init__(
        self, 
        use_fp: bool = False,
        use_2d: bool = False,
        extra_featurizers: Optional[FeatureLike] = None,
        **kwargs
    ):
        super().__init__(
            use_fp=use_fp,
            use_2d=use_2d,
            extra_featurizers=extra_featurizers
        )
        self._model_config = kwargs
        self._special_args = {
            "chemprop_input_column": None,
        }

    def load_training_data(
        self,
        labels: StrOrIterableOfStr,
        features: Optional[FeatureLike] = None, 
        **kwargs
    ) -> None:
        featurizer = self._featurizer_constructor(
            smiles_column=self.smiles_column,
            use_fp=self.use_fp,
            use_2d=self.use_2d,
            extra_featurizers=self.extra_featurizers,
            _allow_no_features=True,
        )
        if features is not None:
            featurizer += features
        featurizer = self._resolve_featurizers(featurizer)
        featurizer += [{
            "name": "chemprop-mol",
            "input_column": self.smiles_column,
            "kwargs": {
                "label_column": labels,
                "extra_featurizers": [f.output_column for f in featurizer],
            },
        }]
        self._special_args["chemprop_input_column"] = Preprocessor.from_dict(featurizer[-1]).output_column
        super().load_training_data(
            labels=labels,
            features=self._resolve_featurizers(featurizer),
            _run_featurizer_constructor_first=False,
            **kwargs,
        )

    def _ingest_data(
        self, *args, **kwargs
    ):
        return super()._ingest_data(
            *args, **kwargs,
            one_column_input=self._special_args["chemprop_input_column"],
        )
