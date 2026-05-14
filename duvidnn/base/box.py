"""Base ModelBox implementation."""
from typing import Iterable
from dataclasses import dataclass, field

@dataclass
class ModelBox:
    data_pipeline: DataPipeline
    adapter_spec: AdapterSpec
    model_spec: ModelSpec
    output_spec: OutTransformSpec = field(default=None)
    information_spec: InformationSpec = field(default=None)
    services: Iterable = field(default_factory=list)
    trainer = field(default=None)

    def __post_init__(self):
        self.adapter = self.model_adapter()
        self.model = self.model_spec()
        self.output_transform = self.output_spec()

    def train(
        self, 
        training_data: Optional[DataLike] = None,
        val_data: Optional[DataLike] = None,
        features: Optional[FeatureLike] = None,
        context: Optional[FeatureLike] = None,
        labels: Optional[StrOrIterableOfStr] = None,
        val_features: Optional[FeatureLike] = None,
        val_context: Optional[FeatureLike] = None,
        val_labels: Optional[StrOrIterableOfStr] = None,
        epochs: int = 1, 
        batch_size: int = _DEFAULT_BATCH_SIZE,
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
        training_data = self.data_pipeline(training_data)
        if val_data is None:
            raise ValueError("No validation data provided!")
        val_data = self.data_pipeline(val_data)

        if trainer_opts is None:
            trainer_opts = {}
        if self.model.model is None or reinitialize:
            self.model.create_model()
        self.model = self.trainer.train(
            model=self.model.model, 
            train_dataloader=self.data_adapter.dataloader, 
            val_dataloader=val_data,
            epochs=epochs,  # number of epochs to train for
            callbacks=callbacks,
            **trainer_opts,
        )
        return None

    def predict(
        self, 
        data: Optional[DataLike] = None,
        features: Optional[FeatureLike] = None,
        context: Optional[FeatureLike] = None,
        labels: Optional[StrOrIterableOfStr] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
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
            context=context,
            labels=labels,
            data=data,
            batch_size=batch_size,
            cache=cache,
            **kwargs
        )
            
        self.eval_mode()
        _in_key = tuple(sorted(
            col for col in data.column_names 
            if col.startswith(self.__class__._in_key)
        ))
        predictions = data.map(
            self._predict,
            fn_kwargs={
                "model": self.model,
                "detacher_fn": self.detach_tensor,
                "_in_key": _in_key,
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
        context: Optional[FeatureLike] = None,
        labels: Optional[StrOrIterableOfStr] = None,
        metrics: Optional[Union[Callable, Iterable[Callable], Mapping[str, Callable]]] = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
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
            context=context,
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
                name: asarray(metric(preds, y_vals)).tolist()
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