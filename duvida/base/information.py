"""Base mixins for calculating uncertainty and information metrics."""

from typing import Callable, Dict, Iterable, List, Mapping, Optional, Union
from abc import abstractmethod, ABC
from functools import partial, reduce

from carabiner import cast
from datasets import Dataset
from numpy import concatenate, ndarray
from numpy.typing import ArrayLike

from .aggregators import get_aggregator, AggFunction


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
    def _add_values(
        x: ArrayLike, 
        y: Mapping[str, ArrayLike]
    ) -> Dict[str, ndarray]:
        return {key: x[key] + _y for key, _y in y.items()}

    @staticmethod
    def _reduce_dataset(
        x: Dataset, 
        reducer: Callable,
        columns: Optional[Union[str, Iterable[str]]] = None,
        batch_size: int = 1000
    ) -> Dataset:
        if columns is not None:
            columns = [
                col for col in cast(columns, to=list) 
                if col in x.column_names
            ]
        return {col: reducer(x[col]) for col in columns}

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
    def information_sensitivity_core(
        fisher_score, 
        fisher_information_diagonal, 
        parameter_gradient, 
        parameter_hessian_diagonal
    ):
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
            dataset = self._check_training_data()
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
                self._process_information, 
                fn_kwargs={
                    "grad_fns": grad_fns,
                    "_in_key": self._in_key,
                    "_out_key": self._out_key,
                },
                batch_size=batch_size,
                batched=True,
                remove_columns=dataset.column_names,
                desc=f"Calculating Fisher score{' and information' if hessian else ''}",
            )
        )
        columns = [
            col for col in ['fisher_score', 'fisher_information_diagonal'] 
            if col in dataset.column_names
        ]
        return (
            self._reduce_dataset(dataset, self._mean_of_dictionaries, col)[col]
            for col in columns
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
        data["score"] = doubtscore
        return data

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
        infosens = aggregator(
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
        data["score"] = infosens
        return data

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
            inner_fn,
            fn_kwargs={
                "param_grad_fn": self.parameter_gradient(model=model), 
                "param_hessian_fn": param_hessian_fn,
                "fisher_score": fisher_score,
                "loss_hessian": fisher_info_diag,
            },
            batched=True, 
            batch_size=batch_size,
            desc=f"Calculating parameter gradients{' and Hessians' if use_hessian and not optimality_approximation else ''}",
        )#.select_columns(['score'])
        return score.rename_column('score', score_type)

    def _check_training_data(self):
        if hasattr(self, "training_data"):
            if getattr(self, "training_data") is None:
                raise AttributeError("Training data is not provided. Run .load_training_data() first!")
            else:
                return self.training_data
        else:
            raise ValueError("Training data is not provided!")

    def _info_score_entrypoint(
        self, 
        score_type: str,
        candidates: Union[ArrayLike, Dataset],
        features: Optional[str] = None,
        labels: Optional[str] = None,
        training_dataset: Optional[Dataset] = None,
        batch_size: int = 16,
        preprocessing_args: Optional[Mapping] = None,
        cache: Optional[str] = None,
        **info_score_kwargs
    ) -> Dataset:
        self.release_memory()
        if preprocessing_args is None:
            preprocessing_args = {}
        if training_dataset is None:  
            dataset = self._check_training_data()

        if hasattr(self, "_prepare_data"):
            candidates = self._prepare_data(
                data=candidates,
                features=features,
                labels=labels,
                batch_size=batch_size,
                cache=cache,
                **preprocessing_args
            )
            
        return self._get_info_score(
            score_type=score_type,
            candidates=candidates,
            dataset=dataset, 
            batch_size=batch_size,
            **info_score_kwargs,
        )

    def doubtscore(
        self, 
        features: Optional[str] = None,
        preprocessing_args: Optional[Mapping] = None,
        **info_score_kwargs
    ) -> Dataset:

        """Calculate doubtscore.
        
        """

        return self._info_score_entrypoint(
            score_type="doubtscore",
            features=features,
            preprocessing_args=preprocessing_args,
            **info_score_kwargs
        )

    def information_sensitivity(
        self, 
        features: Optional[str] = None,
        preprocessing_args: Optional[Mapping] = None,
        **info_score_kwargs
    ) -> Dataset:

        """Calculate information sensitivity.
        
        """

        return self._info_score_entrypoint(
            score_type="information sensitivity",
            features=features,
            preprocessing_args=preprocessing_args,
            **info_score_kwargs
        )
