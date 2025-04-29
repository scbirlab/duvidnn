"""Command-line interface for duvida."""

from typing import Mapping, Optional, Union

from argparse import FileType, Namespace
from collections import defaultdict
import os
import sys

from carabiner.utils import pprint_dict, print_err
from carabiner.cliutils import clicommand, CLIOption, CLICommand, CLIApp

from . import __version__
from .checkpoint_utils import _load_json, save_json

_data_root = os.path.join(
    os.path.dirname(__file__), 
    "data",
)
modelbox_name_file = os.path.join(_data_root, "modelbox-names.json")
if not os.path.exists(modelbox_name_file):
    from .autoclass import AutoModelBox
    from .base.modelbox_registry import DEFAULT_MODELBOX, MODELBOX_NAMES
    if not os.path.exists(_data_root):
        os.makedirs(_data_root)
    save_json([DEFAULT_MODELBOX, MODELBOX_NAMES], modelbox_name_file)
else:
    DEFAULT_MODELBOX, MODELBOX_NAMES = _load_json(_data_root, "modelbox-names.json")

_LR_DEFAULT: float = .01


@clicommand(message='Generating hyperparameter screening configurations')
def _hyperprep(args: Namespace) -> None:

    from .hyperparameters import HyperOpt

    configs = HyperOpt.from_file(args.input_file)

    for i, config in enumerate(configs):
        pprint_dict(config, message=f"Configuration {i}")

    output_dir = os.path.dirname(args.output)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    configs.write(
        args.output, 
        serialize=args.serialize,
    )
        
    return None


def _overwrite_config(
    config: Mapping, 
    config_file: Optional[str] = None, 
    config_idx: int = 0
) -> dict:

    from .hyperparameters import HyperOpt

    if config_file is not None:
        new_config = HyperOpt.from_file(config_file, silent=True)._ranges[config_idx] 
        pprint_dict(
            new_config,
            message=f"Overriding command-line parameters from config file {config_file}",
        )
        # new_config.update({key: val in config.items() if val is not None})  # command line takes precedent
        config.update(new_config)  # config takes precedent
        pprint_dict(
            config,
            message="Initialization parameters are now",
        )
        return config
    else:
        return config


def _init_modelbox(
    cli_config: Mapping[str, Union[str, int, float]],
    checkpoint: Optional[str] = None,
    config_file: Optional[str] = None,
    config_idx: int = 0,
    cache: Optional[str] = None,
    **overrides
):
    from .autoclass import AutoModelBox
    if checkpoint is None:
        if any(
            overrides.get(key) is None for key in ("training", "labels")
        ) and all(
            overrides.get(key) is None
            for key in ("structure", "features")
        ):
                raise ValueError(
                    """If not providing a checkpoint, --training and --labels, 
                    and either --features or --structure must be set.
                    """
                )
        cli_config = _overwrite_config(
            cli_config, 
            config_file=config_file, 
            config_idx=config_idx,
        )
        modelbox = AutoModelBox(**cli_config)._instance
    else:
        modelbox = AutoModelBox.from_pretrained(checkpoint, cache=cache)
    return modelbox
    

def _load_modelbox_training_data(
    modelbox,
    checkpoint: Optional = None,
    cache: Optional[str] = None,
    **overrides
):
    if any([
        overrides.get("training") is not None,  # override checkpoint training data
        checkpoint is None,  # no checkpoint
        modelbox.training_data is None,  # checkpoint without training data
    ]):
        load_data_args = {
            "data": overrides.get("training"), 
            "cache": cache,
            # command-line takes precedent:
            "features": overrides.get("features") or modelbox._input_featurizers,
            "labels": overrides.get("labels") or modelbox._label_cols, 
        }
        if hasattr(modelbox, "tanimoto_column"):  # i.e., is for chemistry
            # command-line takes precedent:
            load_data_args["structure_column"] = overrides.get("structure") or modelbox.structure_column
        pprint_dict(
            load_data_args,
            message="Data-loading configuration",
        )
        modelbox.load_training_data(**load_data_args)
    return modelbox, load_data_args


def _init_modelbox_and_load_training_data(
    cli_config: Mapping[str, Union[str, int, float]],
    checkpoint: Optional[str] = None,
    config_file: Optional[str] = None,
    config_idx: int = 0,
    cache: Optional[str] = None,
    **overrides
):
    modelbox = _init_modelbox(
        cli_config=cli_config,
        checkpoint=checkpoint,
        config_file=config_file,
        config_idx=config_idx,
        cache=cache,
        **overrides,
    )

    modelbox, load_data_args = _load_modelbox_training_data(
        modelbox=modelbox,
        checkpoint=checkpoint,
        cache=cache,
        **overrides,
    )

    if checkpoint is None:  # model not instantiated yet
        modelbox.model = modelbox.create_model()
    return modelbox, load_data_args


def _train_and_save_modelbox(
    modelbox,
    early_stopping: Optional[int] = None,
    prefix: str = ".",
    output_name: Optional[str] = None,
    **training_args
):
    from datasets.fingerprint import Hasher
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

    from .utils.lightning import _get_most_recent_lightning_log
    from .utils.plotting import _plot_history

    if early_stopping is not None:
        callbacks = [EarlyStopping('val_loss', patience=early_stopping)]
    else:
        callbacks = None

    if output_name is None:
        checkpoint_path = os.path.join(
            prefix, 
            f"{modelbox.class_name}_n{modelbox.size}_"
            f"y={'-'.join(modelbox._label_cols)}_"
            f"h={Hasher.hash(modelbox._input_featurizers)}",
        )
    else:
        checkpoint_path = os.path.join(
            prefix, 
            output_name,
        )
    modelbox.train(
        callbacks=callbacks,
        trainer_opts={  # passed to lightning.Trainer()
            "logger": [
                CSVLogger(save_dir=os.path.join(checkpoint_path, "logs-csv")),
                TensorBoardLogger(save_dir=os.path.join(checkpoint_path, "logs")),
            ], 
            "enable_progress_bar": True, 
            "enable_model_summary": True,
        },
        **training_args,
    )

    # get latest CSV log
    max_version = _get_most_recent_lightning_log(
        os.path.join(checkpoint_path, "logs-csv"),
        "metrics.csv",
    )
    _plot_history(
        max_version,
        os.path.join(checkpoint_path, "training-log")
    )
    modelbox.save_checkpoint(checkpoint_path)
        
    return checkpoint_path, training_args


def _evaluate_modelbox_and_save_metrics(
    modelbox,
    metric_filename: str,
    plot_filename: str,
    dataset: Optional = None,
    **kwargs
):
    import numpy as np

    from .utils.plotting import _plot_prediction_scatter
    
    predictions, metrics = modelbox.evaluate(
        data=dataset,
        aggregator=lambda x: np.mean(x, axis=-1, keepdims=True),
        **kwargs,
    )
    save_json(
        metrics, 
        metric_filename,
    )
    _plot_prediction_scatter(
        predictions,
        x=modelbox._prediction_key,
        y=modelbox._out_key,
        filename=plot_filename,
    )
    return metrics | {
        "model_class": modelbox.class_name,
        "n_parameters": modelbox.size,
    }


def _dict_to_pandas(
    d: Mapping,
    filename: Optional[str] = None
):
    import pandas as pd
    try:
        df = pd.DataFrame(d)
    except ValueError as e:  # not all columns same length; should never happen
        pprint_dict(
            {key: len(val) for key, val in d.items()},
            message="Metrics table column lengths"
        )
        raise e
    if filename is not None:
        df.to_csv(filename, index=False)
    return df


@clicommand(message='Training a Pytorch model')
def _train(args: Namespace) -> None:

    from .autoclass import AutoModelBox

    cli_config = {
        "class_name": args.model_class.casefold(),
        "use_2d": args.descriptors,
        "use_fp": args.fp,
        "n_hidden": args.hidden,
        "residual_depth": args.residual,
        "n_units": args.units,
        "dropout": args.dropout,
        "ensemble_size": args.ensemble_size,
        "learning_rate": args.learning_rate,
    }

    modelbox, load_data_args = _init_modelbox_and_load_training_data(
        cli_config=cli_config,
        checkpoint=args.checkpoint,
        config_file=args.config,
        config_idx=args.config_index,
        cache=args.cache,
        # overrides:
        training=args.training,
        structure=args.structure,
        structure_representation=args.input_representation,
        labels=args.labels,
        features=args.features,
    )

    pprint_dict(
        modelbox._model_config, 
        message=f"Initialized model {modelbox.class_name} with {modelbox.size} parameters",
    )
    
    training_args = {
        "epochs": args.epochs, 
        "batch_size": args.batch,
        "val_data": args.validation,
        "early_stopping": args.early_stopping,
    }
    pprint_dict(
        training_args, 
        message=f">> Training {modelbox.class_name} with training configuration",
    )
    checkpoint_path, training_args = _train_and_save_modelbox(
        modelbox=modelbox,
        early_stopping=args.early_stopping,
        prefix=args.prefix,
        epochs=args.epochs, 
        batch_size=args.batch,
        val_data=args.validation,
    )
    for obj, f in zip((training_args, load_data_args), ("training-args.json", "load-data-args.json")):
        save_json(obj, os.path.join(checkpoint_path, f))

    # Reload - built-in test that the checkpointing works!
    modelbox = AutoModelBox.from_pretrained(
        checkpoint_path, 
        cache=args.cache,
    )
    overall_metrics = defaultdict(list)
    for name in ("training", "validation", "test"):
        dataset = getattr(args, name)
        if dataset is not None:  # skip optional extra datasets, e.g. "test"
            if name == "training":
                dataset = None  # Use cached training data
            metrics = _evaluate_modelbox_and_save_metrics(
                modelbox,
                metric_filename=f"eval-metrics_{name}.json",
                plot_filename=f"predictions_{name}",
                dataset=dataset,
            )
            pprint_dict(
                metrics, 
                message=f"Evaluation: {name}",
            )

            overall_metrics["split"].append(name)
            overall_metrics["split_filename"].append(dataset or load_data_args["data"])
            if args.config is not None:
                overall_metrics["config_i"].append(args.config_index)
            keys_added = []
            for d in (
                modelbox._init_kwargs, 
                modelbox._model_config, 
                load_data_args, 
                training_args, 
                metrics,
            ):
                for key, val in d.items():
                    if key != "trainer_opts" and key not in keys_added:
                        overall_metrics[key].append(val)
                        keys_added.append(key)

    _dict_to_pandas(overall_metrics, os.path.join(checkpoint_path, "metrics.csv"))

    return None


def _resolve_and_slice_data(
    data: str,
    start: Optional[int] = None,
    end: Optional[int] = None
):
    from .base.data import DataMixinBase
    from .utils.datasets import to_dataset

    candidates_ds = DataMixinBase._resolve_data(data)
    skip = start or 0
    take = (end or candidates_ds.num_rows) - skip
    return to_dataset(
        candidates_ds
        .to_iterable_dataset()
        .skip(skip)
        .take(take)
    )


def _save_dataset(
    dataset,
    output: str
) -> None:
    print_err("INFO: Saving dataset:\n" + str(dataset) + "\n" + f"at {output} as", end=" ")
    if output.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        print_err("CSV.")
        dataset.to_csv(
            output, 
            sep="," if output.endswith((".csv", ".csv.gz")) else "\t",
        )
    elif output.endswith(".json"):
        print_err("JSON.")
        dataset.to_json(output)
    elif output.endswith(".parquet"):
        print_err("Parquet.")
        dataset.to_parquet(output)
    elif output.endswith(".sql"):
        print_err("SQL.")
        dataset.to_sql(output)
    elif output.endswith(".hf"):
        print_err("Hugging Face dataset.")
        dataset.save_to_disk(output)
    else:
        print_err("Hugging Face dataset.")
        dataset.save_to_disk(output + ".hf")
        print_err(f"WARNING: Unsure what format to save as for filename {output}. Defaulted to Hugging Face dataset.")
    return None


@clicommand("Predicting with the following parameters")
def _predict(args: Namespace) -> None:

    output = os.path.join(args.prefix, args.output)
    out_dir = os.dirname(output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    preprocessing_args = {
        "features": args.features,
        "structure_column": args.structure,
        "structure_representation": args.input_representation,
    }
    common_args = {
        "batch_size": args.batch,
        "cache": args.cache,
    }

    candidates_ds = _resolve_and_slice_data(
        args.test,
        start=args.start,
        end=args.end,
    )

    modelbox = AutoModelBox.from_pretrained(
        args.checkpoint, 
        cache=args.cache,
    )
    candidates_ds = modelbox.predict(
        data=candidates_ds,
        aggregator="mean",
        **preprocessing_args,
        **common_args,
    )
    if args.variance:
        candidates_ds = modelbox.variance(
            candidates=candidates_ds,
            **preprocessing_args,
            **common_args,
        )
    if args.tanimoto:
        if hasattr(modelbox, "tanimoto_nn"):
            candidates_ds = modelbox.tanimoto_nn(
                data=candidates_ds,
                query_structure_column=args.structure,
                query_structure_representation=args.input_representation,
                **common_args,
            )
        else:
            print_err(f"Cannot calculate Tanimoto for non-chemical modelbox from {args.checkpoint}")
    if args.doubtscore:
        modelbox.model.set_model(0)
        candidates_ds = modelbox.doubtscore(
            candidates=candidates_ds,
            preprocessing_args=preprocessing_args,
            **common_args,
        )
    if args.information_sensitivity:
        modelbox.model.set_model(0)
        if args.approx == "bekas":
            extra_args = {"n": args.bekas_n}
        else:
            extra_args = {}
        candidates_ds = modelbox.information_sensitivity(
            candidates=candidates_ds,
            approximator=args.approx,
            optimality_approximation=args.optimality,
            preprocessing_args=preprocessing_args,
            **common_args,
            **extra_args,
        )

    _save_dataset(candidates_ds, output)

    if args.labels is not None:
        overall_metrics = defaultdict(list)
        metric_filename = os.path.join(args.prefix, "predict-eval-metrics.csv")
        plot_filename = os.path.join(args.prefix, "predict-eval-metrics.png")
        metrics = _evaluate_modelbox_and_save_metrics(
            modelbox,
            dataset=candidates_ds,
            **preprocessing_args,
            metric_filename=metric_filename,
            plot_filename=plot_filename,
        )
        pprint_dict(
            metrics, 
            message=f"Evaluation",
        )
        overall_metrics["model_class"].append(modelbox.class_name)
        overall_metrics["n_parameters"].append(modelbox.size)
        keys_added = []
        for d in (
            modelbox._init_kwargs, 
            modelbox._model_config, 
            metrics,
        ):
            for key, val in d.items():
                if key != "trainer_opts" and key not in keys_added:
                    overall_metrics[key].append(val)
                    keys_added.append(key)
        _dict_to_pandas(overall_metrics, os.path.join(args.prefix, "metrics.csv"))

    return None


def main() -> None:

    input_file = CLIOption(
        'input_file',
        type=FileType('r'),
        default=sys.stdin,
        nargs='?',
        help='Input file. Default: STDIN',
    )
    train_data = CLIOption(
        '--training', '-1',
        type=str,
        default=None,
        help='Training dataset. Required if no checkpoint provided.',
    )
    val_data = CLIOption(
        '--validation', '-2',
        type=str,
        required=True,
        help='Validation dataset file.',
    )
    test_data = CLIOption(
        '--test', '-t',
        type=str,
        default=None,
        help='Test dataset file.',
    )
    feature_cols = CLIOption(
        '--features', '-x',
        type=str,
        nargs='*',
        default=None,
        help='Column names from data file that contain features. Required if no checkpoint provided.',
    )
    structure_col = CLIOption(
        '--structure', '-S',
        type=str,
        default=None,
        help="""
        Column names from data file that contains a string representation 
        of chemical structure. Required for chemical model classes.
        """,
    )
    structure_representation = CLIOption(
        '--input-representation', '-R',
        type=str,
        default=None,
        choices=["smiles", "selfies", "inchi", "aa_seq"],
        help='Type of chemical structure string. Default: SMILES if training; for prediction, use same as training data.',
    )
    label_cols = CLIOption(
        '--labels', '-y',
        type=str,
        nargs='*',
        default=None,
        help='Column names from data file that contain labels. Required if no checkpoint provided.',
    )
    cache = CLIOption(
        '--cache',
        type=str,
        default=".",
        help='Where to cache data.',
    )
    model_class = CLIOption(
        '--model-class', '-k',
        type=str,
        default=DEFAULT_MODELBOX,
        choices=MODELBOX_NAMES,
        help='Test dataset file.',
    )
    _checkpoint = CLIOption(
        '--checkpoint',
        type=str,
        default=None,
        help='Load a modelbox from this checkpoint. Default: do not use, make a new modelbox.',
    )
    n_units = CLIOption(
        '--units', '-u',
        type=int,
        default=8,
        help='Number of units per hidden layer.',
    )
    n_hidden = CLIOption(
        '--hidden', '-m',
        type=int,
        default=1,
        help='Number of hidden layers.',
    )
    residual_depth = CLIOption(
        '--residual',
        type=int,
        default=None,
        help='Depth of residual blocks. Default: Do not use residual blocks.',
    )
    _2d = CLIOption(
        '--descriptors', 
        action="store_true",
        help='Use 2d descriptors (needs a SMILES input feature).',
    )
    _fp = CLIOption(
        '--fp', 
        action="store_true",
        help='Use chemical fingerprints (needs a SMILES input feature).',
    )
    dropout = CLIOption(
        '--dropout', '-d',
        type=float,
        default=0.,
        help='Dropout rate for training.',
    )
    ensemble_size = CLIOption(
        '--ensemble-size', '-z',
        type=int,
        default=1,
        help='Number of models to train in an ensemble.',
    )
    batch_size = CLIOption(
        '--batch', '-b',
        type=int,
        default=16,
        help='Batch size for training.',
    )
    n_epochs = CLIOption(
        '--epochs', '-e',
        type=int,
        default=1,
        help='Number of epochs for training.',
    )
    learning_rate = CLIOption(
        '--learning-rate', '-r',
        type=float,
        default=None,
        help=f'Learning rate for training. Default: {_LR_DEFAULT}.',
    )
    early_stopping = CLIOption(
        '--early-stopping', '-s',
        type=int,
        default=None,
        help='Number of epochs to wait for improvement before stopping. Default: no early stopping.',
    )
    model_config = CLIOption(
        '--config', '-c',
        type=str,
        default=None,
        help='Model configuration file. Overrides other options.',
    )
    config_i = CLIOption(
        '--config-index', '-i',
        type=int,
        default=0,
        help='If more than one config in `--config`, choose this one.',
    )
    save_prefix = CLIOption(
        '--prefix', '-p',
        type=str,
        default="duvida-checkpoint",
        help='Prefix to save model checkpoints.',
    )
    output_name = CLIOption(
        '--output', '-o', 
        type=str,
        required=True,
        help='Output filename.',
    )
    serialize = CLIOption(
        '--serialize', '-z', 
        action="store_true",
        help='Pickle instead of JSON output.',
    )
    # output = CLIOption(
    #     '--output', '-o', 
    #     type=FileType('w'),
    #     default=sys.stdout,
    #     help='Output file. Default: STDOUT',
    # )
    # formatting = CLIOption(
    #     '--format', '-f', 
    #     type=str,
    #     default='TSV',
    #     choices=['TSV', 'CSV', 'tsv', 'csv'],
    #     help='Format of files. Default: %(default)s',
    # )

    # slice dataset
    slice_start = CLIOption(
        '--start', 
        type=int,
        default=0,
        help='First row of dataset to process.',
    )
    slice_end = CLIOption(
        '--end', 
        type=int,
        default=None,
        help='Last row of dataset to process. Default: end of dataset.',
    )

    # Information metrics
    variance = CLIOption(
        '--variance', 
        action="store_true",
        help='Calculate ensemble variance.',
    )
    tanimoto = CLIOption(
        '--tanimoto', 
        action="store_true",
        help='Calculate Tanimoto distance to nearest neighbor in training data.',
    )
    doubtscore = CLIOption(
        '--doubtscore', 
        action="store_true",
        help='Calculate doubtscore.',
    )
    info_sens = CLIOption(
        '--information-sensitivity', 
        action="store_true",
        help='Calculate information senstivity.',
    )
    optimality = CLIOption(
        '--optimality', 
        action="store_true",
        help='Whether to make the computationally faster assumption that the model parameters were trained to gradient 0.',
    )
    hess_approx = CLIOption(
        '--approx', 
        type=str,
        default="bekas",
        choices=["exact", "squared_jacobian", "rough_finite_difference", "bekas"],
        help='What type of Hessian approximation to perform.',
    )
    bekas_n = CLIOption(
        '--bekas-n', 
        type=int,
        default=1,
        help='Number of stochastic samples for Hessian approximation.',
    )

    hyperprep = CLICommand(
        "hyperprep",
        description="Prepare inputs for hyperparameter search.",
        options=[
            input_file, 
            serialize,
            output_name,
        ],
        main=_hyperprep,
    )

    train = CLICommand(
        "train",
        description="Train a PyTorch model.",
        options=[
            train_data, 
            val_data,
            test_data,
            feature_cols,
            structure_col,
            structure_representation,
            label_cols,
            model_class,
            _checkpoint,
            n_units,
            n_hidden,
            residual_depth,
            _2d, 
            _fp,
            dropout,
            batch_size,
            n_epochs,
            learning_rate,
            early_stopping,
            ensemble_size,
            model_config,
            config_i,
            save_prefix,
            output_name,
            cache,
        ],
        main=_train,
    )

    predict = CLICommand(
        "predict",
        description="Make predictions and calculate uncertainty using a duvida checkpoint.",
        options=[
            test_data, 
            slice_start,
            slice_end,
            feature_cols,
            label_cols,
            structure_col,
            structure_representation,
            _checkpoint,
            save_prefix,
            cache,
            output_name,
            variance,
            tanimoto,
            doubtscore,
            info_sens,
            optimality,
            hess_approx,
            bekas_n,
            batch_size,
        ],
        main=_predict,
    )

    app = CLIApp(
        "duvida", 
        version=__version__,
        description=(
            "Calculating exact and approximate confidence and "
            "information metrics for deep learning on general "
            "purpose and chemistry tasks."
        ),
        commands=[
            hyperprep,
            train,
            predict,
        ],
    )

    app.run()
    return None


if __name__ == '__main__':
    main()
