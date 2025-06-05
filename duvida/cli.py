"""Command-line interface for duvida."""

from typing import Mapping, Optional, Union

from argparse import FileType, Namespace
from collections import defaultdict
import os
import sys

from carabiner import cast, pprint_dict, print_err
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
        os.makedirs(output_dir)
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
        output_name = (
            f"{modelbox.class_name}_n{modelbox.size}_"
            f"y={'-'.join(modelbox._label_cols)}_"
            f"h={Hasher.hash(modelbox._input_featurizers)}",
        )
    
    checkpoint_path = output_name
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
    import torch
    from .utils.plotting import _plot_prediction_scatter

    modelbox.to("cuda" if torch.cuda.is_available() else "cpu")
    predictions, metrics = modelbox.evaluate(
        data=dataset,
        aggregator="mean",
        agg_kwargs={"keepdims": True},
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
        epochs=args.epochs, 
        batch_size=args.batch,
        val_data=args.validation,
        output_name=args.output,
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
                metric_filename=os.path.join(checkpoint_path, f"eval-metrics_{name}.json"),
                plot_filename=os.path.join(checkpoint_path, f"predictions_{name}"),
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
    end: Optional[int] = None,
    batch_size: int = 1024
):
    from .base.data import DataMixinBase
    # from .utils.datasets import to_dataset

    candidates_ds = DataMixinBase._resolve_data(data)
    nrows = candidates_ds.num_rows
    skip = start or 0
    take = (end or nrows) - skip
    if (take - skip) < nrows:
        print_err(f"INFO: Reading dataset from row {skip} to row {take + skip} / {nrows}.")
    return candidates_ds.skip(skip).take(take)


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
            compression='gzip' if output.endswith(".gz") else None,
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

    import torch
    from .autoclass import AutoModelBox

    output = args.output
    out_dir = os.path.dirname(output)
    if len(out_dir) > 0 and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    preprocessing_args = {
        "structure_column": args.structure,
        "input_representation": args.input_representation,
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
    pprint_dict(
        modelbox._model_config, 
        message=f"Initialized model {modelbox.class_name} with {modelbox.size} parameters",
    )
    for col in cast(modelbox._label_cols, to=list):
        if col not in candidates_ds.column_names:
            from numpy import zeros_like
            candidates_ds = candidates_ds.add_column(
                col,
                zeros_like(
                    candidates_ds
                    .with_format("numpy")
                    [candidates_ds.column_names[0]]
                )
            )
    
    preprocessing_args["_extra_cols_to_keep"] = (args.extras or [])
    modelbox.to("cuda" if torch.cuda.is_available() else "cpu")
    candidates_ds = modelbox.predict(
        data=candidates_ds,
        aggregator="mean",
        features=args.features,
        **preprocessing_args,
        **common_args,
    )
    preprocessing_args["_extra_cols_to_keep"].append(modelbox._prediction_key)
    if args.variance:
        candidates_ds = modelbox.prediction_variance(
            candidates=candidates_ds,
            features=args.features,
            **preprocessing_args,
            **common_args,
        )
        preprocessing_args["_extra_cols_to_keep"].append(modelbox._variance_key)
    if args.tanimoto:
        if hasattr(modelbox, "tanimoto_nn"):
            candidates_ds = modelbox.tanimoto_nn(
                data=candidates_ds,
                query_structure_column=args.structure,
                query_input_representation=args.input_representation,
                **common_args,
            )
            preprocessing_args["_extra_cols_to_keep"].append(modelbox.tanimoto_column)
        else:
            print_err(f"Cannot calculate Tanimoto for non-chemical modelbox from {args.checkpoint}")
    if args.doubtscore:
        modelbox.model.set_model(0)
        candidates_ds = modelbox.doubtscore(
            candidates=candidates_ds,
            features=args.features,
            preprocessing_args=preprocessing_args,
            **common_args,
        )
        preprocessing_args["_extra_cols_to_keep"].append("doubtscore")
    if args.information_sensitivity:
        modelbox.model.set_model(0)
        if args.approx == "bekas":
            extra_args = {"n": args.bekas_n}
        else:
            extra_args = {}
        candidates_ds = modelbox.information_sensitivity(
            candidates=candidates_ds,
            features=args.features,
            preprocessing_args=preprocessing_args,
            approximator=args.approx,
            optimality_approximation=args.optimality,
            **common_args,
            **extra_args,
        )
        preprocessing_args["_extra_cols_to_keep"].append("information sensitivity")
        
    print_err(preprocessing_args)

    _save_dataset(
        candidates_ds.remove_columns([modelbox._in_key, modelbox._out_key]), 
        output,
    )

    if args.labels is not None:
        overall_metrics = defaultdict(list)
        metric_filename = os.path.join(out_dir, "predict-eval-metrics-table.csv")
        plot_filename = os.path.join(out_dir, "predict-eval-scatter")
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
        keys_added = set(overall_metrics.keys())
        for d in (
            modelbox._init_kwargs, 
            modelbox._model_config, 
            metrics,
        ):
            for key, val in d.items():
                if key != "trainer_opts" and key not in keys_added:
                    overall_metrics[key].append(val)
                    keys_added.add(key)
        _dict_to_pandas(
            overall_metrics, 
            os.path.join(out_dir, "metrics.csv"),
        )

    return None


@clicommand("Splitting data with the following parameters")
def _split(args: Namespace) -> None:

    from .utils.splitting import split_dataset
    output = args.output
    out_dir = os.path.dirname(output)
    if len(out_dir) > 0 and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    if args.train is None:
        raise ValueError(f"You need to at least provide a --train fraction.")
    
    ds = _resolve_and_slice_data(
        args.input_file,
        start=args.start,
        end=args.end,
    )
    if args.type == "faiss":
        faiss_opts = {
            "cache": args.cache,
            "n_neighbors": args.n_neighbors,
        }
    else:
        faiss_opts = {}
    ds, splits = split_dataset(
        ds=ds,
        method=args.type,
        structure_column=args.structure,
        input_representation=args.input_representation,
        train=args.train,
        validation=args.validation,
        test=args.test,
        batch_size=args.batch,
        seed=args.seed or 42,
        deterministic=args.seed is not None,
        **faiss_opts,
    )
    root, ext = os.path.splitext(output)
    for key, split_ds in splits.items():
        _save_dataset(
            split_ds, 
            f"{root}_{key}{ext}",
        )

    if args.plot is not None:

        from carabiner.mpl import figsaver
        from .utils.splitting.plot import plot_chemical_splits

        print_err(f"Plotting splits...")
        
        (fig, axes), df = plot_chemical_splits(
            ds=ds,
            structure_column=args.structure,
            input_representation=args.input_representation,
            split_columns="split",
            sample_size=args.plot_sample,
            additional_columns=args.extras,
            seed=args.plot_seed,
            cache=args.cache,
        )
        root, ext = os.path.splitext(args.plot)
        figsaver(format=ext.lstrip("."))(fig, root, df=df)
    return None


@clicommand("Tagging data percentiles with the following parameters")
def _percentile(args: Namespace) -> None:

    if args.plot is not None and args.structure is None:
        raise ValueError(
            f"""
            If you want to save a plot at "{args.plot}", then you need to provide
            a chemical structure (like SMILES) column name using --structure so 
            that a UMAP embedding can be calculated.
            """
        )

    from .utils.splitting.top_k import percentiles
    
    output = args.output
    out_dir = os.path.dirname(output)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    ds = _resolve_and_slice_data(
        args.input_file,
        start=args.start,
        end=args.end,
    )
    print(ds)
    # for b in ds:
    #     print(b)
    #     break
    q = {col: args.percentiles for col in args.columns}
    ds = percentiles(
        ds=ds,
        q=q,
        compression=args.compression,
        delta=args.delta,
        reverse=args.reverse,
        cache=args.cache,
    )
    _save_dataset(ds, output)

    if args.plot is not None:

        from carabiner.mpl import figsaver
        from .utils.splitting.plot import plot_chemical_splits

        print_err(f"Plotting top percentiles...")

        (fig, axes), df = plot_chemical_splits(
            ds=ds,
            structure_column=args.structure,
            input_representation=args.input_representation,
            split_columns=[col for col in ds.column_names if any(f"{_q}_top_" in col for _q in q)],
            sample_size=args.plot_sample,
            additional_columns=args.extras,
            seed=args.plot_seed,
            cache=args.cache,
        )
        root, ext = os.path.splitext(args.plot)
        figsaver(format=ext.lstrip("."))(fig, root, df=df)
    return None


def main() -> None:

    input_file = CLIOption(
        'input_file',
        type=FileType('r'),
        default=sys.stdin,
        nargs='?',
        help='Input file. Default: STDIN',
    )
    input_filename = CLIOption(
        'input_file',
        type=str,
        help='Input file.',
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
        default="smiles",
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
    extra_cols = CLIOption(
        '--extras',
        type=str,
        nargs="*",
        default=None,
        help='Extra columns to retain in prediction table; useful for IDs.',
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

    split_type = CLIOption(
        '--type', 
        type=str,
        default="scaffold",
        choices=["scaffold", "faiss"],
        help='Splitting method.',
    )
    random_seed = CLIOption(
        '--seed', '-i', 
        type=int,
        default=None,
        help='Random seed. Default: determininstic.',
    )
    n_neighbors = CLIOption(
        '--n-neighbors', '-k', 
        type=int,
        default=10,
        help='Number of nearest neighbors for FAISS splitting.',
    )
    train_test_val = [
        CLIOption(
            f'--{key}', 
            type=float,
            default=None,
            help='Fraction of examples for each split. Default: infer.',
        ) for key in ("train", "validation", "test")
    ]

    columns = CLIOption(
        '--columns', '-c',
        type=str,
        nargs='*',
        help='List of columns to tag percentiles for.',
    )
    percentiles = CLIOption(
        '--percentiles', '-p', 
        type=float,
        nargs='*',
        default=[5.],
        help='List of percentiles to calculate.',
    )
    reverse = CLIOption(
        '--reverse', '-r', 
        action='store_true',
        help='Whether to reverse percentiles (i.e. high to low).',
    )
    do_plot = CLIOption(
        '--plot', 
        type=str,
        default=None,
        help='Filename to save UMAP plot under.',
    )
    compression = CLIOption(
        '--compression', '-z', 
        type=int,
        default=500,
        help='How many centroids for quantile approximation. Higher is more accurate, but uses more memory.',
    )
    delta = CLIOption(
        '--delta', '-d', 
        type=float,
        default=1.,
        help='Width from percentile cutoff to buffer as borderline for refinement. Higher is more accurate, but uses more memory.',
    )
    plot_seed = CLIOption(
        '--plot-seed', '-e', 
        type=int,
        default=42,
        help='Seed for UMAP embedding.',
    )
    plot_sample = CLIOption(
        '--plot-sample', '-n', 
        type=int,
        default=20_000,
        help='Subsample size for UMAP embedding.',
    )
    extras = CLIOption(
        '--extras', '-x',
        type=str,
        nargs='*',
        help='Additional columns for coloring UMAP plot.',
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
            extra_cols,
            structure_representation,
            _checkpoint,
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

    split = CLICommand(
        "split",
        description="Make chemical train-test-val splits on out-of-core datasets.",
        options=[
            input_filename, 
            split_type,
            n_neighbors,
            slice_start,
            slice_end,
            structure_col,
            structure_representation,
            do_plot,
            plot_sample,
            plot_seed,
            extras,
            random_seed,
            cache,
            output_name,
            batch_size,
        ] + train_test_val,
        main=_split,
    )

    percentiles = CLICommand(
        "percentiles",
        description="Add columns indicating whether rows are in a percentile.",
        options=[
            input_filename, 
            columns,
            percentiles,
            reverse,
            compression,
            delta,
            slice_start,
            slice_end,
            cache,
            output_name,
            batch_size,
            do_plot,
            structure_col,
            structure_representation,
            plot_sample,
            plot_seed,
            extras,
        ],
        main=_percentile,
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
            split,
            percentiles,
        ],
    )

    app.run()
    return None


if __name__ == '__main__':
    main()
