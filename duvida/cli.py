"""Command-line interface for duvida."""

from argparse import FileType, Namespace
from collections import defaultdict
from glob import glob
import json
import os
import sys

from carabiner.utils import pprint_dict
from carabiner.mpl import grid, figsaver
from carabiner.cliutils import clicommand, CLIOption, CLICommand, CLIApp
# from  pandas import DataFrame

from . import __version__
from .autoclass import (
    AutoModelBox, 
    DEFAULT_MODELBOX, 
    MODELBOX_NAMES
)
from .hyperparameters import HyperOpt

_LR_DEFAULT: float = .01


def _plot_history(
    lightning_csv, 
    filename: str
) -> None:

    from carabiner.mpl import add_legend, grid, figsaver
    import pandas as pd
    import numpy as np

    data_to_plot = (
        pd.read_csv(lightning_csv)
        .groupby(['epoch', 'step'])
        .agg(np.nanmean)
        .reset_index()
    )

    fig, ax = grid(aspect_ratio=1.5)
    for _y in ('val_loss', 'loss', 'learning_rate'):
        if _y in data_to_plot:
            ax.plot(
                'step', _y, 
                data=data_to_plot, 
                label=_y,
            )
            ax.scatter(
                'step', _y, 
                data=data_to_plot,
                s=1.,
            )
    add_legend(ax)
    ax.set(
        xlabel='Training step', 
        ylabel='Loss', 
        yscale='log',
    )
    figsaver(format="png")(fig, name=filename, df=data_to_plot)
    return None


def _get_most_recent_lightning_log(
    save_dir: str,
    filename: str,
    name: str = "lightning_logs",
    version_prefix: str = "version"
) -> str:
    path = os.path.join(save_dir, name)
    logs = sorted(glob(os.path.join(path, f"{version_prefix}_*")))
    max_version = max([int(v.split("_")[-1]) for v in logs])
    max_version = os.path.join(path, f"version_{max_version}", filename)
    if os.path.exists(max_version):
        return max_version
    else:
        raise OSError(f"Could not find most recent log: {max_version}")


def _plot_prediction_scatter(
    df,
    filename: str,
    x: str = "__prediction__",
    y: str = "labels"
) -> None:
    fig, ax = grid()
    ax.scatter(
        x, y,
        data=df,
        s=1.,
    )
    ax.plot(
        ax.get_ylim(),
        ax.get_ylim(),
        color='dimgrey',
        zorder=-5,
    )
    ax.set(
        xlabel=f"Predicted ({x})", 
        ylabel=f"Observed ({y})",
    )
    figsaver(format="png")(
        fig,
        filename,
        df=df,
    )
    return None

@clicommand(message='Generating hyperparameter screening configurations')
def _hyperprep(args: Namespace) -> None:

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


@clicommand(message='Training a Pytorch model')
def _train(args: Namespace) -> None:

    from datasets.fingerprint import Hasher
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    import pandas as pd
    import numpy as np

    from .checkpoint_utils import save_json

    str_config = {
        "class_name": args.model_class.casefold(),
        "use_2d": args.descriptors,
        "use_fp": args.fp,
        "n_hidden": args.hidden,
        "n_units": args.units,
        "dropout": args.dropout,
        "ensemble_size": args.ensemble_size,
        "learning_rate": args.learning_rate,
    }

    if args.checkpoint is not None:
        modelbox = AutoModelBox.from_pretrained(args.checkpoint, cache=args.cache)
    else:
        if args.config is not None:
            new_config = HyperOpt.from_file(args.config, silent=True)._ranges[args.config_index] 
            pprint_dict(
                new_config,
                message=f"Overriding command-line parameters from config file {args.config}",
            )
            str_config.update(new_config)  # override command line
            pprint_dict(
                str_config,
                message="Initialization parameters are now",
            )
        if any(
            a is None for a in (args.training, args.labels)
        ):
            if all([args.structure is None, args.features is None]):
                raise ValueError(
                    """If not providing a checkpoint, --training and --labels, 
                    and either --features or --structure must be set.
                    """
                )
        modelbox = AutoModelBox(**str_config)._instance

    pprint_dict(
        modelbox._model_config,
        message=f"Initialized class {modelbox.class_name}",
    )
    load_data_args = {
        "data": args.training, 
        "features": args.features or modelbox._input_featurizers,
        "labels": args.labels or modelbox._label_cols, 
        "cache": args.cache,
    }
    if modelbox.class_name not in ("mlp", ):
        load_data_args["structure_column"] = args.structure or modelbox.structure_column
    if (
        args.checkpoint is None 
        or args.training is not None 
        or modelbox.training_data is None
    ):
        modelbox.load_training_data(**load_data_args)
    if args.checkpoint is None:
        modelbox.model = modelbox.create_model()
    pprint_dict(
        modelbox._model_config, 
        message=f"Model {modelbox.class_name} with {modelbox.size} parameters",
    )
    _featurizers = modelbox._input_featurizers
    _labels = load_data_args["labels"]
    save_prefix = os.path.join(
        args.prefix, 
        f"{modelbox.class_name}_n{modelbox.size}_y={'-'.join(_labels)}_h={Hasher.hash(_featurizers)}",
    )
    training_args = {
        "val_data": args.validation,
        "epochs": args.epochs, 
        "batch_size": args.batch,
    }
    # if not modelbox.class_name in ("mlp", ):
    #     training_args["structure_column"] = load_data_args["structure_column"]
    pprint_dict(
        modelbox._model_config, 
        message=f">> Training {modelbox.class_name} with configuration",
    )
    if args.early_stopping is not None:
        callbacks = [EarlyStopping('val_loss', patience=args.early_stopping)]
    else:
        callbacks = None
    modelbox.train(
        callbacks=callbacks,
        trainer_opts={  # passed to lightning.Trainer()
            "logger": [
                CSVLogger(save_dir=os.path.join(save_prefix, "logs-csv")),
                TensorBoardLogger(save_dir=os.path.join(save_prefix, "logs")),
            ], 
            "enable_progress_bar": True, 
            "enable_model_summary": True,
        },
        **training_args,
    )

    # get latest CSV log
    max_version = _get_most_recent_lightning_log(
        os.path.join(save_prefix, "logs-csv"),
        "metrics.csv",
    )
    _plot_history(
        max_version,
        os.path.join(save_prefix, "training-log")
    )
    checkpoint_path = os.path.join(
        save_prefix,
        # "checkpoint.dv",
    )
    modelbox.save_checkpoint(checkpoint_path)
    for filename, obj in zip(
        ("data-load-args", "training-args"),
        (load_data_args, training_args),
    ):
        save_json(obj, os.path.join(checkpoint_path, f"{filename}.json"))

    # Check it loads
    # modelbox.load_checkpoint(checkpoint_path)
    modelbox = AutoModelBox.from_pretrained(checkpoint_path)
    overall_metrics = defaultdict(list)
    for name in ("training", "validation", "test"):
        dataset = getattr(args, name)
        if dataset is not None:
            if name == "training":
                dataset = None  # Use cached
            predictions, metrics = modelbox.evaluate(
                data=dataset,
                aggregator=lambda x: np.mean(x, axis=-1, keepdims=True),
            )
            with open(os.path.join(save_prefix, f"eval-metrics_{name}.json"), "w") as f:
                json.dump(metrics, f, sort_keys=True, indent=4)
            _plot_prediction_scatter(
                predictions,
                x=modelbox._prediction_key,
                y=modelbox._out_key,
                filename=os.path.join(save_prefix, f"predictions_{name}"),
            )
            pprint_dict(
                metrics, 
                message=f"Evaluation: {name}",
            )
            overall_metrics["split"].append(name)
            overall_metrics["split_filename"].append(dataset or load_data_args["data"])
            if args.config is not None:
                overall_metrics["config_i"].append(args.config_index)
            overall_metrics["model_class"].append(modelbox.class_name)
            overall_metrics["n_parameters"].append(modelbox.size)
            keys_added = []
            for d in (
                load_data_args, 
                modelbox._init_kwargs, 
                modelbox._model_config, 
                training_args, 
                metrics,
            ):
                for key, val in d.items():
                    if key != "trainer_opts" and key not in keys_added:
                        overall_metrics[key].append(val)
                        keys_added.append(key)
    try:
        pd.DataFrame(overall_metrics).to_csv(
            os.path.join(save_prefix, "metrics.csv"),
            index=False,
        )
    except ValueError as e:  # not all columns same length; should never happen
        pprint_dict(
            {key: len(val) for key, val in overall_metrics.items()},
            message="Metrics table column lengths"
        )
        raise e
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
        default="smiles",
        choices=["smiles", "selfies", "inchi", "aa_seq"],
        help='Type of chemical structure string.',
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
        help='Load a modelbox from this checkpoint. Default: do not use.',
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
            cache,
        ],
        main=_train,
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
        ],
    )

    app.run()
    return None


if __name__ == '__main__':
    main()
