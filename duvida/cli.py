"""Command-line interface for duvida."""

from argparse import FileType, Namespace
from collections import defaultdict
from glob import glob
import json
import os
import pickle
import sys

from carabiner import print_err, pprint_dict
from carabiner.cliutils import clicommand, CLIOption, CLICommand, CLIApp

from . import __version__
from .autoclass import AutoModelBox, _MODEL_CLASS_DEFAULT, _MODEL_CLASSES
from .hyperparameters import HyperOpt
from .evaluation import rmse, pearson_r, spearman_r

_LR_DEFAULT: float = .01

def _plot_history(
    lightning_csv, 
    filename: str
) -> None:

    from carabiner.mpl import add_legend, colorblind_palette, grid, figsaver
    import pandas as pd
    import numpy as np

    data_to_plot = (
        pd.read_csv(lightning_csv)
        .groupby(['epoch', 'step'])
        .agg(np.nanmean)
        .reset_index()
    )

    fig, ax = grid(aspect_ratio=1.5)
    for _y in ('val_loss', 'loss'):
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

    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
    import pandas as pd
    import torch

    str_config = {
        "model_class": args.model_class.casefold(),
        "use_2d": args.descriptors,
        "use_fp": args.fp,
        "n_hidden": args.hidden,
        "n_units": args.units,
        "dropout": args.dropout,
        "ensemble_size": args.ensemble_size,
        "learning_rate": args.learning_rate,
    }

    if args.checkpoint is not None:
        modelbox = AutoModelBox.from_pretrained(args.checkpoint)
    else:
        if args.config is not None:
            new_config = HyperOpt.from_file(args.config, silent=True)._ranges[args.config_index] 
            str_config.update(new_config)
        if any(a is None for a in (args.training, args.features, args.labels)):
            raise ValueError("If not providing a checkpoint, --training, --features, and --labels must be set.")
        modelbox = AutoModelBox(
            **str_config,
        )._instance

    load_data_args = {
        "filename": args.training, 
        "features": args.features or modelbox._input_cols,
        "labels": args.labels or modelbox._label_cols, 
        "cache": args.cache,
    }
    if (
        args.checkpoint is None 
        or args.training is not None 
        or modelbox.training_data is None
    ):
        modelbox.load_training_data(**load_data_args)
    if args.checkpoint is None:
        modelbox.model = modelbox.create_model()
    pprint_dict(
        modelbox._input_configuration, 
        message=f"Model {modelbox.__class__.__name__} with {modelbox.size} parameters.",
    )
    _features = load_data_args["features"]
    _labels = load_data_args["labels"]
    save_prefix = os.path.join(
        args.prefix, 
        f"{modelbox.__class__.__name__}_n{modelbox.size}_x={'-'.join(_features)}_y={'-'.join(_labels)}_epochs={args.epochs}",
    )
    training_args = {
        "val_filename": args.validation,
        "epochs": args.epochs, 
        "batch_size": args.batch,
    }
    pprint_dict(
        modelbox._input_configuration, 
        message=f">> Training {modelbox.__class__.__name__} with configuration",
    )
    if args.early_stopping is not None:
        callbacks = [EarlyStopping('val_loss', patience=args.early_stopping)]
    else:
        callbacks = None
    modelbox.train(
        callbacks=callbacks,
        trainer_opts={  # passed to lightning.Trainer()
            # "logger": True, 
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
        with open(os.path.join(checkpoint_path, f"{filename}.json"), "w") as f:
            json.dump(obj, f, sort_keys=True, indent=4)

    # Check it loads
    # modelbox.load_checkpoint(checkpoint_path)
    modelbox = AutoModelBox.from_pretrained(checkpoint_path)
    overall_metrics = defaultdict(list)
    for name, dataset in zip(
        ("train", "validation", "test"),
        (args.training, args.validation, args.test),
    ):
        if dataset is not None:
            if name == "train":
                dataset = None  # Use cached
            predictions, metrics = modelbox.evaluate(
                filename=dataset,
                metrics={
                    "RMSE": rmse, 
                    "Pearson r": pearson_r, 
                    "Spearman rho": spearman_r,
                },
            )
            with open(os.path.join(save_prefix, f"eval-metrics_{name}.json"), "w") as f:
                json.dump(metrics, f, sort_keys=True, indent=4)
            predictions.to_csv(
                os.path.join(save_prefix, f"predictions_{name}.csv"),
                index=False,
            )
            pprint_dict(
                metrics, 
                message=f"Evaluation: {name}",
            )
            overall_metrics["split"].append(name)
            overall_metrics["split_filename"].append(dataset or load_data_args["filename"])
            for d in (load_data_args, modelbox._input_configuration, training_args, metrics):
                for key, val in d.items():
                    if key != "trainer_opts":
                        overall_metrics[key].append(val)
                
    pd.DataFrame(overall_metrics).to_csv(
        os.path.join(save_prefix, f"metrics.csv"),
        index=False,
    )
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
        default="~/.cache/huggingface",
        help='Where to cache data.',
    )
    model_class = CLIOption(
        '--model-class', '-k',
        type=str,
        default=_MODEL_CLASS_DEFAULT,
        choices=list(_MODEL_CLASSES),
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
    output = CLIOption(
        '--output', '-o', 
        type=FileType('w'),
        default=sys.stdout,
        help='Output file. Default: STDOUT',
    )
    formatting = CLIOption(
        '--format', '-f', 
        type=str,
        default='TSV',
        choices=['TSV', 'CSV', 'tsv', 'csv'],
        help='Format of files. Default: %(default)s',
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