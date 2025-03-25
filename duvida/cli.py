"""Command-line interface for duvida."""

from argparse import FileType, Namespace
from collections import defaultdict
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
    load_data_args = {
        "filename": args.training, 
        "features": args.features, 
        "labels": args.labels,
        "cache": args.cache,
    }
    
    if args.config is not None:
        new_config = HyperOpt.from_file(args.config, silent=True)._ranges[args.config_index] 
        str_config.update(new_config)

    modelbox = AutoModelBox(
        **str_config,
    )._instance
    modelbox.load_training_data(**load_data_args)
    modelbox.model = modelbox.create_model()
    pprint_dict(
        str_config, 
        message=f"Model {modelbox.__class__.__name__} with {modelbox.size} parameters.",
    )

    save_prefix = os.path.join(
        args.prefix, 
        f"{modelbox.__class__.__name__}_x{modelbox.size}_epochs={args.epochs}",
    )
    training_args = {
        "val_filename": args.validation,
        "epochs": args.epochs, 
        "batch_size": args.batch,
    }
    pprint_dict(str_config, message=f">> Training {modelbox.__class__.__name__} with configuration")
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
        'training',
        type=str,
        help='Training dataset.',
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
        required=True,
        help='Column names from data file that contain features.',
    )
    label_cols = CLIOption(
        '--labels', '-y',
        type=str,
        nargs='*',
        required=True,
        help='Column names from data file that contain labels.',
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
        default=1e-4,
        help='Learning rate for training.',
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