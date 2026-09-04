"""Command-line interface for duvidnn."""

from carabiner.cliutils import (
    CLIApp,
    CLICommand,
    CLIOption,
)

from .. import (
    __version__,
    app_name,
)
from .predict import _predict
from .train import _train


def main() -> None:

    config_opt = CLIOption(
        "--config",
        "-c",
        required=True,
        type=str,
    )

    train = CLICommand(
        "train",
        description="Train a model from configuration.",
        options=[
            config_opt,
            CLIOption(
                "--training",
                "-1",
                required=True,
                type=str,
            ),
            CLIOption(
                "--validation",
                "-2",
                default=None,
                type=str,
            ),
            CLIOption(
                "--output",
                "-o",
                required=True,
                type=str,
            ),
        ],
        main=_train,
    )

    predict = CLICommand(
        "predict",
        description=(
            "Predict from a Box checkpoint."
        ),
        options=[
            CLIOption(
                "--checkpoint",
                "-k",
                required=True,
                type=str,
            ),
            CLIOption(
                "--data",
                "-d",
                required=True,
                type=str,
            ),
            config_opt,
            CLIOption(
                "--output",
                "-o",
                required=True,
                type=str,
            ),
            CLIOption(
                "--cache",
                default=None,
                type=str,
            ),
        ],
        main=_predict,
    )

    app = CLIApp(
        app_name,
        version=__version__,
        description=(
            "Train, checkpoint, predict, "
            "and quantify uncertainty for "
            "PyTorch models."
        ),
        commands=[
            train,
            predict,
        ],
    )

    app.run()


if __name__ == "__main__":
    main()
