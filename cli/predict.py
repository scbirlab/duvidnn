"""Predict from a Box checkpoint."""

from argparse import Namespace

from carabiner.cliutils import clicommand

from ..box import Box
from ..checkpoint_utils import load_json
from ..config import instantiate_uncertainty
from .io import _save_dataset


@clicommand(message="Predicting")
def _predict(args: Namespace) -> None:

    box = Box.load(
        args.checkpoint,
        cache_dir=args.cache,
    )

    config = (
        load_json(args.config)
        if args.config is not None
        else {}
    )

    uncertainty = instantiate_uncertainty(
        config.get("uncertainty")
    )

    result = box.predict(
        args.data,
        batch_size=config.get(
            "batch_size",
            32,
        ),
        uncertainty=uncertainty,
        device=config.get(
            "device"
        ),
    )

    _save_dataset(
        result,
        args.output,
    )
