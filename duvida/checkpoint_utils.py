""""Utilities for loading and aving checkpoints."""

from typing import Any, Callable, Dict, Union
from tempfile import TemporaryDirectory
import os
import json

from carabiner import print_err
from datasets import Dataset, IterableDataset, load_from_disk
from huggingface_hub import hf_hub_download
import torch

def _load_json(checkpoint, filename) -> Dict[str, Any]:
    with open(os.path.join(checkpoint, filename), "r") as f:
            obj = json.load(f)
    return obj

def _load_hf_dataset(checkpoint, filename) -> Union[Dataset, IterableDataset]:
    return load_from_disk(os.path.join(checkpoint, filename))

def _load_torch_weights(checkpoint, filename):
    return torch.load(
        os.path.join(checkpoint, filename),
        weights_only=True,
    )

FILE_LOADING_CALLBACKS = {
    "json": _load_json,
    "hf-dataset": _load_hf_dataset, 
    "pt": _load_torch_weights,
}

def load_checkpoint_file(
    checkpoint: str,
    filename: str,
    callback: Union[str, Callable] = "json",
    none_on_error: bool = False,
    *args, **kwargs
) -> Union[Any, None]:
    if isinstance(callback, str):
        if callback in FILE_LOADING_CALLBACKS:
            callback = FILE_LOADING_CALLBACKS[callback.casefold()]
        else:
            raise ValueError(
                """
                File loading callback must be callable or name.
                """
            )
    if os.path.exists(checkpoint):
        obj = callback(checkpoint, filename)
    elif checkpoint.startswith("hf://"):
        checkpoint = checkpoint.split("hf://")[-1]
        with TemporaryDirectory() as tmpdirname:
            try:
                hf_hub_download(
                    repo_id=checkpoint,
                    filename=filename,
                    local_dir=tmpdirname,
                    *args, **kwargs
                )
            except Exception as e:
                print_err(e)
                if none_on_error:
                    return None
                else:
                    raise e
            else:
                obj = callback(tmpdirname, filename)

    return obj