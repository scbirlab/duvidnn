"""Initializing ModelBoxes automatically."""

from typing import Optional

import json
import os

from carabiner import print_err
from huggingface_hub import hf_hub_download

from .base_classes import ModelBoxBase
from .checkpoint_utils import load_checkpoint_file
try:
    from .torch.models import _MODEL_CLASS_DEFAULT, _MODEL_CLASSES
except ImportError as e:
    print_err(e)
    raise ImportError(
        """
        Modelling is not installed. Try reinstalling duvida with:

        $ pip install duvida[torch]

        or to use chemistry ML/AI:

        $ pip install duvida[chem]

        """
    )

class AutoModelBox:

    _config_file: str = "modelbox-config.json"
    _model_class_key: str = "model_class"

    def __init__(
        self,
        model_class: str = _MODEL_CLASS_DEFAULT,
        **kwargs
    ):  
        self._class = _MODEL_CLASSES[model_class]
        self._instance = self._class(**{
            key: val for key, val in kwargs.items() 
            if key != self._model_class_key
        })

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        cache_dir: Optional[str] = None,
        **kwargs
    ) -> ModelBoxBase:
        config_file = cls._config_file
        config = load_checkpoint_file(
            checkpoint=checkpoint,
            filename=config_file,
            callback="json",
            cache_dir=cache_dir,
            **kwargs
        )
        modelbox = cls(**config)._instance
        modelbox.load_checkpoint(checkpoint, cache_dir=cache_dir)
        return modelbox

