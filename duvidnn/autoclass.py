"""Initializing ModelBoxes automatically."""

from typing import Optional

from carabiner import print_err

from .checkpoint_utils import load_checkpoint_file
try:
    from .torch.modelbox import *
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
else:
    from .base.modelboxes import ModelBoxBase
    from .base.modelbox_registry import DEFAULT_MODELBOX, MODELBOX_REGISTRY

class AutoModelBox:

    _init_kwargs_file: str = ModelBoxBase._init_kwargs_filename
    _model_config_file: str = ModelBoxBase._model_config_filename
    _model_class_key: str = "class_name"

    def __init__(
        self,
        class_name: str = DEFAULT_MODELBOX,
        **kwargs
    ):  
        self._class = MODELBOX_REGISTRY[class_name]
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
        config_file = cls._init_kwargs_file
        config = load_checkpoint_file(
            checkpoint=checkpoint,
            filename=config_file,
            callback="json",
            cache_dir=cache_dir,
            **kwargs
        )
        modelbox = cls(**config)._instance
        modelbox.load_checkpoint(checkpoint, cache_dir=cache_dir)
        modelbox._default_cache = cache_dir
        return modelbox

    @classmethod
    def show(cls):
        return tuple(MODELBOX_REGISTRY)
