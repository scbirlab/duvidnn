from duvida.config import config
config.set_backend('torch', precision='float')

from .modelboxes import (
    TorchCNN2DModelBox,
    TorchMLPModelBox,
    TorchFingerprintModelBox,
    ChempropModelBox
)