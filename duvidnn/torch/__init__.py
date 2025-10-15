from duvida.config import config
config.set_backend('torch', precision='float')

from .models import (
    ChempropEnsemble,
    TorchMLPEnsemble
)
from .modelbox import (
    ChempropModelBox,
    TorchFingerprintModelBox,
    TorchMLPModelBox
)
