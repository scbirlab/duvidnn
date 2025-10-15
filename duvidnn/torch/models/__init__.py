from duvida.config import config
config.set_backend('torch', precision='float')

from .chemprop import ChempropEnsemble
from .cnn import TorchCNN2DEnsemble
from .mlp import TorchMLPEnsemble 