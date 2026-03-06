from duvida.config import config
config.set_backend('torch', precision='float')

from .bilinear import TorchBilinearEnsemble
from .chemprop import ChempropEnsemble
from .cnn import TorchCNN2DEnsemble
from .lasso import TorchLassoEnsemble
from .mlp import TorchMLPEnsemble 