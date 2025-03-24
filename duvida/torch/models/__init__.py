from ..chem import ChempropModelBox, FPMLPModelBox
from .mlp import MLPModelBox

_MODEL_CLASS_DEFAULT = "MLP"
_MODEL_CLASSES = {
    "mlp": MLPModelBox,
    "fingerprint": FPMLPModelBox,
    "fp": FPMLPModelBox,
    "chemprop": ChempropModelBox,
}