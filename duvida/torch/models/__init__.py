from ..chem import ChempropModelBox, FPMLPModelBox
from .mlp import MLPModelBox

_MODEL_CLASS_DEFAULT = "MLP"
_MODEL_CLASSES = {
    "mlp": MLPModelBox,
    "fingerprint": FPMLPModelBox,
    "fp": FPMLPModelBox,
    "chemprop": ChempropModelBox,
}

for _class in list(_MODEL_CLASSES.values()):
    _MODEL_CLASSES[_class.__name__] = _class