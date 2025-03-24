"""Backend configuration."""

from dataclasses import dataclass, field
import importlib
import os

DUVIDA_BACKEND = "DUVIDA_BACKEND"
DUVIDA_PRECISION = "DUVIDA_PRECISION"
if DUVIDA_BACKEND not in os.environ:
    os.environ[DUVIDA_BACKEND] = "jax"
if DUVIDA_PRECISION not in os.environ:
    os.environ[DUVIDA_PRECISION] = "double"

@dataclass
class Config:
    backend: str = os.environ[DUVIDA_BACKEND]
    precision: str = os.environ[DUVIDA_PRECISION]
    fallback: bool = True

    def __post_init__(self):
        self.backend_installed: bool = False
        self.set_backend(self.backend, precision=self.precision)

    def set_backend(self, backend: str = None, precision: str = None):
        if backend is not None:
            self.backend = backend
        self._import_backend()
        if precision is not None:
            self.precision = precision
        self._set_precision()
        return self.backend

    @staticmethod
    def _import_jax() -> bool:
        import jax
        return True

    @staticmethod
    def _import_torch() -> bool:
        import torch
        return True

    @staticmethod
    def _set_precision_jax(precision: str) -> None:
        from jax import config
        if precision == 'double':
            return config.update('jax_enable_x64', True)
        elif precision == 'float':
            return config.update('jax_enable_x32', True)
        elif precision == 'half':
            return config.update('jax_enable_x16', True)
        else:
            raise ValueError(f"Precision '{precision}' not valid.")

    @staticmethod
    def _set_precision_torch(precision: str) -> None:
        from torch import float64, float32, float16, set_default_dtype, set_default_device
        # set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
        if precision == 'double':
            return set_default_dtype(float64)
        elif precision == 'float':
            return set_default_dtype(float32)
        elif precision == 'half':
            return set_default_dtype(float16)
        else:
            raise ValueError(f"Precision '{precision}' not valid.")

    def _set_precision(self) -> None:
        if self.backend_installed:
            if self.backend == 'jax':
                precision_setter = self._set_precision_jax
            elif self.backend == 'torch':
                precision_setter = self._set_precision_torch
        else:
            raise AttributeError(f"Backend '{self.backend}' is not installed.")
        precision_setter(self.precision)
        os.environ[DUVIDA_PRECISION] = self.precision
        return None

    def _import_backend(self) -> None:
        if self.backend == 'jax':
            importable = importlib.util.find_spec("jax") is not None
            if not importable and self.fallback:
                self.backend == 'torch'
        
        if self.backend == 'torch':
            importable = importlib.util.find_spec("torch") is not None
        elif not importable:
            raise ValueError(f"Backend '{self.backend}' is not supported.")

        if importable:
            self.backend_installed = importable
        else:
            raise ImportError(f"Backend '{self.backend}' could not be imported."
                              f"Try reinstalling duvida with `pip install duvida[{self.backend}]`.")
        os.environ[DUVIDA_BACKEND] = self.backend
        return None

config = Config()