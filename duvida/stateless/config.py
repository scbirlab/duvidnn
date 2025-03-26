"""Backend configuration."""

from typing import Optional
from dataclasses import dataclass, field
import importlib
import os

from carabiner import print_err

DUVIDA_BACKEND = "DUVIDA_BACKEND"
DUVIDA_PRECISION = "DUVIDA_PRECISION"
if DUVIDA_BACKEND not in os.environ:
    os.environ[DUVIDA_BACKEND] = "jax"
print_err(f"Default duvida backend is: {os.environ[DUVIDA_BACKEND]}")
if DUVIDA_PRECISION not in os.environ:
    os.environ[DUVIDA_PRECISION] = "double"
print_err(f"Default duvida precision is: {os.environ[DUVIDA_PRECISION]}")

@dataclass
class Config:
    backend: str = os.environ[DUVIDA_BACKEND]
    precision: str = os.environ[DUVIDA_PRECISION]
    fallback: bool = True

    def __post_init__(self):
        self.backend_installed: bool = self._backend_installed()
        self.set_backend()

    def set_backend(
        self, 
        backend: Optional[str] = None, 
        precision: Optional[str] = None
    ) -> None:
        """Set the autodiff backend for duvida.

        Examples
        --------
        >>> os.environ[DUVIDA_BACKEND]
        'jax'
        >>> os.environ[DUVIDA_PRECISION]
        'double'
        >>> config.set_backend("jax"); config
        Config(backend='jax', precision='double', fallback=True)
        >>> config.set_backend("torch"); config
        Config(backend='torch', precision='double', fallback=True)
        
        """
        if backend is not None:
            self.backend = backend
        self._import_backend()
        if precision is not None:
            self.precision = precision
        self._set_precision()
        return None

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
            return config.update('jax_enable_x64', False)
        elif precision == 'half':
            # TODO: Find implementation of half precision in JAX. JMP?
            raise NotImplementedError("Half precision not available on JAX.")
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

    def _backend_installed(self) -> bool:
        return importlib.util.find_spec(self.backend) is not None

    def _import_backend(self) -> None:
        if self.backend == 'jax':
            importable = self._backend_installed()
            if not importable and self.fallback:
                self.backend == 'torch'
        
        if self.backend == 'torch':
            importable = self._backend_installed()
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