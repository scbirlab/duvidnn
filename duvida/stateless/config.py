"""Backend configuration."""

from typing import Optional
from dataclasses import dataclass, field
import importlib
import logging
import os

_BACKEND_FLAG: str = "DUVIDA_BACKEND"
_PRECISION_FLAG: str = "DUVIDA_PRECISION"

log = logging.getLogger("duvida.config")


@dataclass
class Config:
    backend: Optional[str] = field(default=None)
    precision: Optional[str] = field(default=None)

    def __post_init__(self):
        self.backend = self.backend or os.getenv(_BACKEND_FLAG, "jax")
        self.precision = self.precision or os.getenv(_PRECISION_FLAG, "double")
        self.backend_installed: bool = self._backend_installed()
        # self.set_backend()

    def set_backend(
        self, 
        backend: Optional[str] = None, 
        precision: Optional[str] = None
    ) -> None:
        """Set the autodiff backend for duvida.

        Examples
        --------
        >>> config.set_backend("jax", precision="double"); config
        Config(backend='jax', precision='double')
        >>> config.set_backend("torch", precision="float"); config
        Config(backend='torch', precision='float')
        
        """
        if backend is not None:
            self.backend = backend
        if precision is not None:
            self.precision = precision
        self._import_backend()
        self._set_precision()
        return None

    @staticmethod
    def _set_precision_jax(precision: str) -> None:
        from jax import config as jax_config
        if precision == 'double':
            return jax_config.update('jax_enable_x64', True)
        elif precision == 'float':
            return jax_config.update('jax_enable_x64', False)
        elif precision == 'half':
            # TODO: Find implementation of half precision in JAX. JMP?
            raise NotImplementedError("Half precision not available on JAX.")
        else:
            raise ValueError(f"Precision '{precision}' not valid.")
        log.debug(f"JAX precision set to {precision}")

    @staticmethod
    def _set_precision_torch(precision: str) -> None:
        from torch import set_default_dtype  # , set_default_device
        # set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
        if precision == 'double':
            from torch import float64
            return set_default_dtype(float64)
        elif precision == 'float':
            from torch import float32
            return set_default_dtype(float32)
        elif precision == 'half':
            from torch import float16
            return set_default_dtype(float16)
        else:
            raise ValueError(f"Precision '{precision}' not valid.")
        log.debug(f"torch precision set to {precision}")

    def _set_precision(self) -> None:
        if self.backend_installed:
            if self.backend == 'jax':
                precision_setter = self._set_precision_jax
            elif self.backend == 'torch':
                precision_setter = self._set_precision_torch
        else:
            raise AttributeError(f"Backend '{self.backend}' is not installed.")
        precision_setter(self.precision)
        os.environ[_PRECISION_FLAG] = self.precision
        return None

    def _backend_installed(self) -> bool:
        return importlib.util.find_spec(self.backend) is not None

    def _import_backend(self) -> None:
        if self.backend in ('jax', 'torch'):
            importable = self._backend_installed()
        else:
            raise AttributeError(f"Backend '{self.backend}' is not supported.")

        if importable:
            self.backend_installed = importable
        else:
            raise ImportError(
                f"""
                Backend '{self.backend}' could not be imported.
                Try reinstalling duvida with:
                    pip install duvida[{self.backend}]
                or install backend with:
                    pip install {self.backend}
                """
            )
        os.environ[_BACKEND_FLAG] = self.backend
        return None


config = Config()
