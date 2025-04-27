"""Automated registry of modelboxes."""

from typing import Callable, Dict, List

from .modelboxes import ModelBoxBase

# Registry to map function names to actual implementations
MODELBOX_REGISTRY: Dict[str, ModelBoxBase] = {}
MODELBOX_REGISTRY_INVERTED: Dict[ModelBoxBase, str] = {}
MODELBOX_NAMES: List[str] = []
DEFAULT_MODELBOX: str = "mlp"


def register_modelbox(name: str) -> Callable:
    """Decorator to register a function in the function registry.
    
    """
    def decorator(c: ModelBoxBase) -> ModelBoxBase:
        c.class_name = name           
        MODELBOX_REGISTRY[name] = c
        MODELBOX_REGISTRY_INVERTED[c] = name
        MODELBOX_NAMES.append(name)
        return c
        
    return decorator
