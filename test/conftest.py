"""Pytest configuration."""

import importlib
import sys

import pytest
from duvida.stateless import config, hessians, hvp, information

MODULES_TO_RELOAD = {
    "duvida.stateless.hessians": hessians,
    "duvida.stateless.hessians": hvp,
    "duvida.stateless.information": information,
    "duvida.stateless.config": config,
}

@pytest.fixture(autouse=True)
def conditional_reload_my_module(request):
    """Reload my_module only for modules depending on global state"""
    if request.module.__name__ in MODULES_TO_RELOAD:
        if "torch" in sys.modules:
            del sys.modules["torch"]