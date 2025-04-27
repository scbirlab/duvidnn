"""Pytest configuration."""

import sys
import pytest


@pytest.fixture(autouse=True)
def reload_package():
    global duvida    # reach the global scope
    import duvida    # reimport package every before test
    yield            # run test

    # delete all modules from package
    loaded_package_modules = [key for key, value in sys.modules.items() if "duvida" in str(value)]
    for key in loaded_package_modules:
        del sys.modules[key]
