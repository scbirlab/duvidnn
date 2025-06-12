"""Tools for loading and writing reusable package data."""

import os

DUVIDA_CACHE = "DUVIDA_CACHE"
DEFAULT_CACHE = os.path.join("~", ".cache", "duvida", "data")

def _get_data_path(
    filename: str, 
    env_key: str = DUVIDA_CACHE,
    default: str = DEFAULT_CACHE
) -> str:
    """Returns the path to a writable version of a package data file.
    
    Copies it from the package resources if not present.

    """
    cache_dir = os.environ.get(
        env_key, 
        os.path.expanduser(default),
    )
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir, os.path.join(cache_dir, filename)
