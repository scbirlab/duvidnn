"""Utilities for dealing with pytorch Lightning logs and outputs."""

from glob import glob
import os

def _get_most_recent_lightning_log(
    save_dir: str,
    filename: str,
    name: str = "lightning_logs",
    version_prefix: str = "version"
) -> str:
    path = os.path.join(save_dir, name)
    logs = sorted(glob(os.path.join(path, f"{version_prefix}_*")))
    max_version = max([int(v.split("_")[-1]) for v in logs])
    max_version = os.path.join(path, f"version_{max_version}", filename)
    if os.path.exists(max_version):
        return max_version
    else:
        raise OSError(f"Could not find most recent log: {max_version}")
