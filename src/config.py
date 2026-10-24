"""Load per-system YAML configs (configs/system_<id>.yaml)."""

from pathlib import Path

import yaml


def load_config(path):
    """Read a YAML config into a dict. safe_load does not execute arbitrary YAML tags."""
    with open(Path(path), encoding="utf-8") as f:
        return yaml.safe_load(f)
