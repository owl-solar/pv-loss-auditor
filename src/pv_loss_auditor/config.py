"""Load and check per-system YAML configs (configs/system_<id>.yaml)."""

from pathlib import Path
from typing import Any

import yaml

from pv_loss_auditor.validation import require_keys

# Sections every pipeline stage relies on; stage-specific keys are checked where they are used.
REQUIRED_SECTIONS = (
    "system",
    "capacity",
    "inverters",
    "modules",
    "model",
    "kpi",
    "losses",
    "clean",
    "validation",
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config into a dict and check that the required sections are present.

    safe_load does not execute arbitrary YAML tags. Raises FileNotFoundError for a missing file and
    ValueError for an empty file or a file without the required sections.
    """
    with open(Path(path), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: config is empty or not a mapping")
    require_keys(config, REQUIRED_SECTIONS, where=str(path))
    return config
