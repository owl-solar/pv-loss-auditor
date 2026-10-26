"""Input checks shared by the model, KPI and loss functions.

Each check raises a ValueError with a message that names the argument and the offending value,
instead of letting a nonsensical input produce a plausible-looking number. The thresholds come from
the `validation:` section of the config; nothing here changes a result for valid data.
"""

import logging
from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def require_keys(config: Mapping[str, Any], keys: Sequence[str], where: str = "config") -> None:
    """Raise ValueError naming every key of `keys` that is missing in `config`."""
    missing = [k for k in keys if k not in config]
    if missing:
        raise ValueError(f"{where}: missing required section(s) {missing}")


def require_positive(value: float, name: str) -> float:
    """Return `value` if it is a finite number above zero (rated power, capacity, count)."""
    if not (value == value and value > 0):          # NaN fails both comparisons
        raise ValueError(f"{name} must be positive, got {value!r}")
    return value


def require_nonempty(obj: pd.DataFrame | pd.Series, name: str) -> None:
    """Raise ValueError for an empty period (no rows), which would give 0 / 0 in every ratio."""
    if len(obj) == 0:
        raise ValueError(f"{name} is empty: no rows in the selected period")


def require_aligned(reference: pd.DataFrame | pd.Series, other: pd.DataFrame | pd.Series,
                    names: tuple[str, str]) -> None:
    """The two objects must share one time index, otherwise sums would mix different steps."""
    if not reference.index.equals(other.index):
        raise ValueError(f"{names[0]} and {names[1]} have different time indices")


def require_choice(value: Hashable, allowed: Sequence[Hashable], name: str) -> None:
    """Raise ValueError unless `value` is one of `allowed`."""
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, got {value!r}")


def check_poa(poa: pd.Series, config: Mapping[str, Any]) -> None:
    """POA irradiance cannot be negative. Small negative values (sensor offset) are tolerated.

    The tolerance is validation.poa_min_wm2 (-5 W/m2, as in the cleaning); anything below means a
    broken sensor or a wrong column, so the audit stops instead of computing with it.
    """
    limit = config["validation"]["poa_min_wm2"]
    lowest = poa.min()
    if pd.notna(lowest) and lowest < limit:
        raise ValueError(f"POA irradiance has negative values: minimum {lowest:.1f} W/m2 "
                         f"(allowed down to {limit} W/m2)")


def warn_suspicious(inputs: pd.DataFrame, config: Mapping[str, Any]) -> None:
    """Log warnings (no exception) for data that is usable but suspicious.

    - a hole in the time index longer than validation.max_gap_hours;
    - a daylight sensor series with a single value (flat / frozen sensor);
    - POA missing on more than validation.max_missing_share of the daylight steps.
    """
    v = config["validation"]
    steps = inputs.index.to_series().diff()
    long_gaps = steps[steps > pd.Timedelta(hours=v["max_gap_hours"])]
    for end, length in long_gaps.items():
        log.warning("gap of %s before %s in the model inputs", length, end)

    day = inputs["daylight"]
    for column in ("poa_wm2", "t_mod_c", "p_ac_meter_kw"):
        values = inputs.loc[day, column].dropna()
        if len(values) > 1 and values.nunique() == 1:
            log.warning("flat sensor: %s has the single value %s on all %d valid daylight steps",
                        column, values.iloc[0], len(values))
    if day.any():
        missing = inputs.loc[day, "poa_wm2"].isna().mean()
        if missing > v["max_missing_share"]:
            log.warning("POA is missing on %.1f %% of the daylight steps", 100 * missing)
