"""SYNTHETIC string faults for testing the anomaly detector (Phase 5).

Real data have no fault labels, so faults are injected into the real AC series: a fault of k
lost strings multiplies the AC power of one inverter by (1 - k / N), N = strings per inverter
(config `anomalies.synthetic`). Everything here is synthetic and must be labelled so wherever it
is shown. Real measurements are never changed on disk: `inject_faults` returns a modified copy.

Simplification: the loss is applied to AC power, not to DC before the inverter curve. Away from
clipping the inverter efficiency is nearly constant, and the residuals use only unclipped steps.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from pv_loss_auditor.validation import require_nonempty

LOCAL_TZ = "America/New_York"


def _blocked_days(
    index: pd.DatetimeIndex, inverters: Sequence[str], config: Mapping[str, Any]
) -> pd.DataFrame:
    """Boolean frame (local dates x inverters): True = day inside an excluded real event."""
    s = config["anomalies"]["synthetic"]
    blocked = pd.DataFrame(False, index=index, columns=inverters)
    for e in s["exclude_plant"]:
        blocked.loc[e["start"] : e["end"], :] = True
    for e in s["exclude_inverter"]:
        if e["inverter"] in blocked:
            blocked.loc[e["start"] : e["end"], e["inverter"]] = True
    return blocked


def eligible_starts(valid_days: pd.DataFrame, config: Mapping[str, Any], window_days: int) -> pd.DataFrame:
    """True where a fault window may start (rows: local dates, columns: inverters).

    valid_days: boolean frame, True = the inverter has a daily feature on that day. A start is
    eligible if enough window days are valid and no excluded real event lies within the window
    or within buffer_days of it.
    """
    s = config["anomalies"]["synthetic"]
    w, b = window_days, s["buffer_days"]
    valid = valid_days.asfreq("D").fillna(False).astype(float)
    # Mean of validity over [t, t + w - 1]: a trailing rolling mean, shifted back to the start day.
    share = valid.rolling(w, min_periods=w).mean().shift(-(w - 1))
    blocked = _blocked_days(valid.index, valid.columns, config).astype(float)
    # Any excluded day in [t - b, t + w - 1 + b]: rolling max of length w + 2b, shifted back.
    near_block = blocked.rolling(w + 2 * b, min_periods=1).max().shift(-(w - 1 + b)).fillna(1.0)
    return (share >= s["min_valid_share"]) & (near_block == 0)


def plan_faults(valid_days: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Random fault events: inverter, start, end, duration_days, strings, loss_fraction, is_synthetic.

    For every duration and every level, events_per_level windows are drawn from the eligible
    starts with a fixed seed. Two windows of the same inverter never overlap (nor touch).
    """
    require_nonempty(valid_days, "valid_days")
    s = config["anomalies"]["synthetic"]
    rng = np.random.default_rng(s["seed"])
    taken: dict[str, list[tuple[int, int]]] = {
        c: [] for c in valid_days.columns
    }  # accepted (first, last) row per inverter
    rows = []
    for w in s["durations_days"]:
        ok = eligible_starts(valid_days, config, w)
        dates, cols = ok.index, list(ok.columns)
        cand = np.argwhere(ok.to_numpy())  # (row, col) of every eligible start
        for k in s["levels_strings"]:
            n_done = 0
            for i in rng.permutation(len(cand)):
                r, c = cand[i]
                name = cols[c]
                if any(r <= last + 1 and r + w - 1 >= first - 1 for first, last in taken[name]):
                    continue
                taken[name].append((r, r + w - 1))
                rows.append(
                    {
                        "inverter": name,
                        "start": dates[r],
                        "end": dates[r + w - 1],
                        "duration_days": w,
                        "strings": k,
                        "loss_fraction": k / s["strings_per_inverter"],
                        "is_synthetic": True,
                    }
                )
                n_done += 1
                if n_done == s["events_per_level"]:
                    break
            if n_done < s["events_per_level"]:
                raise ValueError(f"only {n_done} windows of {w} days for {k} strings: relax config")
    return pd.DataFrame(rows)


def inject_faults(ac: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Copy of `ac` (kW, columns = inverters) with the events applied. NaN stays NaN."""
    unknown = sorted(set(events["inverter"]) - set(ac.columns))
    if unknown:
        raise ValueError(f"fault events name inverters that are not in ac: {unknown}")
    if ((events["loss_fraction"] < 0) | (events["loss_fraction"] > 1)).any():
        raise ValueError("loss_fraction of a fault must be between 0 and 1")
    out = ac.copy()
    local_day = ac.index.tz_convert(LOCAL_TZ).normalize().tz_localize(None)
    for e in events.itertuples():
        in_window = (local_day >= e.start) & (local_day <= e.end)
        out.loc[in_window, e.inverter] *= 1.0 - e.loss_fraction
    return out
