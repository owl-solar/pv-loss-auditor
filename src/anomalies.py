"""Residuals of each inverter against the model and against its peers (Phase 5).

r = measured AC / expected AC is ~0.9 for a healthy inverter (the model is not calibrated, see
model.py). Plant-wide effects (season, soiling, the 2023 step, sensor bias) move r of ALL
inverters together, so r is divided by the median r of the running inverters at the same step:

    rel = r_i / median_j(r_j)        ~1.0 for a healthy inverter, 0.97 if it lost 3 % to its peers.

This removes everything the inverters share and keeps what is specific to one inverter, which is
what a failed string or combiner does. The price: a loss that hits every inverter equally is
invisible here (it is covered by the waterfall and Phase 4).
"""

import numpy as np
import pandas as pd

from src.kpi import inverter_states
from src.model import expected_power


def relative_residual(inputs, ac, t_cell, config):
    """Peer-normalised residual rel per inverter-step (DataFrame like `ac`, NaN = not usable).

    A step counts if it is daylight, POA > min_poa_wm2, T_cell known and the model is below
    max_expected_fraction x rating (no clipping). Stopped inverters (down) are excluded: they are
    downtime, already in the waterfall. The peer median needs at least min_peers inverters.
    """
    a = config["anomalies"]
    rating = config["inverters"]["ac_kw"][0]["ac_kw"]
    poa = inputs["poa_wm2"]
    expected = expected_power(poa, t_cell, config)["p_ac_kw"]
    step_ok = (inputs["daylight"] & (poa > a["min_poa_wm2"]) & t_cell.notna()
               & (expected < a["max_expected_fraction"] * rating))
    _, down = inverter_states(ac, poa, config)
    usable = ac.notna() & ~down & step_ok.to_numpy()[:, None]

    r = ac.div(expected, axis=0).where(usable)
    peer = r.median(axis=1).where(r.notna().sum(axis=1) >= a["min_peers"])
    return r.div(peer, axis=0)


def daily_features(rel, config):
    """One row per (local date, inverter) with summary features of rel.

    rel_median   typical level of the day (a lost string shifts it);
    rel_p10      lower tail (partial-day problems);
    frac_below95 share of steps with rel < 0.95 (persistent vs short);
    rel_spread   p90 - p10 (unstable behaviour);
    n_steps      number of valid steps; rows below min_day_steps are dropped.
    """
    day = rel.index.tz_convert("America/New_York").normalize().tz_localize(None)
    long = rel.stack().rename("rel").reset_index()
    long.columns = ["time", "inverter", "rel"]
    long["date"] = day[rel.index.get_indexer(long["time"])]
    g = long.groupby(["date", "inverter"])["rel"]
    out = pd.DataFrame({
        "rel_median": g.median(),
        "rel_p10": g.quantile(0.10),
        "frac_below95": g.apply(lambda s: (s < 0.95).mean()),
        "rel_spread": g.quantile(0.90) - g.quantile(0.10),
        "n_steps": g.size(),
    })
    return out[out["n_steps"] >= config["anomalies"]["min_day_steps"]]


def deviation_from_own_norm(daily_rel_median, window_days=61, min_days=20):
    """Daily rel_median minus the inverter's own rolling median (centred, +-30 days).

    Healthy inverters sit at fixed offsets from their peers (0.96 ... 1.05, stable over years:
    different mismatch, shading, sensor position). Those offsets are not faults, so the detector
    looks at the change against the inverter's own recent level. Input: DataFrame, rows =
    local dates, columns = inverters (as `daily_features(...)["rel_median"].unstack()`).
    Limit: a slow real degradation of one inverter is absorbed into its norm.
    """
    daily = daily_rel_median.asfreq("D")           # regular grid, so 61 rows = 61 days
    norm = daily.rolling(window_days, center=True, min_periods=min_days).median()
    return daily - norm
