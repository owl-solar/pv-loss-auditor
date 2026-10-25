"""Daily series for RdTools (degradation, soiling) from the 9069 clean data.

Two candidate series, both over RUNNING inverters only (stopped inverters are downtime, which the
waterfall already counts; letting them in would be read by RdTools as a performance drop):
- normalized energy: measured AC / PVWatts expected AC (model.expected_power, incl. clipping);
  a ratio of sums over the inverter-steps that are usable and not down;
- PR (C): the weather-corrected PR of Phase 2, variant C, excluding downtime.
A day is kept only if enough inverter-steps are usable (kpi.day_min_coverage), enough of them
are running (a few survivors make a noisy ratio) and the daily
plane-of-array insolation is high enough for a stable ratio (degradation.min_daily_poa_kwh_m2).
"""

import pandas as pd

from src.kpi import _period, inverter_states, kpi_table, step_hours, weighted_t_ref
from src.model import expected_power, weighted_gamma


def daily_normalized_energy(inputs, ac, t_cell, config, gamma=None):
    """Daily measured / expected energy of the running inverters (dimensionless, ~0.9).

    Returns a DataFrame indexed by local date with energy_norm, poa_kwh_m2 and coverage
    (usable inverter-steps / daylight inverter-steps; stopped inverters are usable,
    they are downtime, not missing data) and running_share (usable steps not down).
    """
    dt = step_hours(inputs.index)
    poa = inputs["poa_wm2"]
    expected = expected_power(poa, t_cell, config, gamma)["p_ac_kw"]   # same for every inverter

    base = inputs["daylight"] & poa.notna() & t_cell.notna()
    valid = ac.notna() & base.to_numpy()[:, None]
    _, down = inverter_states(ac, poa, config)
    up = valid & ~down

    steps = pd.DataFrame({
        "e_meas_kwh": ac.where(up).sum(axis=1) * dt,
        "e_exp_kwh": expected * up.sum(axis=1) * dt,
        "n_up": up.sum(axis=1),
        "n_valid": valid.sum(axis=1),
        "n_daylight": inputs["daylight"].astype(int) * ac.shape[1],
        "poa_kwh_m2": poa.where(base) * dt / 1000,
    })
    day = steps.groupby(_period(inputs.index, "D")).sum()
    day = day[day["n_daylight"] > 0]
    out = pd.DataFrame(index=day.index)
    out["energy_norm"] = day["e_meas_kwh"] / day["e_exp_kwh"]
    out["poa_kwh_m2"] = day["poa_kwh_m2"]
    out["coverage"] = day["n_valid"] / day["n_daylight"]          # usable, running or not
    out["running_share"] = day["n_up"] / day["n_valid"]           # share not down
    return out


def daily_series(inputs, ac, t_cell, config, gamma=None):
    """Both RdTools candidates side by side, after the coverage and insolation filters.

    Columns: energy_norm, pr_wc (excl. downtime), poa_kwh_m2, coverage. Index = local date
    (daily DatetimeIndex, NaN on rejected days) so RdTools sees a regular daily grid.
    """
    gamma = weighted_gamma(config) if gamma is None else gamma
    norm = daily_normalized_energy(inputs, ac, t_cell, config, gamma)

    base = inputs["daylight"] & inputs["poa_wm2"].notna() & t_cell.notna()
    t_ref = weighted_t_ref(t_cell, inputs["poa_wm2"], base)
    pr = kpi_table(inputs, ac, t_cell, config, gamma, t_ref, freq="D")["pr_wc_excl_down"]

    out = norm.join(pr.rename("pr_wc"))
    keep = ((out["coverage"] >= config["kpi"]["day_min_coverage"])
            & (out["running_share"] >= config["degradation"]["min_running_share"])
            & (out["poa_kwh_m2"] >= config["degradation"]["min_daily_poa_kwh_m2"]))
    out.loc[~keep, ["energy_norm", "pr_wc"]] = float("nan")
    out.index = out.index.to_timestamp()                 # Period -> naive local midnight
    return out.asfreq("D")


def yoy_rate(series, uncertainty_method="simple"):
    """RdTools year-on-year degradation in %/year with its confidence interval.

    The YoY method compares each day with the same day one year later (so season cancels), then
    takes the median of all pairs: robust against outlier days. Returns (rd, ci_low, ci_high).
    """
    from rdtools.degradation import degradation_year_on_year
    rd, ci, _ = degradation_year_on_year(series.dropna(), recenter=True,
                                         uncertainty_method=uncertainty_method)
    return rd, ci[0], ci[1]


def filter_sensitivity(inputs, ac, t_cell, config, poa_grid, running_grid, end="2022-12-31"):
    """YoY rate of normalized energy for a grid of day-filter thresholds, before and after `end`.

    Shows whether the degradation number depends on the (assumed) filters.
    """
    rows = []
    for poa_min in poa_grid:
        for run_min in running_grid:
            cfg = {**config, "degradation": {"min_daily_poa_kwh_m2": poa_min,
                                             "min_running_share": run_min}}
            s = daily_series(inputs, ac, t_cell, cfg)["energy_norm"]
            for label, part in (("to " + end, s[:end]), ("full", s)):
                rd, lo, hi = yoy_rate(part)
                rows.append({"min_poa": poa_min, "min_running": run_min, "period": label,
                             "days": int(part.notna().sum()), "rd_pct_yr": rd, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)
