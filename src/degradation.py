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

import numpy as np
import pandas as pd

from src.kpi import _period, inverter_states, kpi_table, step_hours, weighted_t_ref
from src.model import expected_power, inverter_dc_kw, weighted_gamma


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


def yoy_rate(series, uncertainty_method="simple", seed=1):
    """RdTools year-on-year degradation in %/year with its confidence interval.

    The YoY method compares each day with the same day one year later (so season cancels), then
    takes the median of all pairs: robust against outlier days. Returns (rd, ci_low, ci_high).
    The interval is bootstrapped with numpy's global random state, so the seed is fixed (as in
    soiling_run): the rate itself does not depend on it, the interval moves by up to ~0.04 %/yr.
    """
    from rdtools.degradation import degradation_year_on_year
    np.random.seed(seed)
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


def daily_precipitation(weather):
    """Daily rain (mm) per local day from hourly Open-Meteo data, on the same index as daily_series."""
    local = weather["precipitation"].copy()
    local.index = local.index.tz_convert("America/New_York").tz_localize(None)
    return local.resample("D").sum()


def soiling_run(daily, precip=None, precip_threshold=2.0, reps=200, seed=1):
    """RdTools SRR soiling ratio of normalized energy; returns (ratio, ci_low, ci_high, n_valid, n_all).

    Without precip, cleaning events are inferred from upward jumps of the series ('shift');
    with precip, a day with rain >= precip_threshold mm also counts as a cleaning
    ('precip_and_shift'). The ratio is the median soiling ratio over the period, so
    1 - ratio is the average energy lost to soiling. Random seed fixed: SRR bootstraps.
    """
    import numpy as np
    from rdtools.soiling import soiling_srr

    np.random.seed(seed)
    kwargs = {}
    if precip is not None:
        kwargs = {"precipitation_daily": precip.reindex(daily.index).fillna(0),
                  "precip_threshold": precip_threshold, "clean_criterion": "precip_and_shift"}
    ratio, ci, info = soiling_srr(daily["energy_norm"], daily["poa_kwh_m2"].fillna(0),
                                  reps=reps, **kwargs)
    table = info["soiling_interval_summary"]
    return ratio, ci[0], ci[1], int(table["valid"].sum()), len(table)


def pr_by_poa_source(inputs, ac, config, external, min_valid=10):
    """Annual raw PR (variant A, no temperature factor) of running inverters per POA source.

    PR = sum(E_AC) / sum(P0 x H_poa / 1000), same hours and inverters for every source. Hours are
    the preceding-hour windows used by Open-Meteo; an hour counts only if the plant POA has at
    least `min_valid` of 12 valid 5-min values. Temperature correction is left out on purpose: the
    reanalysis gives air, not cell temperature, so the comparison isolates the irradiance source.
    `external`: dict name -> hourly POA Series (W/m2, UTC, hour-end labels).
    """
    poa = inputs["poa_wm2"]
    _, down = inverter_states(ac, poa, config)
    up = ac.notna() & ~down & inputs["daylight"].to_numpy()[:, None]
    n_up = up.sum(axis=1)
    e_kwh = ac.where(up).sum(axis=1) / 12                  # 5-min steps -> kWh

    hourly = pd.DataFrame({"e_kwh": e_kwh, "n_up": n_up, "plant": poa}).resample(
        "1h", closed="right", label="right")
    h = pd.DataFrame({"e_kwh": hourly["e_kwh"].sum(), "n_up": hourly["n_up"].sum() / 12,
                      "plant": hourly["plant"].mean()})
    h = h.where(hourly["plant"].count() >= min_valid)
    p0 = inverter_dc_kw(config)
    cols = {"plant": h["plant"], **{k: v.reindex(h.index) for k, v in external.items()}}
    rows = {}
    for name, series in cols.items():
        ok = h["e_kwh"].notna() & series.notna() & (h["n_up"] > 0)
        year = h.index[ok].tz_convert("America/New_York").year
        energy = h["e_kwh"][ok].groupby(year).sum()
        # n_up / 12 is the number of inverter-hours (n_up counted per 5-min step)
        reference = (p0 * series[ok] / 1000 * h["n_up"][ok]).groupby(year).sum()
        rows[name] = energy / reference
    return pd.DataFrame(rows)


def yoy_table(daily, end="2022-12-31"):
    """YoY degradation of both RdTools candidates, for the full period and up to `end`.

    The step in 2023 is not ageing, so both periods are reported side by side.
    """
    rows = []
    for col in ("energy_norm", "pr_wc"):
        for label, s in (("full", daily[col]), ("to " + end[:7], daily[col][:end])):
            rd, lo, hi = yoy_rate(s)
            rows.append({"series": col, "period": label, "days": int(s.notna().sum()),
                         "rd_pct_yr": rd, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def soiling_table(daily, rain, thresholds=(1, 2, 5, 10), end="2022-12-31"):
    """SRR soiling ratio for different cleaning information, loss_pct = (1 - ratio) x 100.

    Rows: upward jumps only, rain >= each threshold (mm) + jumps, and the 2 mm case up to `end`.
    The 2 mm case on the full period is the one used in the shortfall split.
    """
    names = ["ratio", "ci_low", "ci_high", "valid", "intervals"]
    rows = [{"cleaning_info": "upward jumps only", "period": "full",
             **dict(zip(names, soiling_run(daily)))}]
    for th in thresholds:
        rows.append({"cleaning_info": f"rain >= {th} mm + jumps", "period": "full",
                     **dict(zip(names, soiling_run(daily, rain, th)))})
    rows.append({"cleaning_info": "rain >= 2 mm + jumps", "period": "to " + end[:7],
                 **dict(zip(names, soiling_run(daily[:end], rain, 2.0)))})
    table = pd.DataFrame(rows)
    table["loss_pct"] = (1 - table["ratio"]) * 100
    return table


def hourly_plant_poa(poa, min_valid=10):
    """Plant POA as hourly means of the PRECEDING hour (labels at hour end, as Open-Meteo).

    An hour counts only if at least `min_valid` of its 12 five-minute values exist.
    """
    h = poa.resample("1h", closed="right", label="right")
    return h.mean().where(h.count() >= min_valid)


def external_poa_comparison(plant_hourly, external, min_poa=200, window=12):
    """External hourly POA against the plant sensors.

    Returns (stats, ratio):
    - stats: per source, on hours with plant POA above `min_poa`: MBE and nRMSE in % of the mean
      plant POA, and the correlation r;
    - ratio: monthly external / plant insolation (sums over hours where both exist) and its
      `window`-month rolling mean (columns <source>_12m, NaN at the start). A flat ratio means no
      sensor drift. Index = naive month start.
    """
    stats, monthly = [], {}
    for name, series in external.items():
        both = pd.concat([plant_hourly.rename("p"), series.rename("e")], axis=1).dropna()
        sun = both[both.p > min_poa]
        stats.append({"source": name, "hours": len(sun),
                      "mbe_pct": (sun.e - sun.p).mean() / sun.p.mean() * 100,
                      "nrmse_pct": np.sqrt(((sun.e - sun.p) ** 2).mean()) / sun.p.mean() * 100,
                      "r": sun.e.corr(sun.p)})
        m = both.resample("MS").sum()
        m = m[m.p > 0]
        monthly[name] = m.e / m.p
    ratio = pd.DataFrame(monthly)
    ratio.index = ratio.index.tz_localize(None)
    ratio = ratio.join(ratio.rolling(window).mean().add_suffix(f"_{window}m"))
    return pd.DataFrame(stats), ratio


def weighted_loss_pct(poa, loss_fraction):
    """Period loss in % of POA: sum(poa x loss) / sum(poa). Used for the AOI loss."""
    return float((poa * loss_fraction).sum() / poa.sum() * 100)


def shortfall_split(daily, soiling_pct, aoi_pct):
    """Yearly shortfall (median of 1 - energy_norm, % of expected) and what explains it.

    Soiling is an estimate, AOI an upper bound (the reference cells lose to reflection too), so
    the unexplained part is a range: min subtracts both, max only soiling.
    """
    split = pd.DataFrame({"shortfall_pct": (1 - daily["energy_norm"]).groupby(daily.index.year).median() * 100})
    split["soiling_pct"] = soiling_pct
    split["aoi_upper_bound_pct"] = aoi_pct
    split["unexplained_min_pct"] = split["shortfall_pct"] - soiling_pct - aoi_pct
    split["unexplained_max_pct"] = split["shortfall_pct"] - soiling_pct
    return split
