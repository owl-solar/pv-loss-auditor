"""Plant KPIs: performance ratio (three temperature treatments), specific yield, availability.

PR follows IEC 61724-1: PR = sum(E_AC) / sum(P0 x G/1000 x dt), optionally with the temperature
factor [1 + gamma x (T_cell - T_ref)] in the denominator:
- pr_raw : no temperature factor (A);
- pr_stc : T_ref = 25 C (B, IEC "temperature-corrected PR");
- pr_wc  : T_ref = POA-weighted mean cell temperature of the whole period (C, NREL
           "weather-corrected PR", Dierauf et al. 2013): no seasonal swing, familiar scale.
PR is always a ratio of sums over the same mask, never a mean of 5-min ratios.

Inverter-step accounting: every inverter has the same P0, so each inverter at each 5-min step
contributes P0_inv x G/1000 x dt to the denominator only where that inverter's AC is known.
A gap in one inverter's data removes only that inverter-step, not the whole plant step.
"""

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from pv_loss_auditor.geometry import LOCAL_TZ
from pv_loss_auditor.model import inverter_dc_kw
from pv_loss_auditor.validation import (
    check_poa,
    require_aligned,
    require_choice,
    require_nonempty,
    require_positive,
)

log = logging.getLogger(__name__)


def step_hours(index: pd.DatetimeIndex) -> float:
    """Length of one time step in hours (5-min grid -> 1/12): the median step of the index."""
    if len(index) < 2:
        raise ValueError(f"need at least two time steps to find the step length, got {len(index)}")
    hours = pd.Series(index).diff().median() / pd.Timedelta(hours=1)
    if not hours > 0:
        raise ValueError("time index must be strictly increasing")
    return hours


def reference_power(poa: pd.Series, t_cell: pd.Series, p0_kw: float, gamma: float | None = None,
                    t_ref: float | None = None) -> pd.Series:
    """Reference (expected-at-nameplate) power in kW.

        P_ref = P0 x G_poa / 1000 x [1 + gamma x (T_cell - T_ref)]

    P0 is the DC nameplate (kW), G_poa the plane-of-array irradiance (W/m2), gamma the power
    temperature coefficient (1/C, negative). Without t_ref the bracket is 1 (variant A).
    """
    require_positive(p0_kw, "P0 (nameplate power)")
    if t_ref is not None and gamma is None:
        raise ValueError("gamma is required when t_ref is given")
    ref = p0_kw * poa / 1000
    if t_ref is not None:
        ref = ref * (1 + gamma * (t_cell - t_ref))
    return ref


def weighted_t_ref(t_cell: pd.Series, poa: pd.Series, mask: pd.Series) -> float:
    """POA-weighted mean cell temperature: the 'typical operating temperature' of the plant.

        T_ref = sum(T_cell x G_poa) / sum(G_poa)   over the steps in `mask`
    """
    m = mask & t_cell.notna() & poa.notna()
    if not poa[m].sum() > 0:
        raise ValueError("no irradiance in the selected steps: cannot weight the reference temperature")
    return float((t_cell[m] * poa[m]).sum() / poa[m].sum())


def inverter_states(ac: pd.DataFrame, poa: pd.Series,
                    config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Boolean frames (known, down) per inverter-step, for daylight steps with POA > threshold.

    known: POA above the availability threshold and the inverter's AC is valid;
    down:  known and AC below down_p_ac_fraction x rating (e.g. < 8.25 kW of 825 kW).
    Steps with unknown AC (NaN) are neither available nor down: they leave the denominator.
    """
    k = config["kpi"]
    rating = config["inverters"]["ac_kw"][0]["ac_kw"]
    lit = (poa > k["availability_poa_min_wm2"]).to_numpy()[:, None]   # one column, broadcast
    known = ac.notna() & lit
    down = known & (ac < k["down_p_ac_fraction"] * rating)
    return known, down


def _period(index: pd.DatetimeIndex, freq: str) -> pd.PeriodIndex:
    """Local-time period label ('M' month, 'Y' year, 'D' day) for grouping."""
    return index.tz_convert(LOCAL_TZ).tz_localize(None).to_period(freq)


def kpi_table(inputs: pd.DataFrame, ac: pd.DataFrame, t_cell: pd.Series, config: Mapping[str, Any],
              gamma: float, t_ref_wc: float, freq: str = "M") -> pd.DataFrame:
    """KPIs per period. One row per local month ("M"), year ("Y") or day ("D").

    Columns: energy, reference yield, the three PR variants with and without downtime, the meter
    PR, specific yield and time-based availability, plus the share of usable inverter-steps.

    Formulas (sums over the usable inverter-steps of the period, dt in hours):

        PR_variant          = sum(P_ac x dt) / sum(P_ref x dt)
        P_ref (A raw)       = P0 x G_poa / 1000
        P_ref (B, C)        = P0 x G_poa / 1000 x [1 + gamma x (T_cell - T_ref)]
                              T_ref = 25 C (B) or the POA-weighted mean cell temperature (C)
        specific yield      = sum(P_ac x dt) / (P0 x number of inverters)      [kWh/kWp]
        availability (time) = 1 - down inverter-steps / known inverter-steps
                              down: P_ac < down_p_ac_fraction x rating at POA > availability_poa_min_wm2
    A ratio of sums, never a mean of 5-min ratios.
    """
    require_nonempty(inputs, "inputs")
    require_aligned(inputs, ac, ("inputs", "ac"))
    require_aligned(inputs, t_cell, ("inputs", "t_cell"))
    require_choice(freq, ("M", "Y", "D"), "freq")
    if ac.shape[1] == 0:
        raise ValueError("ac has no inverter columns")
    check_poa(inputs["poa_wm2"], config)
    dt = step_hours(inputs.index)
    p0_inv = inverter_dc_kw(config)
    n_inv = ac.shape[1]
    poa = inputs["poa_wm2"]

    # Step usable for PR: daylight, POA and cell temperature known.
    base = inputs["daylight"] & poa.notna() & t_cell.notna()
    valid = ac.notna() & base.to_numpy()[:, None]            # inverter-steps usable for PR
    lit_known, down = inverter_states(ac, poa, config)
    up = valid & ~down                                      # same, without stopped inverters

    refs = {
        "raw": reference_power(poa, t_cell, p0_inv),
        "stc": reference_power(poa, t_cell, p0_inv, gamma, 25.0),
        "wc": reference_power(poa, t_cell, p0_inv, gamma, t_ref_wc),
    }
    # Per step: measured energy of the usable inverters, and how many there are.
    parts = {
        "e_ac_kwh": ac.where(valid).sum(axis=1) * dt,
        "e_ac_up_kwh": ac.where(up).sum(axis=1) * dt,
        "n_valid": valid.sum(axis=1),
        "n_up": up.sum(axis=1),
    }
    for name, ref in refs.items():
        parts[f"ref_{name}_kwh"] = ref * parts["n_valid"] * dt
        parts[f"ref_{name}_up_kwh"] = ref * parts["n_up"] * dt
    # Meter: whole plant (all inverters' P0), only where the meter value is known.
    meter_ok = base & inputs["p_ac_meter_kw"].notna()
    parts["e_meter_kwh"] = inputs["p_ac_meter_kw"].where(meter_ok) * dt
    parts["ref_wc_meter_kwh"] = refs["wc"].where(meter_ok) * n_inv * dt
    # Cross-check: sum of inverters vs meter on daylight steps where all inverters and the meter
    # are known. Normal value about 1.02 (transformer and cable losses after the inverters);
    # clearly below 1 means the meter sees energy that the inverter channels do not report.
    all_known = inputs["daylight"] & inputs["p_ac_meter_kw"].notna() & (ac.notna().sum(axis=1) == n_inv)
    parts["e_inv_check_kwh"] = ac.sum(axis=1).where(all_known) * dt
    parts["e_meter_check_kwh"] = inputs["p_ac_meter_kw"].where(all_known) * dt
    # Specific yield: all known AC energy in daylight, regardless of the irradiance data.
    day_known = ac.notna() & inputs["daylight"].to_numpy()[:, None]
    parts["e_ac_known_kwh"] = ac.where(day_known).sum(axis=1) * dt
    parts["n_known"] = day_known.sum(axis=1)
    parts["n_daylight"] = inputs["daylight"].astype(int) * n_inv
    parts["h_poa_kwh_m2"] = poa.where(base) * dt / 1000
    # Availability counts.
    parts["n_avail_known"] = lit_known.sum(axis=1)
    parts["n_avail_down"] = down.sum(axis=1)

    g = pd.DataFrame(parts).groupby(_period(inputs.index, freq)).sum()
    # The first UTC hours of the period fall on the local evening before: no daylight, no row.
    g = g[g["n_daylight"] > 0]
    g.index.name = "local_period"
    out = pd.DataFrame(index=g.index)
    out["e_ac_mwh"] = g["e_ac_kwh"] / 1000
    out["h_poa_kwh_m2"] = g["h_poa_kwh_m2"]
    for name in refs:
        out[f"pr_{name}"] = g["e_ac_kwh"] / g[f"ref_{name}_kwh"]
        out[f"pr_{name}_excl_down"] = g["e_ac_up_kwh"] / g[f"ref_{name}_up_kwh"]
    out["pr_wc_meter"] = g["e_meter_kwh"] / g["ref_wc_meter_kwh"]
    # Measured energy per kWp of plant DC. Missing data are not filled in: read it together with
    # yield_coverage (share of daylight inverter-steps with known AC).
    out["specific_yield_kwh_kwp"] = g["e_ac_known_kwh"] / (p0_inv * n_inv)
    out["availability"] = 1 - g["n_avail_down"] / g["n_avail_known"]
    out["inv_to_meter"] = g["e_inv_check_kwh"] / g["e_meter_check_kwh"]
    out["pr_coverage"] = g["n_valid"] / g["n_daylight"]       # usable share of inverter-steps
    out["yield_coverage"] = g["n_known"] / g["n_daylight"]
    return out


def downtime_share(ac: pd.DataFrame, poa: pd.Series, config: Mapping[str, Any],
                   freq: str = "M") -> pd.DataFrame:
    """Share of lit, known inverter-steps with the inverter down: inverters x periods."""
    known, down = inverter_states(ac, poa, config)
    period = _period(ac.index, freq)
    share = down.groupby(period).sum() / known.groupby(period).sum()
    return share.T   # rows = inverters, columns = periods (NaN where nothing is known)
