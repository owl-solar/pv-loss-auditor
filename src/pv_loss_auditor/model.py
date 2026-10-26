"""Expected-yield model of one inverter block: measured POA and cell temperature -> AC power.

Chain (pvlib functions called directly, so every step stays visible):
  POA (reference cells) --pvwatts_dc--> DC at the modules --x (1 - design losses)--> DC at the
  inverter input --pvwatts_inverter--> AC, clipped at the inverter rating.
The model uses nameplate data and explicit design losses only; it is NOT fitted to the
measurements, so everything else the plant loses ends up in the residual (measured - expected).
All 40 inverters have the same DC capacity (notebook 01 section 3, notebook 02 section 3), so one
expected series describes every inverter.
"""

import logging
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pvlib
import pyarrow.parquet as pq

from pv_loss_auditor.data import load_clean
from pv_loss_auditor.validation import check_poa, require_nonempty, require_positive, warn_suspicious

log = logging.getLogger(__name__)

# A temperature coefficient outside this range is a unit mistake (percent instead of fraction).
GAMMA_RANGE_PER_C = (-0.02, 0.0)


def valid(data: pd.DataFrame, column: str) -> pd.Series:
    """Column with flagged values (flag_stale / flag_range) set to NaN."""
    bad = pd.Series(False, index=data.index)
    for test in ("flag_stale", "flag_range"):
        flag = f"{test}__{column}"
        if flag in data:
            bad |= data[flag].astype(bool)
    return data[column].where(~bad)


def inverter_columns(columns: Iterable[str]) -> list[str]:
    """Sorted names of the inverter AC power columns (p_ac_inv01_kw ...)."""
    return sorted(c for c in columns if re.match(r"p_ac_inv\d{2}_kw$", c))


def load_inputs(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Model inputs from the clean dataset, flagged values already set to NaN.

    Returns (inputs, ac):
    - inputs: poa_wm2, t_mod_c (means of the valid sensors), t_amb_c, p_ac_meter_kw, daylight,
      sun_elevation_deg, poa_clearsky_wm2;
    - ac: one column per inverter in kW. NaN = unknown (missing or flagged), 0 = measured zero.
    """
    m = config["model"]
    first_file = sorted(Path(config["clean"]["out_dir"]).glob("9069_clean_*.parquet"))[0]
    all_columns = pq.read_schema(first_file).names   # read the column list without the data
    inverters = inverter_columns(all_columns)
    meter = config["kpi"]["meter_channel"]
    sensors = m["poa_channels"] + m["t_mod_channels"] + [m["t_amb_channel"], meter]
    wanted = sensors + inverters + ["flag_daylight", "sun_elevation_deg", "poa_clearsky_wm2"]
    flags = [c for c in all_columns
             if c.startswith(("flag_stale__", "flag_range__")) and c.split("__", 1)[1] in wanted]
    data = load_clean(config, columns=wanted + flags)   # parquet reads only these columns

    # Mean over the valid sensors: NaN only if none of them is valid at that step.
    inputs = pd.DataFrame({
        "poa_wm2": pd.concat([valid(data, c) for c in m["poa_channels"]], axis=1).mean(axis=1),
        "t_mod_c": pd.concat([valid(data, c) for c in m["t_mod_channels"]], axis=1).mean(axis=1),
        "t_amb_c": valid(data, m["t_amb_channel"]),
        "p_ac_meter_kw": valid(data, meter),
        "daylight": data["flag_daylight"].astype(bool),
        "sun_elevation_deg": data["sun_elevation_deg"],
        "poa_clearsky_wm2": data["poa_clearsky_wm2"],
    })
    ac = pd.DataFrame({c: valid(data, c) for c in inverters})

    require_nonempty(inputs, "clean dataset")
    check_poa(inputs["poa_wm2"], config)
    warn_suspicious(inputs, config)
    log.info("model inputs: %d steps %s .. %s, %d inverters", len(inputs), inputs.index.min(),
             inputs.index.max(), ac.shape[1])
    return inputs, ac


def weighted_gamma(config: Mapping[str, Any]) -> float:
    """DC-weighted temperature coefficient of the module table, in 1/C (e.g. -0.00416)."""
    modules = config["modules"]
    dc = [mod["quantity"] * mod["pnom_w"] for mod in modules]
    require_positive(sum(dc), "DC nameplate of the module table")
    gamma = sum(w * mod["gamma_pmp_pct_per_c"] for w, mod in zip(dc, modules)) / sum(dc)
    gamma = gamma / 100
    if not GAMMA_RANGE_PER_C[0] <= gamma < GAMMA_RANGE_PER_C[1]:
        raise ValueError(f"temperature coefficient {gamma:.5f} 1/C is outside {GAMMA_RANGE_PER_C}: "
                         "gamma_pmp_pct_per_c is in percent per degree and must be negative")
    return gamma


def cell_temperature(inputs: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    """Cell temperature in C: mean back-of-module thermocouple + delta_t x G/1000.

    pvlib sapm_cell_from_module: the junction is hotter than the back sheet by deltaT at
    1000 W/m2. NaN where no thermocouple is valid; there is no modelled fallback (see config).
    """
    return pvlib.temperature.sapm_cell_from_module(
        inputs["t_mod_c"], inputs["poa_wm2"], deltaT=config["model"]["delta_t_cell_c"],
    ).rename("t_cell_c")


def dc_loss_fraction(config: Mapping[str, Any]) -> float:
    """Combined DC design losses as a fraction (pvwatts_losses multiplies the factors)."""
    losses = config["model"]["dc_losses_pct"]
    # Everything the audit should measure is set to 0 explicitly, not left at pvlib's defaults.
    total_pct = pvlib.pvsystem.pvwatts_losses(
        soiling=0, shading=0, snow=0, availability=0, age=0, lid=losses["lid"],
        mismatch=losses["mismatch"], wiring=losses["wiring"], connections=losses["connections"],
        nameplate_rating=losses["nameplate_rating"])
    return total_pct / 100


def inverter_dc_kw(config: Mapping[str, Any]) -> float:
    """DC nameplate per inverter: plant DC split equally over the inverters."""
    dc_kw = require_positive(config["capacity"]["dc_kw"], "capacity.dc_kw")
    return dc_kw / require_positive(config["inverters"]["count"], "inverters.count")


def expected_power(poa: pd.Series, t_cell: pd.Series, config: Mapping[str, Any],
                   gamma: float | None = None) -> pd.DataFrame:
    """Expected DC and AC power of one inverter block (kW), DataFrame with p_dc_kw and p_ac_kw.

    gamma (1/C) defaults to the DC-weighted nameplate value; pass another value for sensitivity.
    """
    gamma = weighted_gamma(config) if gamma is None else gamma
    check_poa(poa, config)
    pdc0 = inverter_dc_kw(config)
    # PVWatts DC: pdc0 x G/1000 x (1 + gamma x (T_cell - 25)). Linear in G: no low-light loss.
    p_dc_module = pvlib.pvsystem.pvwatts_dc(poa, t_cell, pdc0, gamma, temp_ref=25.0)
    p_dc = p_dc_module * (1 - dc_loss_fraction(config))

    inv = config["model"]["inverter"]
    pac0 = require_positive(config["inverters"]["ac_kw"][0]["ac_kw"], "inverter AC rating")  # 825 kW, verified in QA
    # pvwatts_inverter clips at pdc0_inv x eta_inv_nom; choosing pdc0_inv = pac0 / eta_inv_nom
    # puts the clipping level exactly at the AC rating.
    p_ac = pvlib.inverter.pvwatts(p_dc, pdc0=pac0 / inv["eta_inv_nom"],
                                  eta_inv_nom=inv["eta_inv_nom"], eta_inv_ref=inv["eta_inv_ref"])
    # At night (G = 0) pvwatts returns a small negative tare loss; the model reports 0.
    return pd.DataFrame({"p_dc_kw": p_dc, "p_ac_kw": p_ac.clip(lower=0)})
