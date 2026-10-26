"""Loss waterfall: nameplate energy at the measured irradiance -> energy at the plant meter.

Sequential cascade, every step is the difference of two neighbouring cumulative quantities, so the
steps add up to (nominal - metered) by construction:

  nominal  P0 x G/1000                                   (STC energy at the measured irradiance)
  - temperature        gamma x (T_cell - 25 C), negative loss = gain in cold weather
  - design DC losses   LID, mismatch, wiring, ... (an ASSUMPTION, see `model:` in the config)
  - inverter           PVWatts efficiency curve, unclipped
  - clipping           AC above the inverter rating
  = expected           the model of src/pv_loss_auditor/model.py
  - downtime           inverter reports ~0 at daylight (AC < 1 % of rating at POA > 50 W/m2)
  - unknown AC         inverter-steps with no valid AC value (missing or flagged)
  - channel credit     energy that the meter sees but the inverter channels do not report
                       (days when sum(inverters) / meter is low): NOT a loss, it corrects the two
                       lines above, so it is negative
  - running shortfall  expected - measured on inverters that run (soiling, degradation, wrong
                       design losses, sensor bias ...): split further in Phase 4
  - reconciliation     meter vs sum of inverters on the remaining days (noise, calibration)
  - after inverters    transformer and cable losses between inverter and meter
  = metered energy

Scope: 5-min steps with daylight, POA, cell temperature and meter all known, so both sides of
every difference cover the same steps. Signs: positive = loss.
"""

import copy
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pv_loss_auditor.kpi import _period, inverter_states, step_hours
from pv_loss_auditor.loss_groups import CAUSES, STEPS, by_cause  # noqa: F401  (re-exported: losses.STEPS etc.)
from pv_loss_auditor.model import dc_loss_fraction, expected_power, inverter_dc_kw, weighted_gamma
from pv_loss_auditor.validation import (
    check_poa,
    require_aligned,
    require_choice,
    require_nonempty,
    require_positive,
)

log = logging.getLogger(__name__)


def inverter_ac_unclipped(p_dc_kw: pd.Series, config: Mapping[str, Any]) -> pd.Series:
    """PVWatts inverter AC without the clipping at the rating (kW).

    Same formula as pvlib.inverter.pvwatts (Driesse), which offers no switch to turn the clip
    off; tests/test_losses.py checks that both agree below the rating.
    """
    inv = config["model"]["inverter"]
    eta_nom, eta_ref = inv["eta_inv_nom"], inv["eta_inv_ref"]
    pdc0 = config["inverters"]["ac_kw"][0]["ac_kw"] / eta_nom  # as in model.expected_power
    p_dc = p_dc_kw.clip(lower=0)
    zeta = (p_dc / pdc0).where(p_dc > 0)  # avoid 0 / 0 at night
    eta = eta_nom / eta_ref * (-0.0162 * zeta - 0.0059 / zeta + 0.9858)
    return (eta * p_dc).clip(lower=0).fillna(0.0)


def waterfall_scope(
    inputs: pd.DataFrame, ac: pd.DataFrame, t_cell: pd.Series
) -> tuple[pd.Series, pd.DataFrame]:
    """Steps used by the waterfall (daylight, POA, T_cell and meter known) as a Series, and the
    same as a frame with one column per inverter."""
    scope = inputs["daylight"] & inputs["poa_wm2"].notna() & t_cell.notna() & inputs["p_ac_meter_kw"].notna()
    scope_2d = pd.DataFrame(
        np.repeat(scope.to_numpy()[:, None], ac.shape[1], axis=1), index=ac.index, columns=ac.columns
    )
    return scope, scope_2d


def channel_loss_flag(
    s_known_kwh: pd.Series, meter_at_ac_kwh: pd.Series, index: pd.DatetimeIndex, tolerance: float
) -> pd.Series:
    """True on every step of a local day whose sum(inverters) / meter is below 1 - tolerance.

    s_known_kwh: energy the inverter channels report per step; meter_at_ac_kwh: meter energy
    brought to inverter-AC level (x inv_to_meter). Both on the waterfall scope.
    """
    day = _period(index, "D")
    ratio = s_known_kwh.groupby(day).transform("sum") / meter_at_ac_kwh.groupby(day).transform("sum")
    return (ratio < 1 - tolerance).fillna(False)


def loss_waterfall(
    inputs: pd.DataFrame,
    ac: pd.DataFrame,
    t_cell: pd.Series,
    config: Mapping[str, Any],
    gamma: float | None = None,
    freq: str = "Y",
) -> pd.DataFrame:
    """Waterfall in kWh per local period (rows), one column per step.

    Columns: nominal_kwh, the loss steps in STEPS (kWh, positive = loss), expected_kwh,
    metered_kwh and closure_kwh (nominal - sum of steps - metered; must be ~0).

    Balance, by construction of the cascade (every step is a difference of neighbouring
    cumulative quantities):

        nominal = sum(STEPS) + metered        =>   closure_kwh = nominal - sum(STEPS) - metered = 0
    """
    require_nonempty(inputs, "inputs")
    require_aligned(inputs, ac, ("inputs", "ac"))
    require_aligned(inputs, t_cell, ("inputs", "t_cell"))
    require_choice(freq, ("M", "Y", "D"), "freq")
    check_poa(inputs["poa_wm2"], config)
    gamma = weighted_gamma(config) if gamma is None else gamma
    cfg = config["losses"]
    require_positive(cfg["inv_to_meter"], "losses.inv_to_meter")
    if not 0 <= cfg["channel_tolerance"] < 1:
        raise ValueError(f"losses.channel_tolerance must be in [0, 1), got {cfg['channel_tolerance']!r}")
    n = ac.shape[1]
    dt = step_hours(inputs.index)
    poa = inputs["poa_wm2"]
    meter = inputs["p_ac_meter_kw"]

    # Scope: the same steps for every quantity of the waterfall.
    scope, scope_2d = waterfall_scope(inputs, ac, t_cell)

    # --- model side: one inverter block, then x n inverters --------------------------------
    p0 = inverter_dc_kw(config)
    nominal = p0 * poa / 1000  # kW per inverter
    after_temp = nominal * (1 + gamma * (t_cell - 25.0))
    after_dc = after_temp * (1 - dc_loss_fraction(config))
    ac_unclipped = inverter_ac_unclipped(after_dc, config)
    ac_expected = expected_power(poa, t_cell, config, gamma)["p_ac_kw"]
    # Model side in kWh for the whole plant, only on steps in scope.
    to_kwh = n * dt
    w = pd.DataFrame(
        {
            "nominal_kwh": nominal * to_kwh,
            "temperature": (nominal - after_temp) * to_kwh,
            "dc_losses": (after_temp - after_dc) * to_kwh,
            "inverter": (after_dc - ac_unclipped) * to_kwh,
            "clipping": (ac_unclipped - ac_expected) * to_kwh,
            "expected_kwh": ac_expected * to_kwh,
        }
    ).where(scope, 0.0)

    # --- measured side ----------------------------------------------------------------------
    _, down_all = inverter_states(ac, poa, config)
    known = ac.notna() & scope_2d
    down = down_all & scope_2d
    unknown = ac.isna() & scope_2d
    running = known & ~down
    exp_2d = pd.DataFrame(
        np.repeat(ac_expected.to_numpy()[:, None], n, axis=1), index=ac.index, columns=ac.columns
    )
    gap = (exp_2d - ac).where(known)  # expected - measured, kW

    w["downtime"] = gap.where(down).sum(axis=1) * dt
    w["unknown_ac"] = exp_2d.where(unknown).sum(axis=1) * dt
    w["running_shortfall"] = gap.where(running).sum(axis=1) * dt

    # Meter at inverter-AC level; the part between is lost after the inverters.
    k = cfg["inv_to_meter"]
    meter_kwh = (meter * dt).where(scope, 0.0)
    w["after_inverters"] = meter_kwh * (k - 1)
    s_known = ac.where(known).sum(axis=1) * dt  # what the channels report
    diff = meter_kwh * k - s_known  # what the meter sees on top

    # Daily test: low sum(inverters) / meter => channels lose energy (zeros while running).
    channel_day = channel_loss_flag(s_known, meter_kwh * k, inputs.index, cfg["channel_tolerance"])
    w["channel_credit"] = -diff.where(channel_day, 0.0)
    w["reconciliation"] = -diff.where(~channel_day, 0.0)
    w["metered_kwh"] = meter_kwh
    w["channel_day"] = channel_day & scope

    g = w.drop(columns="channel_day").groupby(_period(inputs.index, freq)).sum()
    g = g[g["nominal_kwh"] > 0]
    g["closure_kwh"] = g["nominal_kwh"] - g[STEPS].sum(axis=1) - g["metered_kwh"]
    g.index.name = "local_period"
    log.info("waterfall: %d periods (%s), %d channel-loss steps", len(g), freq, int(w["channel_day"].sum()))
    return g


def check_closure(table: pd.DataFrame, tolerance_kwh: float = 1.0) -> None:
    """Stop if the steps do not add up to nominal - metered in any period."""
    worst = table["closure_kwh"].abs().max()
    if worst > tolerance_kwh:
        raise ValueError(f"waterfall does not close: worst period off by {worst:.1f} kWh")


def to_usd(
    table_kwh: pd.DataFrame, price_base: float, price_low: float, price_high: float
) -> dict[str, pd.DataFrame]:
    """Table in kWh -> dict of tables in USD at the base price and the sensitivity band."""
    for name, p in (("base", price_base), ("low", price_low), ("high", price_high)):
        require_positive(p, f"price_{name} (USD/MWh)")
    return {
        name: table_kwh * p / 1000
        for name, p in (("base", price_base), ("low", price_low), ("high", price_high))
    }


def dc_loss_sensitivity(
    inputs: pd.DataFrame,
    ac: pd.DataFrame,
    t_cell: pd.Series,
    config: Mapping[str, Any],
    gamma: float | None = None,
) -> pd.DataFrame:
    """Annual running shortfall and total losses for each assumed design DC-loss total.

    Mostly `dc_losses` and `running_shortfall` trade places (downtime moves slightly because it
    is valued at the expected power), which shows how much of the unexplained shortfall depends
    on the assumed design losses.
    """
    rows = {}
    for total in config["losses"]["dc_loss_sensitivity_pct"]:
        c = copy.deepcopy(config)
        # One loss item with the whole total: pvwatts_losses combines the items multiplicatively,
        # so a single item reproduces the total exactly.
        c["model"]["dc_losses_pct"] = {
            "lid": total,
            "mismatch": 0,
            "wiring": 0,
            "connections": 0,
            "nameplate_rating": 0,
        }
        t = loss_waterfall(inputs, ac, t_cell, c, gamma, freq="Y")
        check_closure(t)
        rows[total] = t[["dc_losses", "downtime", "running_shortfall"]].sum()
    return pd.DataFrame(rows).T.rename_axis("dc_loss_pct")


def availability_revised(
    inputs: pd.DataFrame,
    ac: pd.DataFrame,
    t_cell: pd.Series,
    config: Mapping[str, Any],
    gamma: float | None = None,
    freq: str = "Y",
) -> pd.DataFrame:
    """Availability corrected for lost data channels (Phase 2 KPI, revised after Phase 3).

    The time-based availability of Phase 2 counts every inverter that reports ~0 as stopped. On
    days when the meter sees more energy than the inverter channels report (channel-loss days of
    the waterfall) part of those zeros are lost data, not stops, and which inverter lies cannot
    be read from the files. So three numbers per period:
    - availability_time_raw    all zeros are real stops (identical to kpi_table, a LOWER bound);
    - availability_time_upper  apparent stops on channel-loss days leave the denominator (an
                               UPPER bound: real stops of those days are dropped as well);
    - availability_energy      central: 1 - (downtime + channel credit) / expected energy of the
                               waterfall, i.e. the stop energy the meter confirms. Energy-weighted,
                               not time-weighted. The credit is the energy the missing channels
                               would have reported at their MEASURED level, so the running
                               shortfall of those inverters (~10 % of their expected energy) stays
                               in the stop energy: a small bias towards lower availability.
    channel_days is the number of channel-loss local days in the period.

        availability_energy = 1 - (downtime + channel_credit) / expected_kwh      (waterfall columns)
        availability_time   = 1 - down inverter-steps / known inverter-steps
    """
    require_nonempty(inputs, "inputs")
    require_aligned(inputs, ac, ("inputs", "ac"))
    gamma = weighted_gamma(config) if gamma is None else gamma
    poa = inputs["poa_wm2"]
    dt = step_hours(inputs.index)

    # Channel-loss days: the same test as in the waterfall, on the waterfall scope.
    scope, scope_2d = waterfall_scope(inputs, ac, t_cell)
    s_known = ac.where(ac.notna() & scope_2d).sum(axis=1) * dt
    meter_at_ac = (inputs["p_ac_meter_kw"] * dt).where(scope, 0.0) * config["losses"]["inv_to_meter"]
    flag = (
        channel_loss_flag(s_known, meter_at_ac, inputs.index, config["losses"]["channel_tolerance"]) & scope
    )
    day = _period(inputs.index, "D")
    channel_day_step = flag.groupby(day).transform("max")  # every step of a flagged day

    known, down = inverter_states(ac, poa, config)  # POA > 50 W/m2, as in kpi_table
    on_channel_day = pd.DataFrame(
        np.repeat(channel_day_step.to_numpy()[:, None], ac.shape[1], axis=1),
        index=ac.index,
        columns=ac.columns,
    )
    period = _period(inputs.index, freq)
    n_known = known.sum(axis=1).groupby(period).sum()
    n_down = down.sum(axis=1).groupby(period).sum()
    n_known_up = (known & ~(down & on_channel_day)).sum(axis=1).groupby(period).sum()
    n_down_up = (down & ~on_channel_day).sum(axis=1).groupby(period).sum()

    w = loss_waterfall(inputs, ac, t_cell, config, gamma, freq=freq)
    flag_day = flag.groupby(day).max()  # one bool per local day
    channel_days = flag_day.groupby(flag_day.index.asfreq(freq)).sum()
    out = pd.DataFrame(
        {
            "availability_time_raw": 1 - n_down / n_known,
            "availability_time_upper": 1 - n_down_up / n_known_up,
            "availability_energy": 1 - (w["downtime"] + w["channel_credit"]) / w["expected_kwh"],
            "channel_days": channel_days,
        }
    )
    out.index.name = "local_period"
    return out.loc[w.index]
