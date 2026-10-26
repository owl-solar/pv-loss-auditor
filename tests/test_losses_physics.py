"""Loss waterfall balance and clipping on SYNTHETIC plants (only for testing the code)."""

import numpy as np
import pandas as pd
import pytest

from conftest import hourly_plant, plant_config
from pv_loss_auditor.kpi import step_hours
from pv_loss_auditor.losses import STEPS, check_closure, loss_waterfall, waterfall_scope
from pv_loss_auditor.model import expected_power


def random_plant(config, seed, n_inv=6, hours=72):
    """Hourly plant with sun/night, noisy inverters, real outages, lost channels and an arbitrary meter."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-06-01 00:00", periods=hours, freq="1h", tz="UTC")
    hod = np.arange(hours) % 24
    poa = pd.Series(np.clip(900 * np.sin(np.pi * (hod - 6) / 12), 0, None), index=index)
    t_cell = pd.Series(25 + 20 * poa / 1000, index=index)
    exp = expected_power(poa, t_cell, config)["p_ac_kw"]
    ac = pd.DataFrame(
        {f"p_ac_inv{i:02d}_kw": exp * rng.uniform(0.85, 1.0, hours) for i in range(1, n_inv + 1)}
    )
    stopped = rng.random(ac.shape) < 0.10
    unknown = rng.random(ac.shape) < 0.05
    ac = ac.mask(stopped, 0.0).mask(unknown)
    meter = ac.sum(axis=1) / config["losses"]["inv_to_meter"] * rng.uniform(0.9, 1.15, hours)
    inputs = pd.DataFrame({"poa_wm2": poa, "daylight": poa > 0, "p_ac_meter_kw": meter})
    return inputs, ac, t_cell


@pytest.mark.parametrize("seed", range(5))
def test_waterfall_closes_for_arbitrary_data(config, seed):
    # nominal = sum of all loss steps + metered, whatever the data look like (the cascade is built
    # from differences of neighbouring cumulative quantities, so nothing can leak).
    inputs, ac, t_cell = random_plant(config, seed)
    for freq in ("D", "Y"):
        t = loss_waterfall(inputs, ac, t_cell, config, freq=freq)
        check_closure(t, tolerance_kwh=1e-6)
        assert (t["nominal_kwh"] - t[STEPS].sum(axis=1) - t["metered_kwh"]).abs().max() < 1e-6


@pytest.mark.parametrize("seed", range(5))
def test_expected_equals_actual_plus_lost_energy(config, seed):
    # The measured side of the waterfall:
    #   expected = energy the inverters report + downtime + unknown channels + running shortfall.
    # Each inverter-step is in exactly one of three states (down, unknown, running), and the gap
    # expected - measured is booked in the matching step.
    inputs, ac, t_cell = random_plant(config, seed)
    t = loss_waterfall(inputs, ac, t_cell, config, freq="Y").iloc[0]
    _, scope_2d = waterfall_scope(inputs, ac, t_cell)
    reported_kwh = ac.where(scope_2d).sum().sum() * step_hours(inputs.index)
    assert t["expected_kwh"] == pytest.approx(
        reported_kwh + t["downtime"] + t["unknown_ac"] + t["running_shortfall"], rel=1e-9
    )


def test_every_running_inverter_at_90_percent_gives_a_10_percent_shortfall(config):
    # No outages, no unknowns: the loss between expected and measured is exactly 10 % of expected.
    cfg = plant_config(config, dc_kw=400, n_inverters=2)
    inputs, _, t_cell = hourly_plant(np.full(24, 600.0), 0.0, t_cell=40.0)
    exp = expected_power(inputs["poa_wm2"], t_cell, cfg)["p_ac_kw"]
    ac = pd.DataFrame({"p_ac_inv01_kw": 0.9 * exp, "p_ac_inv02_kw": 0.9 * exp})
    inputs["p_ac_meter_kw"] = ac.sum(axis=1) / cfg["losses"]["inv_to_meter"]
    t = loss_waterfall(inputs, ac, t_cell, cfg, freq="Y").iloc[0]
    assert t["running_shortfall"] == pytest.approx(0.1 * t["expected_kwh"])
    assert t["downtime"] == 0 and t["unknown_ac"] == 0
    assert abs(t["reconciliation"]) < 1e-6 and abs(t["channel_credit"]) < 1e-6


def test_no_clipping_when_dc_equals_ac_and_power_is_below_rating(config):
    # DC/AC = 1.0 (825 kWp on an 825 kW inverter). At 800 W/m2 and 40 C the DC side gives
    # 825 x 0.8 x (1 - 0.00416 x 15) x (1 - 0.0634) = ~580 kW < 825 kW: nothing to clip.
    cfg = plant_config(config, dc_kw=825.0, ac_kw=825.0)
    inputs, ac, t_cell = hourly_plant(np.full(24, 800.0), 500.0, t_cell=40.0)
    t = loss_waterfall(inputs, ac, t_cell, cfg, freq="Y").iloc[0]
    assert t["clipping"] == pytest.approx(0.0, abs=1e-9)
    assert expected_power(inputs["poa_wm2"], t_cell, cfg)["p_ac_kw"].max() < 825.0


def test_clipping_appears_with_oversized_dc_and_expected_power_stays_at_the_rating(config):
    # DC/AC = 1.5: at 1000 W/m2 and 25 C the DC side gives ~1160 kW > 825 kW. The surplus is clipped
    # and the model output sits exactly at the rating.
    cfg = plant_config(config, dc_kw=1237.5, ac_kw=825.0)
    inputs, ac, t_cell = hourly_plant(np.full(24, 1000.0), 800.0, t_cell=25.0)
    t = loss_waterfall(inputs, ac, t_cell, cfg, freq="Y").iloc[0]
    assert t["clipping"] > 0
    assert t["expected_kwh"] == pytest.approx(825.0 * 24)
