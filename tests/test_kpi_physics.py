"""KPIs on cases with a known answer (SYNTHETIC plants, only for testing the code)."""

import numpy as np
import pandas as pd
import pytest

from conftest import hourly_plant, plant_config
from pv_loss_auditor import kpi

GAMMA = -0.004


def table(config, inputs, ac, t_cell, t_ref=25.0, freq="Y"):
    return kpi.kpi_table(inputs, ac, t_cell, config, GAMMA, t_ref, freq=freq).iloc[0]


def test_pr_ideal_case_is_the_ratio_of_energy_to_reference(config):
    # 1000 h at 1000 W/m2 = 1000 kWh/m2 on a 100 kW plant: reference yield 100 kW x 1000 h = 100 000 kWh.
    # The plant delivers 85 000 kWh (85 kW all the time), so PR = 0.85 in every variant at T_cell = T_ref.
    cfg = plant_config(config, dc_kw=100)
    inputs, ac, t_cell = hourly_plant(np.full(1000, 1000.0), 85.0, t_cell=25.0)
    r = table(cfg, inputs, ac, t_cell)
    assert r["h_poa_kwh_m2"] == pytest.approx(1000.0)
    assert r["e_ac_mwh"] == pytest.approx(85.0)
    for variant in ("pr_raw", "pr_stc", "pr_wc", "pr_wc_meter"):
        assert r[variant] == pytest.approx(0.85)
    assert r["specific_yield_kwh_kwp"] == pytest.approx(850.0)  # 85 000 kWh / 100 kWp


def test_temperature_corrected_pr_removes_the_thermal_loss_and_raw_pr_keeps_it(config):
    # Cells at 45 C: output falls by gamma x dT = 0.4 %/C x 20 C = 8 %, to 85 x 0.92 = 78.2 kW.
    # Raw PR sees the loss (0.782); PR corrected to 25 C divides it out and returns 0.85.
    cfg = plant_config(config, dc_kw=100)
    inputs, ac, t_cell = hourly_plant(np.full(1000, 1000.0), 85.0 * (1 + GAMMA * 20), t_cell=45.0)
    r = table(cfg, inputs, ac, t_cell, t_ref=25.0)
    assert r["pr_raw"] == pytest.approx(0.782)
    assert r["pr_stc"] == pytest.approx(0.85)
    # T_ref = 45 C (the weather-corrected variant at the operating temperature): no correction needed.
    assert table(cfg, inputs, ac, t_cell, t_ref=45.0)["pr_wc"] == pytest.approx(0.782)


def test_pr_is_a_ratio_of_sums_not_a_mean_of_ratios(config):
    # Two steps: 1000 W/m2 delivering 90 kW and 100 W/m2 delivering 5 kW (P0 = 100 kW).
    # Ratio of sums: 95 / (100 + 10) = 0.8636; mean of ratios would be (0.9 + 0.5) / 2 = 0.70.
    cfg = plant_config(config, dc_kw=100)
    inputs, ac, t_cell = hourly_plant([1000.0, 100.0], [90.0, 5.0])
    assert table(cfg, inputs, ac, t_cell)["pr_raw"] == pytest.approx(95 / 110)


def test_weighted_t_ref_is_the_poa_weighted_mean(config):
    t_cell = pd.Series([20.0, 40.0])
    poa = pd.Series([100.0, 300.0])
    assert kpi.weighted_t_ref(t_cell, poa, pd.Series([True, True])) == pytest.approx(35.0)


def test_availability_night_hours_are_not_downtime(config):
    # 2 days. Sun (800 W/m2) 08:00-16:00 with the inverter running; at night POA = 0 and AC = 0.
    # A night zero is the normal state: availability must stay 1.
    cfg = plant_config(config, dc_kw=100)
    hours = np.arange(48) % 24
    day = (hours >= 8) & (hours < 16)
    inputs, ac, t_cell = hourly_plant(np.where(day, 800.0, 0.0), np.where(day, 60.0, 0.0))
    assert table(cfg, inputs, ac, t_cell)["availability"] == pytest.approx(1.0)


def test_availability_full_outage_is_zero_and_no_outage_is_one(config):
    cfg = plant_config(config, dc_kw=100)
    poa = np.full(48, 800.0)
    up = table(cfg, *hourly_plant(poa, 60.0))
    down = table(cfg, *hourly_plant(poa, 0.0))
    assert up["availability"] == pytest.approx(1.0)
    assert down["availability"] == pytest.approx(0.0)


def test_availability_half_day_outage_is_one_half_and_unknown_data_is_ignored(config):
    # 10 sunny hours: 5 running, 3 down (AC = 0), 2 unknown (NaN). Unknown steps leave the
    # denominator: availability = 1 - 3 / 8 = 0.625.
    cfg = plant_config(config, dc_kw=100)
    ac_power = [60.0] * 5 + [0.0] * 3 + [np.nan] * 2
    inputs, ac, t_cell = hourly_plant(np.full(10, 800.0), ac_power)
    assert table(cfg, inputs, ac, t_cell)["availability"] == pytest.approx(0.625)


def test_stopped_inverter_lowers_pr_but_not_pr_of_running_inverters(config):
    # Two inverters, one stopped for the whole time: plant PR halves, PR over running inverters does not.
    cfg = plant_config(config, dc_kw=200, n_inverters=2)  # P0 = 100 kW per inverter
    inputs, ac, t_cell = hourly_plant(np.full(100, 1000.0), 85.0)
    ac["p_ac_inv02_kw"] = 0.0
    r = kpi.kpi_table(inputs, ac, t_cell, cfg, GAMMA, 25.0, freq="Y").iloc[0]
    assert r["pr_raw"] == pytest.approx(0.425)
    assert r["pr_raw_excl_down"] == pytest.approx(0.85)
