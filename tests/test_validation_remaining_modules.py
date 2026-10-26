"""Input checks of degradation, outages, prices, Open-Meteo, synthetic faults and dashboard helpers.

SYNTHETIC inputs only (to test the code). Each test states the rule it protects.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from conftest import hourly_plant, plant_config
from pv_loss_auditor import dashboard_data as dd
from pv_loss_auditor import degradation as dg
from pv_loss_auditor import openmeteo, outages, prices
from pv_loss_auditor.synthetic_faults import inject_faults, plan_faults

# --- degradation -------------------------------------------------------------------------------


def test_daily_normalized_energy_rejects_empty_negative_poa_and_misaligned_inputs(config):
    cfg = plant_config(config, dc_kw=100)
    inputs, ac, t_cell = hourly_plant([800.0] * 4, 60.0)
    with pytest.raises(ValueError, match="empty"):
        dg.daily_normalized_energy(inputs.iloc[:0], ac.iloc[:0], t_cell.iloc[:0], cfg)
    with pytest.raises(ValueError, match="different time indices"):
        dg.daily_normalized_energy(inputs, ac.iloc[:-1], t_cell, cfg)
    inputs.iloc[1, inputs.columns.get_loc("poa_wm2")] = -50.0
    with pytest.raises(ValueError, match="negative"):
        dg.daily_normalized_energy(inputs, ac, t_cell, cfg)


def test_degradation_rate_needs_at_least_one_valid_day():
    series = pd.Series(np.nan, index=pd.date_range("2021-01-01", periods=10, freq="D"))
    with pytest.raises(ValueError, match="no valid days"):
        dg.yoy_rate(series)


def test_soiling_needs_data_and_a_positive_number_of_bootstrap_repetitions():
    empty = pd.DataFrame({"energy_norm": [], "poa_kwh_m2": []}, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="empty"):
        dg.soiling_run(empty)
    day = pd.DataFrame(
        {"energy_norm": [0.9], "poa_kwh_m2": [5.0]}, index=pd.date_range("2021-01-01", periods=1)
    )
    with pytest.raises(ValueError, match="reps"):
        dg.soiling_run(day, reps=0)


def test_weighted_loss_is_the_poa_weighted_mean_and_needs_irradiance():
    # 10 % loss while the POA is 100, none while it is 300: (100 x 0.10) / 400 = 2.5 % of the insolation.
    assert dg.weighted_loss_pct(pd.Series([100.0, 300.0]), pd.Series([0.10, 0.0])) == pytest.approx(2.5)
    with pytest.raises(ValueError, match="irradiance"):
        dg.weighted_loss_pct(pd.Series([0.0, 0.0]), pd.Series([0.1, 0.1]))


# --- outages -----------------------------------------------------------------------------------


def test_down_days_rejects_empty_and_misaligned_inputs(config):
    inputs, ac, _ = hourly_plant([800.0] * 4, 60.0)
    with pytest.raises(ValueError, match="empty"):
        outages.down_days(ac.iloc[:0], inputs["poa_wm2"].iloc[:0], config)
    with pytest.raises(ValueError, match="different time indices"):
        outages.down_days(ac, inputs["poa_wm2"].iloc[:-1], config)


def test_inverter_meter_ratio_is_one_when_the_meter_sees_what_the_inverters_report(config):
    # Two inverters report 60 kW each; the meter sees their sum divided by the inverter-to-meter factor
    # (transformer losses). sum(inverters) / (meter x factor) must then be exactly 1.
    k = config["losses"]["inv_to_meter"]
    inputs, ac, _ = hourly_plant(np.full(24, 800.0), 60.0)
    ac["p_ac_inv02_kw"] = 60.0
    inputs["p_ac_meter_kw"] = 120.0 / k
    ratio = outages.inverter_meter_ratio(inputs, ac, config)  # 24 UTC hours span two local days
    assert ratio.to_numpy() == pytest.approx(1.0)
    bad = copy.deepcopy(config)
    bad["losses"]["inv_to_meter"] = 0
    with pytest.raises(ValueError, match="positive"):
        outages.inverter_meter_ratio(inputs, ac, bad)


# --- prices and Open-Meteo ---------------------------------------------------------------------


def price_config(fraction):
    return {"price": {"year": 2026, "technology": "Solar", "sensitivity_fraction": fraction}}


def test_price_band_is_symmetric_around_the_base_price(monkeypatch):
    # 30.19 USD/MWh with a band of 30 %: 21.13 ... 39.25 (the figures quoted in the README).
    monkeypatch.setattr(
        prices, "load_avoided_energy_cost", lambda c: pd.DataFrame({"Solar": [30.19]}, index=[2026])
    )
    base, low, high = prices.price_usd_per_mwh(price_config(0.30))
    assert (base, low, high) == pytest.approx((30.19, 21.133, 39.247))


@pytest.mark.parametrize("fraction", [-0.1, 1.0, 1.5])
def test_price_band_outside_zero_and_one_is_rejected(monkeypatch, fraction):
    monkeypatch.setattr(
        prices, "load_avoided_energy_cost", lambda c: pd.DataFrame({"Solar": [30.19]}, index=[2026])
    )
    with pytest.raises(ValueError, match="sensitivity_fraction"):
        prices.price_usd_per_mwh(price_config(fraction))


def test_non_positive_price_in_the_table_is_rejected(monkeypatch):
    monkeypatch.setattr(
        prices, "load_avoided_energy_cost", lambda c: pd.DataFrame({"Solar": [0.0]}, index=[2026])
    )
    with pytest.raises(ValueError, match="positive"):
        prices.price_usd_per_mwh(price_config(0.3))


def test_unknown_weather_source_is_rejected_before_any_request(config, tmp_path):
    # "best_match" would silently give IFS data under the wrong label: only era5 and ifs are allowed.
    with pytest.raises(ValueError, match="source"):
        openmeteo.fetch("best_match", config, out_dir=str(tmp_path))


# --- synthetic faults --------------------------------------------------------------------------


def test_fault_events_must_name_existing_inverters_and_a_share_between_0_and_1():
    ac = pd.DataFrame(
        {"p_ac_inv01_kw": 100.0}, index=pd.date_range("2021-06-01", periods=24, freq="1h", tz="UTC")
    )
    event = {
        "inverter": "p_ac_inv01_kw",
        "start": pd.Timestamp("2021-06-01"),
        "end": pd.Timestamp("2021-06-01"),
    }
    with pytest.raises(ValueError, match="not in ac"):
        inject_faults(ac, pd.DataFrame([{**event, "inverter": "p_ac_inv99_kw", "loss_fraction": 0.2}]))
    with pytest.raises(ValueError, match="between 0 and 1"):
        inject_faults(ac, pd.DataFrame([{**event, "loss_fraction": 1.2}]))


def test_a_fault_of_a_fifth_of_the_strings_scales_ac_by_four_fifths():
    index = pd.date_range("2021-06-01", periods=48, freq="1h", tz="UTC")
    ac = pd.DataFrame({"p_ac_inv01_kw": 100.0}, index=index)
    event = {
        "inverter": "p_ac_inv01_kw",
        "start": pd.Timestamp("2021-06-01"),
        "end": pd.Timestamp("2021-06-01"),
        "loss_fraction": 0.2,
    }
    out = inject_faults(ac, pd.DataFrame([event]))
    local_day = index.tz_convert("America/New_York").normalize().tz_localize(None)
    assert out.loc[local_day == "2021-06-01", "p_ac_inv01_kw"].eq(80.0).all()
    assert out.loc[local_day != "2021-06-01", "p_ac_inv01_kw"].eq(100.0).all()
    assert ac["p_ac_inv01_kw"].eq(100.0).all()  # the real series is never modified


def test_fault_plan_needs_valid_days(config):
    with pytest.raises(ValueError, match="empty"):
        plan_faults(pd.DataFrame(), config)


# --- dashboard helpers -------------------------------------------------------------------------


def test_usd_is_kwh_times_price_per_mwh_and_the_price_cannot_be_negative():
    assert dd.to_usd(1000.0, 30.0) == pytest.approx(30.0)  # 1 MWh x 30 USD/MWh
    with pytest.raises(ValueError, match="negative"):
        dd.to_usd(1000.0, -1.0)


def test_causes_table_accepts_only_year_or_total():
    monthly = pd.DataFrame({"nominal_kwh": [1.0]}, index=["2021-06"])
    with pytest.raises(ValueError, match="per must be"):
        dd.causes_table(monthly, per="month")
