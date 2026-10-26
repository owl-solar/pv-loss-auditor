"""Cleaning flags, warnings and ValueErrors on SYNTHETIC inputs (only for testing the code)."""

import logging

import numpy as np
import pandas as pd
import pytest

from conftest import hourly_plant, plant_config
from pv_loss_auditor import anomalies as an
from pv_loss_auditor import kpi, losses, model
from pv_loss_auditor.config import load_config
from pv_loss_auditor.data import flag_ac, flag_sensor_limits, flag_stale
from pv_loss_auditor.validation import warn_suspicious

# --- cleaning ------------------------------------------------------------------------------


def day_frame(values, elevation=30.0):
    index = pd.date_range("2021-06-01 12:00", periods=len(values), freq="5min", tz="UTC")
    return pd.DataFrame({"x": np.asarray(values, dtype=float), "sun_elevation_deg": elevation}, index=index)


def test_frozen_sensor_in_daylight_is_flagged_but_a_changing_one_is_not(config):
    # 10 identical readings in a row (more than the 30-minute window): every repeat is flagged, the
    # first reading of the run is still a legitimate value. A signal that moves every step is never stuck.
    cl = config["clean"]
    stuck = flag_stale(day_frame([500, 510, 520] + [530] * 10 + [540, 550]), ["x"], cl)["flag_stale__x"]
    assert stuck.iloc[4:13].all() and not stuck.iloc[:4].any()
    moving = flag_stale(day_frame(np.arange(500, 520)), ["x"], cl)["flag_stale__x"]
    assert not moving.any()


def test_identical_values_at_night_are_not_flagged(config):
    # Zero irradiance at night is real: the stuck test runs in daylight only (sun elevation > 5 deg).
    night = day_frame([0.0] * 20, elevation=-20.0)
    assert not flag_stale(night, ["x"], config["clean"])["flag_stale__x"].any()


def test_inverter_zero_and_clipping_plateau_are_not_stuck_but_a_frozen_value_is(config):
    cl = config["clean"]
    for value, expected_flag in ((0.0, False), (825.0, False), (400.0, True)):
        flags = flag_stale(day_frame([value] * 12), ["x"], cl, ac_rating_kw=825.0)["flag_stale__x"]
        assert flags.any() == expected_flag, value


def test_ac_out_of_range_and_sensor_limits_are_flagged(config):
    data = day_frame([100.0, 900.0, -50.0, 825.0])
    # above 1.05 x rating (866 kW) or below -10 kW
    assert flag_ac(data, ["x"], config["clean"], 825.0)["flag_range__x"].tolist() == [False, True, True, False]
    temp = day_frame([20.0, 95.0, -40.0]).rename(columns={"x": "t_mod_tc01s1_c"})
    flags = flag_sensor_limits(temp, config["clean"]["sensor_limits"])
    assert flags["flag_range__t_mod_tc01s1_c"].tolist() == [False, True, True]


def test_flagged_values_become_nan_in_the_model_inputs():
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "flag_stale__a": [False, True, False],
                         "flag_range__a": [False, False, True]})
    assert model.valid(data, "a").isna().tolist() == [False, True, True]


def test_load_inputs_keeps_missing_values_as_nan_and_never_negative_poa(fixture_config):
    inputs, ac = model.load_inputs(fixture_config)
    assert inputs.index.equals(ac.index) and ac.shape[1] == 40
    assert (inputs["poa_wm2"].dropna() >= -5).all()
    assert ac.isna().any().any()                              # real unknown values stay NaN, not filled


# --- warnings ------------------------------------------------------------------------------


def make_inputs(poa):
    index = pd.date_range("2021-06-01", periods=len(poa), freq="60min", tz="UTC")
    return pd.DataFrame({"poa_wm2": np.asarray(poa, dtype=float), "t_mod_c": 30.0 + np.arange(len(poa)) * 0.1,
                         "p_ac_meter_kw": 10.0 + np.arange(len(poa)), "daylight": True}, index=index)


def test_flat_sensor_gives_a_warning(config, caplog):
    with caplog.at_level(logging.WARNING):
        warn_suspicious(make_inputs(np.full(20, 800.0)), config)
    assert "flat sensor: poa_wm2" in caplog.text


def test_healthy_inputs_give_no_warning(config, caplog):
    with caplog.at_level(logging.WARNING):
        warn_suspicious(make_inputs(np.linspace(100, 900, 20)), config)
    assert caplog.text == ""


def test_large_gap_gives_a_warning(config, caplog):
    inputs = make_inputs(np.linspace(100, 900, 20))
    shifted = inputs.index.to_series()
    shifted.iloc[10:] += pd.Timedelta(days=3)                   # 3-day hole in the middle
    inputs.index = pd.DatetimeIndex(shifted)
    with caplog.at_level(logging.WARNING):
        warn_suspicious(inputs, config)
    assert "gap of" in caplog.text


def test_missing_poa_gives_a_warning(config, caplog):
    inputs = make_inputs(np.linspace(100, 900, 20))
    inputs.loc[inputs.index[:10], "poa_wm2"] = np.nan           # half of the daylight steps without POA
    with caplog.at_level(logging.WARNING):
        warn_suspicious(inputs, config)
    assert "POA is missing" in caplog.text


# --- ValueError on nonsensical inputs --------------------------------------------------------


def test_empty_period_is_rejected(config):
    inputs, ac, t_cell = hourly_plant([800.0, 800.0], 60.0)
    inputs, ac, t_cell = inputs.iloc[:0], ac.iloc[:0], t_cell.iloc[:0]
    with pytest.raises(ValueError, match="empty"):
        kpi.kpi_table(inputs, ac, t_cell, config, -0.004, 25.0)
    with pytest.raises(ValueError, match="empty"):
        losses.loss_waterfall(inputs, ac, t_cell, config)


def test_negative_poa_is_rejected(config):
    cfg = plant_config(config, dc_kw=100)
    inputs, ac, t_cell = hourly_plant([800.0, -120.0], 60.0)
    with pytest.raises(ValueError, match="negative"):
        kpi.kpi_table(inputs, ac, t_cell, cfg, -0.004, 25.0)
    with pytest.raises(ValueError, match="negative"):
        model.expected_power(inputs["poa_wm2"], t_cell, cfg)


def test_small_negative_sensor_offset_is_tolerated(config):
    cfg = plant_config(config, dc_kw=100)
    inputs, ac, t_cell = hourly_plant([800.0, -2.0], 60.0)
    kpi.kpi_table(inputs, ac, t_cell, cfg, -0.004, 25.0)


@pytest.mark.parametrize("dc_kw, n_inv, ac_kw", [(0, 1, 825), (-5, 1, 825), (100, 0, 825), (100, 1, 0)])
def test_zero_power_plant_is_rejected(config, dc_kw, n_inv, ac_kw):
    cfg = plant_config(config, dc_kw=dc_kw, n_inverters=n_inv, ac_kw=ac_kw)
    with pytest.raises(ValueError, match="positive"):
        model.expected_power(pd.Series([800.0]), pd.Series([25.0]), cfg)


def test_reference_power_rejects_zero_p0_and_missing_gamma():
    poa, t = pd.Series([800.0]), pd.Series([25.0])
    with pytest.raises(ValueError, match="positive"):
        kpi.reference_power(poa, t, 0.0)
    with pytest.raises(ValueError, match="gamma"):
        kpi.reference_power(poa, t, 100.0, t_ref=25.0)


def test_weighted_t_ref_without_irradiance_is_rejected():
    with pytest.raises(ValueError, match="irradiance"):
        kpi.weighted_t_ref(pd.Series([20.0]), pd.Series([0.0]), pd.Series([True]))


def test_single_timestep_has_no_step_length():
    with pytest.raises(ValueError, match="two time steps"):
        kpi.step_hours(pd.date_range("2021-01-01", periods=1, freq="5min", tz="UTC"))


def test_misaligned_indices_and_unknown_freq_are_rejected(config):
    cfg = plant_config(config, dc_kw=100)
    inputs, ac, t_cell = hourly_plant([800.0] * 4, 60.0)
    with pytest.raises(ValueError, match="different time indices"):
        kpi.kpi_table(inputs, ac.iloc[:-1], t_cell, cfg, -0.004, 25.0)
    with pytest.raises(ValueError, match="freq"):
        kpi.kpi_table(inputs, ac, t_cell, cfg, -0.004, 25.0, freq="W")


def test_positive_temperature_coefficient_is_rejected(config):
    cfg = plant_config(config, dc_kw=100)
    for mod in cfg["modules"]:
        mod["gamma_pmp_pct_per_c"] = 0.4
    with pytest.raises(ValueError, match="temperature coefficient"):
        model.weighted_gamma(cfg)


def test_non_positive_price_is_rejected():
    with pytest.raises(ValueError, match="price"):
        losses.to_usd(pd.DataFrame({"a": [1.0]}), 0.0, 21.0, 39.0)


def test_config_without_required_sections_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("system: {id: 1}\n", encoding="utf8")
    with pytest.raises(ValueError, match="missing required"):
        load_config(path)
    path.write_text("", encoding="utf8")
    with pytest.raises(ValueError, match="empty"):
        load_config(path)


def test_flat_residual_cannot_be_scored_by_the_rule():
    x = pd.DataFrame({"dev_rel_median": np.zeros(10)})
    with pytest.raises(ValueError, match="sigma"):
        an.rule_score(x)


def test_waterfall_that_does_not_close_is_reported():
    broken = pd.DataFrame({"closure_kwh": [0.0, 250.0]})
    with pytest.raises(ValueError, match="does not close"):
        losses.check_closure(broken)


def test_usd_value_is_kwh_times_price_per_mwh():
    # 2000 kWh = 2 MWh; at 30 USD/MWh = 60 USD, band 21 ... 39 USD/MWh.
    usd = losses.to_usd(pd.DataFrame({"a": [2000.0]}), 30.0, 21.0, 39.0)
    assert [usd[k]["a"].iloc[0] for k in ("base", "low", "high")] == pytest.approx([60.0, 42.0, 78.0])


def test_decreasing_time_index_and_empty_inverter_table_are_rejected(config):
    with pytest.raises(ValueError, match="increasing"):
        kpi.step_hours(pd.DatetimeIndex(["2021-01-01 00:10", "2021-01-01 00:05"], tz="UTC"))
    inputs, ac, t_cell = hourly_plant([800.0, 800.0], 60.0)
    with pytest.raises(ValueError, match="no inverter columns"):
        kpi.kpi_table(inputs, ac.iloc[:, :0], t_cell, config, -0.004, 25.0)


def test_downtime_share_per_inverter(config):
    # inverter a: 1 of 4 sunny steps at 0 kW; inverter b: never down
    inputs, ac, _ = hourly_plant([800.0] * 4, [60.0, 0.0, 60.0, 60.0])
    ac["b"] = 60.0
    share = kpi.downtime_share(ac.rename(columns={"p_ac_inv01_kw": "a"}), inputs["poa_wm2"], config, freq="Y")
    assert share.iloc[:, 0].to_dict() == pytest.approx({"a": 0.25, "b": 0.0})
