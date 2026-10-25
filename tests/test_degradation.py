"""Tests of the Phase 4 helpers (src/degradation.py) on SYNTHETIC data, only for testing the code.

Expected values are worked out by hand (see the comments), not copied from the code output.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from src import degradation as dg
from src.config import load_config
from src.model import expected_power, inverter_dc_kw

CONFIG = load_config("configs/system_9069.yaml")
N_INV = 4


def synthetic_days(n_days=6, poa=800.0):
    """Five-minute UTC data: daylight 12:00-23:55 UTC (08:00-19:55 EDT), night otherwise.

    Returns (inputs, ac, t_cell). Every inverter delivers 0.90 x the model while it runs.
    """
    index = pd.date_range("2021-06-01 00:00", periods=288 * n_days, freq="5min", tz="UTC")
    daylight = pd.Series((index.hour >= 12), index=index)
    inputs = pd.DataFrame({"poa_wm2": np.where(daylight, poa, 0.0), "daylight": daylight,
                           "p_ac_meter_kw": 0.0}, index=index)
    t_cell = pd.Series(40.0, index=index)
    exp = expected_power(inputs["poa_wm2"], t_cell, CONFIG)["p_ac_kw"]
    ac = pd.DataFrame({f"p_ac_inv{i:02d}_kw": 0.9 * exp for i in range(1, N_INV + 1)})
    return inputs, ac, t_cell


def day(index, text):
    return index.tz_convert("America/New_York").normalize().tz_localize(None) == pd.Timestamp(text)


def test_normalized_energy_counts_only_running_inverters():
    inputs, ac, t_cell = synthetic_days(n_days=2)
    stopped = day(ac.index, "2021-06-02")
    ac.loc[stopped, ["p_ac_inv01_kw", "p_ac_inv02_kw"]] = 0.0        # half of the plant stopped
    out = dg.daily_normalized_energy(inputs, ac, t_cell, CONFIG)
    # Both days: the running inverters deliver exactly 0.90 x the model. The stopped ones are
    # downtime (waterfall), not a performance drop, so the ratio stays 0.90 on day 2.
    assert out["energy_norm"].tolist() == pytest.approx([0.9, 0.9])
    assert out["running_share"].tolist() == pytest.approx([1.0, 0.5])
    assert out["coverage"].tolist() == pytest.approx([1.0, 1.0])     # stopped = usable, not missing
    # 800 W/m2 for 12 h = 9.6 kWh/m2
    assert out["poa_kwh_m2"].iloc[0] == pytest.approx(9.6)


def test_daily_series_applies_the_three_day_filters():
    inputs, ac, t_cell = synthetic_days(n_days=6)
    i = ac.index
    ac.loc[day(i, "2021-06-02"), ["p_ac_inv01_kw", "p_ac_inv02_kw"]] = 0.0     # 50 % running: kept
    ac.loc[day(i, "2021-06-03"), ["p_ac_inv01_kw", "p_ac_inv02_kw", "p_ac_inv03_kw"]] = 0.0  # 25 %: dropped
    inputs.loc[day(i, "2021-06-04"), "poa_wm2"] *= 0.1               # 0.96 kWh/m2 < 2: dropped
    ac.loc[day(i, "2021-06-05"), "p_ac_inv01_kw"] = np.nan           # 75 % usable < 90 %: dropped
    out = dg.daily_series(inputs, ac, t_cell, CONFIG)
    kept = out["energy_norm"].notna()
    assert kept.tolist() == [True, True, False, False, False, True]
    assert out["energy_norm"][kept].tolist() == pytest.approx([0.9, 0.9, 0.9])
    assert out.index.freqstr == "D" and out.index.tz is None         # regular grid for RdTools
    assert out["pr_wc"][~kept].isna().all()                          # PR column is filtered too


def test_yoy_rate_recovers_a_known_linear_degradation():
    days = pd.date_range("2018-01-01", periods=3 * 365, freq="D")
    t = np.arange(len(days)) / 365.0
    rng = np.random.default_rng(0)
    # 0.9 normalized energy, -1 %/yr of the starting level, +-3 % seasonal swing, 0.5 % noise
    series = pd.Series(0.9 * (1 - 0.01 * t) * (1 + 0.03 * np.sin(2 * np.pi * t))
                       + rng.normal(0, 0.0045, len(days)), index=days)
    rd, lo, hi = dg.yoy_rate(series)
    assert rd == pytest.approx(-1.0, abs=0.15)
    assert lo <= rd <= hi


def test_yoy_rate_of_a_flat_series_is_zero():
    days = pd.date_range("2018-01-01", periods=3 * 365, freq="D")
    t = np.arange(len(days)) / 365.0
    series = pd.Series(0.9 * (1 + 0.03 * np.sin(2 * np.pi * t)), index=days)   # season only
    assert dg.yoy_rate(series)[0] == pytest.approx(0.0, abs=0.01)


def test_filter_sensitivity_passes_each_threshold_pair_to_daily_series(monkeypatch):
    seen = []

    def fake_daily_series(inputs, ac, t_cell, config, gamma=None):
        seen.append((config["degradation"]["min_daily_poa_kwh_m2"],
                     config["degradation"]["min_running_share"]))
        days = pd.date_range("2018-01-01", periods=3 * 365, freq="D")
        t = np.arange(len(days)) / 365.0
        return pd.DataFrame({"energy_norm": 0.9 * (1 - 0.01 * t)}, index=days)

    monkeypatch.setattr(dg, "daily_series", fake_daily_series)
    table = dg.filter_sensitivity(None, None, None, CONFIG, poa_grid=[1.0, 2.0],
                                  running_grid=[0.3, 0.5], end="2019-12-31")
    assert seen == [(1.0, 0.3), (1.0, 0.5), (2.0, 0.3), (2.0, 0.5)]
    assert len(table) == 4 * 2                                       # two periods per pair
    assert set(table["period"]) == {"to 2019-12-31", "full"}
    assert table["days"].max() == 3 * 365 and table["days"].min() == 2 * 365
    assert (table["rd_pct_yr"] < 0).all()


def test_daily_precipitation_sums_over_local_days():
    hours = pd.date_range("2021-06-01 00:00", periods=72, freq="1h", tz="UTC")
    rain = pd.Series(0.0, index=hours)
    rain["2021-06-01 03:00"] = 1.0          # 23:00 EDT of 05-31
    rain["2021-06-01 05:00"] = 2.5          # 01:00 EDT of 06-01
    rain["2021-06-01 20:00"] = 0.5          # 16:00 EDT of 06-01
    out = dg.daily_precipitation(pd.DataFrame({"precipitation": rain}))
    assert out.index.tz is None
    assert out["2021-05-31"] == pytest.approx(1.0)
    assert out["2021-06-01"] == pytest.approx(3.0)
    assert out["2021-06-02"] == pytest.approx(0.0)


def sawtooth_soiling(n_days=3 * 365, period=50, rate=0.002):
    """Normalized energy that loses `rate` per day and is reset by a cleaning every `period` days."""
    days = pd.date_range("2018-01-01", periods=n_days, freq="D")
    since_clean = np.arange(n_days) % period
    daily = pd.DataFrame({"energy_norm": 0.9 * (1 - rate * since_clean),
                          "poa_kwh_m2": 6.0}, index=days)
    rain = pd.Series(0.0, index=days)
    rain.iloc[np.arange(0, n_days, period)] = 5.0                    # rain on every cleaning day
    return daily, rain


def test_soiling_loss_scales_with_the_soiling_rate_of_a_sawtooth():
    # RdTools reports an insolation-weighted soiling ratio against a "perfect clean" level, not
    # the plain mean loss since the last cleaning: for this sawtooth it is 1.3-1.5 x the plain
    # mean (rate x 24.5 days), consistently over rates (characterised, not derived by hand).
    loss = {}
    for rate in (0.001, 0.002):
        daily, rain = sawtooth_soiling(rate=rate)
        ratio, lo, hi, n_valid, n_all = dg.soiling_run(daily, precip=rain, reps=30)
        loss[rate] = 1 - ratio
        plain = rate * 24.5
        assert 1.2 * plain < loss[rate] < 1.6 * plain
        assert lo - 0.005 <= ratio <= hi + 0.005                    # CI brackets the estimate
        assert n_valid == n_all and n_valid >= 10                   # every interval is a clean sawtooth
    assert loss[0.002] / loss[0.001] == pytest.approx(2.0, abs=0.3)  # twice the rate, twice the loss


def test_soiling_run_without_rain_data_finds_the_same_cleanings_on_a_clean_sawtooth():
    daily, rain = sawtooth_soiling()
    with_rain = dg.soiling_run(daily, precip=rain, reps=30)
    without = dg.soiling_run(daily, precip=None, reps=30)
    assert without[0] == pytest.approx(with_rain[0], abs=0.005)     # jumps are obvious without rain


def test_soiling_run_is_reproducible_with_the_fixed_seed():
    daily, rain = sawtooth_soiling()
    a = dg.soiling_run(daily, precip=rain, reps=30)
    b = dg.soiling_run(daily, precip=rain, reps=30)
    assert a == b


def hourly_plant(n_up_down=0):
    """5-min plant data 12:00-15:00 UTC; four inverters at 400 kW, POA 800 W/m2."""
    index = pd.date_range("2021-06-01 12:00", "2021-06-01 15:00", freq="5min", tz="UTC")
    inputs = pd.DataFrame({"poa_wm2": 800.0, "daylight": True}, index=index)
    ac = pd.DataFrame({f"p_ac_inv{i:02d}_kw": 400.0 for i in range(1, N_INV + 1)}, index=index)
    if n_up_down:
        ac.iloc[:, -n_up_down:] = 0.0
    return inputs, ac


def test_pr_by_poa_source_matches_a_hand_calculation():
    inputs, ac = hourly_plant()
    hours = pd.date_range("2021-06-01 13:00", periods=3, freq="1h", tz="UTC")   # hour-end labels
    external = {"too_high": pd.Series(880.0, index=hours)}                        # +10 % vs plant
    pr = dg.pr_by_poa_source(inputs, ac, CONFIG, external)
    # Per inverter-hour: 400 kWh / (967.175 kWp x 0.8) = 0.5170. External POA +10 %: 0.5170 / 1.1.
    p0 = inverter_dc_kw(CONFIG)
    assert pr["plant"].loc[2021] == pytest.approx(400 / (p0 * 0.8))
    assert pr["too_high"].loc[2021] == pytest.approx(400 / (p0 * 0.88))


def test_pr_by_poa_source_leaves_stopped_inverters_out_and_skips_thin_hours():
    inputs, ac = hourly_plant(n_up_down=1)                          # inverter 4 stopped
    hours = pd.date_range("2021-06-01 12:00", periods=4, freq="1h", tz="UTC")
    external = {"same": pd.Series(800.0, index=hours)}
    pr = dg.pr_by_poa_source(inputs, ac, CONFIG, external)
    p0 = inverter_dc_kw(CONFIG)
    # Only the three running inverters count, on both sides. The label 12:00 holds one 5-min step
    # (< min_valid = 10 of 12) and is dropped, which does not change the ratio here.
    assert pr["plant"].loc[2021] == pytest.approx(400 / (p0 * 0.8))
    assert pr["same"].loc[2021] == pytest.approx(pr["plant"].loc[2021])
