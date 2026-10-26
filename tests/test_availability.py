"""Tests of the channel-loss-corrected availability on one SYNTHETIC day (only for testing the code).

Four inverters, each delivering 0.95 x the model (E = expected energy of one inverter-block over
the day). The meter is set by hand, so the true state of every scenario is known.
"""

import numpy as np
import pandas as pd
import pytest

from pv_loss_auditor.config import load_config
from pv_loss_auditor.losses import availability_revised
from pv_loss_auditor.model import expected_power

CONFIG = load_config("configs/system_9069.yaml")
K = CONFIG["losses"]["inv_to_meter"]


def scenario(zero_inverters=(), meter_inverters=(1, 2, 3, 4)):
    """One local day (12 h of daylight at 800 W/m2). zero_inverters report 0; the meter sees the
    energy of meter_inverters running at 0.95 x the model."""
    index = pd.date_range("2021-06-01 12:00", periods=144, freq="5min", tz="UTC")
    inputs = pd.DataFrame({"poa_wm2": 800.0, "daylight": True}, index=index)
    t_cell = pd.Series(45.0, index=index)
    exp = expected_power(inputs["poa_wm2"], t_cell, CONFIG)["p_ac_kw"]
    ac = pd.DataFrame({f"p_ac_inv{i:02d}_kw": 0.95 * exp for i in range(1, 5)})
    inputs["p_ac_meter_kw"] = 0.95 * exp * len(meter_inverters) / K
    for i in zero_inverters:
        ac[f"p_ac_inv{i:02d}_kw"] = 0.0
    return inputs, ac, t_cell


def run(zero_inverters=(), meter_inverters=(1, 2, 3, 4)):
    inputs, ac, t_cell = scenario(zero_inverters, meter_inverters)
    return availability_revised(inputs, ac, t_cell, CONFIG, freq="D").iloc[0]


def test_healthy_plant_is_fully_available_in_all_three_measures():
    r = run()
    assert r["availability_time_raw"] == pytest.approx(1.0)
    assert r["availability_time_upper"] == pytest.approx(1.0)
    assert r["availability_energy"] == pytest.approx(1.0)
    assert r["channel_days"] == 0


def test_real_stop_is_confirmed_by_the_meter_and_stays_a_stop():
    # inverter 1 really stopped: the meter loses its energy too, sum / meter = 1
    r = run(zero_inverters=(1,), meter_inverters=(2, 3, 4))
    assert r["channel_days"] == 0
    assert r["availability_time_raw"] == pytest.approx(0.75)
    assert r["availability_time_upper"] == pytest.approx(0.75)      # not a channel-loss day: no change
    assert r["availability_energy"] == pytest.approx(0.75)          # 1 stopped of 4, all confirmed


def test_false_zero_is_a_channel_loss_day_and_not_a_stop():
    # inverter 1 reports 0 but the meter still sees all four: lost data channel
    r = run(zero_inverters=(1,), meter_inverters=(1, 2, 3, 4))
    assert r["channel_days"] == 1
    assert r["availability_time_raw"] == pytest.approx(0.75)        # lower bound: every zero is a stop
    assert r["availability_time_upper"] == pytest.approx(1.0)       # zero leaves the denominator
    # Waterfall: downtime E (expected - 0) and credit 0.95 E (what the meter saw of it): 0.05 E
    # stays as "stop", the running shortfall of that inverter (documented bias): 1 - 0.05 / 4.
    assert r["availability_energy"] == pytest.approx(1 - 0.05 / 4)


def test_mixed_case_the_upper_bound_overstates_and_the_energy_measure_stays_close():
    # inverter 1 false zero (meter sees it), inverter 2 really stopped (meter does not)
    r = run(zero_inverters=(1, 2), meter_inverters=(1, 3, 4))
    assert r["channel_days"] == 1
    assert r["availability_time_raw"] == pytest.approx(0.5)
    assert r["availability_time_upper"] == pytest.approx(1.0)       # drops the real stop as well
    # truth is 0.75 (one real stop of four); the energy measure gives 1 - (2 - 0.95) / 4 = 0.7375
    assert r["availability_energy"] == pytest.approx(0.7375)
    assert abs(r["availability_energy"] - 0.75) < 0.02
    assert r["availability_time_raw"] < r["availability_energy"] < r["availability_time_upper"]
