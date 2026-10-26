"""Tests of the expected-yield model and the KPI helpers on small synthetic inputs.

Expected values are worked out by hand (see the comments), not copied from the code output.
"""

import numpy as np
import pandas as pd
import pytest

from pv_loss_auditor import kpi, model
from pv_loss_auditor.config import load_config

CONFIG = load_config("configs/system_9069.yaml")


def test_weighted_gamma_is_dc_weighted_nameplate():
    # 25 992 x 295 W at -0.28 and 103 398 x 300 W at -0.45 %/C: DC-weighted mean -0.416 %/C.
    assert model.weighted_gamma(CONFIG) == pytest.approx(-0.00416, abs=2e-5)


def test_dc_loss_fraction_multiplies_the_items():
    # 1 - 0.985 x 0.98 x 0.985 x 0.995 x 0.99 = 0.0634 (adding the items would give 6.5 %).
    assert model.dc_loss_fraction(CONFIG) == pytest.approx(0.0634, abs=5e-4)


def test_inverter_dc_is_plant_dc_split_equally():
    assert model.inverter_dc_kw(CONFIG) == pytest.approx(38687 / 40)


def test_expected_power_clips_at_rating_and_is_zero_at_night():
    poa = pd.Series([0.0, 200.0, 1000.0])
    t_cell = pd.Series([20.0, 30.0, 25.0])
    out = model.expected_power(poa, t_cell, CONFIG)
    assert out["p_ac_kw"].iloc[0] == 0.0                         # night: no negative tare loss
    assert out["p_ac_kw"].iloc[2] == pytest.approx(825.0)        # 906 kW DC > rating: clipped
    # 200 W/m2: AC below DC (inverter loss) but within 10 % of it.
    low = out.iloc[1]
    assert 0.9 * low["p_dc_kw"] < low["p_ac_kw"] < low["p_dc_kw"]


def test_expected_power_falls_with_temperature():
    poa = pd.Series([500.0, 500.0])
    out = model.expected_power(poa, pd.Series([25.0, 55.0]), CONFIG)
    # Below the clipping level the loss is gamma x dT = 0.416 %/C x 30 C = 12.5 % of DC power.
    ratio = out["p_dc_kw"].iloc[1] / out["p_dc_kw"].iloc[0]
    assert ratio == pytest.approx(1 - 0.00416 * 30, abs=1e-4)


def test_cell_temperature_adds_delta_t_at_full_sun():
    inputs = pd.DataFrame({"t_mod_c": [40.0, 40.0], "poa_wm2": [1000.0, 0.0]})
    t_cell = model.cell_temperature(inputs, CONFIG)
    assert t_cell.tolist() == pytest.approx([43.0, 40.0])        # +3 C x G/1000


def test_reference_power_with_and_without_temperature_factor():
    poa, t_cell = pd.Series([800.0]), pd.Series([35.0])
    assert kpi.reference_power(poa, t_cell, 1000.0).iloc[0] == pytest.approx(800.0)
    # 1000 kW x 0.8 x (1 - 0.004 x (35 - 25)) = 768 kW
    ref = kpi.reference_power(poa, t_cell, 1000.0, gamma=-0.004, t_ref=25.0)
    assert ref.iloc[0] == pytest.approx(768.0)


def test_weighted_t_ref_weights_by_poa():
    t_cell = pd.Series([20.0, 40.0, 99.0])
    poa = pd.Series([100.0, 300.0, 500.0])
    mask = pd.Series([True, True, False])                         # third step excluded
    # (20 x 100 + 40 x 300) / 400 = 35
    assert kpi.weighted_t_ref(t_cell, poa, mask) == pytest.approx(35.0)


def test_step_hours_for_five_minute_grid():
    index = pd.date_range("2021-06-01", periods=5, freq="5min", tz="UTC")
    assert kpi.step_hours(index) == pytest.approx(1 / 12)


def test_inverter_states_unknown_is_neither_available_nor_down():
    index = pd.date_range("2021-06-01 12:00", periods=5, freq="5min", tz="UTC")
    poa = pd.Series([0.0, 40.0, 600.0, 600.0, 600.0], index=index)
    ac = pd.DataFrame({"a": [0.0, 0.0, 0.0, 500.0, np.nan]}, index=index)
    known, down = kpi.inverter_states(ac, poa, CONFIG)
    # POA <= 50 W/m2 does not count; NaN is unknown (leaves the denominator), 0 at sun is down.
    assert known["a"].tolist() == [False, False, True, True, False]
    assert down["a"].tolist() == [False, False, True, False, False]
