"""Tests of the loss waterfall on a synthetic day (SYNTHETIC data, only for testing the code)."""

import numpy as np
import pandas as pd
import pvlib

from src.config import load_config
from src.losses import STEPS, check_closure, inverter_ac_unclipped, loss_waterfall
from src.model import expected_power

CONFIG = load_config("configs/system_9069.yaml")


def synthetic(n_inv=4, steps=144):
    """One local day of 5-min steps at constant POA 800 W/m2 and T_cell 45 C."""
    index = pd.date_range("2021-06-01 12:00", periods=steps, freq="5min", tz="UTC")
    inputs = pd.DataFrame({"poa_wm2": 800.0, "p_ac_meter_kw": 0.0, "daylight": True}, index=index)
    t_cell = pd.Series(45.0, index=index)
    exp = expected_power(inputs["poa_wm2"], t_cell, CONFIG)["p_ac_kw"]
    ac = pd.DataFrame({f"p_ac_inv{i:02d}_kw": 0.95 * exp for i in range(1, n_inv + 1)})
    return inputs, ac, t_cell, exp


def meter_for(ac, k):
    return ac.sum(axis=1) / k


def test_unclipped_matches_pvlib_below_rating():
    p_dc = pd.Series(np.linspace(1, 800, 50))
    inv = CONFIG["model"]["inverter"]
    ref = pvlib.inverter.pvwatts(p_dc, 825 / inv["eta_inv_nom"], inv["eta_inv_nom"], inv["eta_inv_ref"])
    assert np.allclose(inverter_ac_unclipped(p_dc, CONFIG), ref.clip(lower=0))


def test_closes_and_classifies():
    inputs, ac, t_cell, exp = synthetic()
    k = CONFIG["losses"]["inv_to_meter"]
    ac["p_ac_inv01_kw"] = 0.0                      # a real outage: the meter does not see it
    ac["p_ac_inv02_kw"] = np.nan                   # unknown channel
    inputs["p_ac_meter_kw"] = (0.95 * exp * 2 + 0.95 * exp) / k   # inverters 03, 04 + 02 running
    t = loss_waterfall(inputs, ac, t_cell, CONFIG, freq="D")
    check_closure(t)
    row = t.iloc[0]
    dt = 5 / 60
    assert np.isclose(row["downtime"], (exp * dt).sum(), rtol=1e-6)         # one inverter down
    assert np.isclose(row["running_shortfall"], (0.05 * exp * 2 * dt).sum(), rtol=1e-6)
    # Inverter 02 is unknown in the channels but produced 0.95 x expected (the meter sees it):
    # the credit takes back 0.95 x expected, the missing 5 % stays as its shortfall.
    assert np.isclose(row["unknown_ac"] + row["channel_credit"], (0.05 * exp * dt).sum(), rtol=1e-6)


def test_false_zero_is_credited_not_downtime():
    inputs, ac, t_cell, exp = synthetic()
    k = CONFIG["losses"]["inv_to_meter"]
    true_ac = ac.copy()
    ac["p_ac_inv01_kw"] = 0.0                       # channel writes 0, inverter really runs
    inputs["p_ac_meter_kw"] = true_ac.sum(axis=1) / k
    row = loss_waterfall(inputs, ac, t_cell, CONFIG, freq="D").iloc[0]
    dt = 5 / 60
    # The false zero is booked as downtime, then credited back at the real output (0.95 x exp).
    assert np.isclose(row["downtime"] + row["channel_credit"], (0.05 * exp * dt).sum(), rtol=1e-6)
    assert set(STEPS) <= set(row.index)
