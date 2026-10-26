"""Tests of the dashboard export on a synthetic plant (SYNTHETIC data, only for testing the code)."""

import numpy as np
import pandas as pd

from pv_loss_auditor.config import load_config
from pv_loss_auditor.export_dashboard import kpi_tables, waterfall_tables, write_tables
from pv_loss_auditor.model import expected_power, weighted_gamma

CONFIG = load_config("configs/system_9069.yaml")


def synthetic(n_inv=4):
    """Two local days in June 2020 at constant POA 800 W/m2 and T_cell 45 C; inverters 5 % below
    the model, meter consistent with them."""
    index = pd.date_range("2020-06-01 12:00", periods=288, freq="5min", tz="UTC")
    inputs = pd.DataFrame({"poa_wm2": 800.0, "p_ac_meter_kw": 0.0, "daylight": True}, index=index)
    t_cell = pd.Series(45.0, index=index)
    exp = expected_power(inputs["poa_wm2"], t_cell, CONFIG)["p_ac_kw"]
    ac = pd.DataFrame({f"p_ac_inv{i:02d}_kw": 0.95 * exp for i in range(1, n_inv + 1)})
    inputs["p_ac_meter_kw"] = ac.sum(axis=1) / CONFIG["losses"]["inv_to_meter"]
    return inputs, ac, t_cell


def test_waterfall_months_close_and_years_follow():
    inputs, ac, t_cell = synthetic()
    tables = waterfall_tables(inputs, ac, t_cell, CONFIG, weighted_gamma(CONFIG))
    monthly = tables["waterfall_monthly"]
    assert list(monthly.index) == ["2020-06"]
    assert monthly["closure_kwh"].abs().max() < 1.0
    assert set(tables["dc_loss_sensitivity"].index) == set(CONFIG["losses"]["dc_loss_sensitivity_pct"])


def test_yearly_kpi_keeps_the_first_year():
    # Regression: a string slice ".loc['2020-05':]" on the yearly index dropped the label "2020".
    inputs, ac, t_cell = synthetic()
    tables, t_ref = kpi_tables(inputs, ac, t_cell, CONFIG, weighted_gamma(CONFIG))
    assert list(tables["kpi_yearly"].index) == ["2020"]
    assert list(tables["kpi_monthly"].index) == ["2020-06"]
    assert np.isclose(t_ref, 45.0)
    # No stops, no lost channels: the three availability numbers agree at 1.
    assert np.allclose(tables["kpi_yearly"][["availability_energy", "availability_time_raw"]], 1.0)


def test_write_tables_roundtrip(tmp_path):
    table = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index(["2020", "2021"], name="period"))
    sizes = write_tables({"demo": table}, tmp_path / "data")
    assert "demo" in sizes
    back = pd.read_parquet(tmp_path / "data" / "demo.parquet")
    pd.testing.assert_frame_equal(back, table)
