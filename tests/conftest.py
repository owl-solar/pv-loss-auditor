"""Shared helpers for the physics tests (synthetic plants with a known answer)."""

import copy

import numpy as np
import pandas as pd
import pytest

from pv_loss_auditor.config import load_config

CONFIG_PATH = "configs/system_9069.yaml"
FIXTURE_DIR = "tests/fixtures/9069_clean"


@pytest.fixture(scope="session")
def config():
    """The real 9069 config (read-only: tests that change it must use `plant_config`)."""
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def fixture_config():
    """The 9069 config pointing at the 14-day REAL fixture (tests/fixtures/9069_clean)."""
    cfg = load_config(CONFIG_PATH)
    cfg["clean"]["out_dir"] = FIXTURE_DIR
    return cfg


def plant_config(base, dc_kw, n_inverters=1, ac_kw=825.0):
    """Copy of the config for a small synthetic plant: P0 per inverter = dc_kw / n_inverters."""
    cfg = copy.deepcopy(base)
    cfg["capacity"]["dc_kw"] = dc_kw
    cfg["inverters"]["count"] = n_inverters
    cfg["inverters"]["ac_kw"] = [{"ids": "all", "ac_kw": ac_kw}]
    return cfg


def hourly_plant(poa, p_ac_kw, t_cell=25.0, start="2021-03-01 00:00", meter_kw=None):
    """inputs, ac, t_cell for ONE inverter on an hourly grid (1 h steps make kWh = kW)."""
    poa = pd.Series(np.asarray(poa, dtype=float))
    index = pd.date_range(start, periods=len(poa), freq="1h", tz="UTC")
    poa.index = index
    p_ac = pd.Series(np.broadcast_to(np.asarray(p_ac_kw, dtype=float), len(poa)).copy(), index=index)
    meter = p_ac if meter_kw is None else pd.Series(meter_kw, index=index, dtype=float)
    inputs = pd.DataFrame({"poa_wm2": poa, "daylight": True, "p_ac_meter_kw": meter})
    ac = pd.DataFrame({"p_ac_inv01_kw": p_ac})
    t = pd.Series(t_cell, index=index, dtype=float)
    return inputs, ac, t
