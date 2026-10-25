"""Tests of the peer-normalised residuals on SYNTHETIC data (only for testing the code)."""

import numpy as np
import pandas as pd

from src.anomalies import daily_features, deviation_from_own_norm, relative_residual
from src.config import load_config
from src.model import expected_power

CONFIG = load_config("configs/system_9069.yaml")
N_INV = 12


def synthetic(steps=288):
    """One UTC day at constant POA 600 W/m2 (below clipping), T_cell 40 C, 12 inverters at 0.90."""
    index = pd.date_range("2021-06-01 12:00", periods=steps, freq="5min", tz="UTC")
    inputs = pd.DataFrame({"poa_wm2": 600.0, "daylight": True}, index=index)
    t_cell = pd.Series(40.0, index=index)
    exp = expected_power(inputs["poa_wm2"], t_cell, CONFIG)["p_ac_kw"]
    ac = pd.DataFrame({f"p_ac_inv{i:02d}_kw": 0.90 * exp for i in range(1, N_INV + 1)})
    return inputs, ac, t_cell


def test_common_bias_is_removed():
    inputs, ac, t_cell = synthetic()
    rel = relative_residual(inputs, ac, t_cell, CONFIG)
    assert np.allclose(rel.stack(), 1.0)           # all inverters at 0.90 -> rel = 1


def test_one_inverter_loss_shows_and_does_not_move_peers():
    inputs, ac, t_cell = synthetic()
    ac["p_ac_inv03_kw"] *= 0.95                    # 5 % loss in one inverter
    rel = relative_residual(inputs, ac, t_cell, CONFIG)
    assert np.allclose(rel["p_ac_inv03_kw"], 0.95)
    assert np.allclose(rel.drop(columns="p_ac_inv03_kw"), 1.0)   # median ignores one outlier


def test_stopped_and_unknown_inverters_are_excluded():
    inputs, ac, t_cell = synthetic()
    ac["p_ac_inv01_kw"] = 0.0                      # down
    ac["p_ac_inv02_kw"] = np.nan                   # unknown
    rel = relative_residual(inputs, ac, t_cell, CONFIG)
    assert rel[["p_ac_inv01_kw", "p_ac_inv02_kw"]].isna().all().all()
    assert np.allclose(rel.drop(columns=["p_ac_inv01_kw", "p_ac_inv02_kw"]).stack(), 1.0)


def test_too_few_peers_gives_nan():
    inputs, ac, t_cell = synthetic()
    ac.iloc[:, 3:] = np.nan                        # 3 inverters left, min_peers is 10
    assert relative_residual(inputs, ac, t_cell, CONFIG).isna().all().all()


def test_low_light_and_clipping_steps_are_skipped():
    inputs, ac, t_cell = synthetic()
    inputs.loc[inputs.index[:100], "poa_wm2"] = 100.0      # below min_poa_wm2
    inputs.loc[inputs.index[100:150], "poa_wm2"] = 1000.0  # model near the rating
    rel = relative_residual(inputs, ac, t_cell, CONFIG)
    assert rel.iloc[:150].isna().all().all()
    assert rel.iloc[150:].notna().all().all()


def test_daily_features_and_own_norm():
    inputs, ac, t_cell = synthetic()
    ac["p_ac_inv03_kw"] *= 0.95
    feat = daily_features(relative_residual(inputs, ac, t_cell, CONFIG), CONFIG)
    assert np.isclose(feat.loc[(pd.Timestamp("2021-06-01"), "p_ac_inv03_kw"), "rel_median"], 0.95)
    assert feat["n_steps"].min() >= CONFIG["anomalies"]["min_day_steps"]

    # 90 days: inverter A sits at a fixed 0.97 (not a fault), inverter B drops by 5 % on day 80
    dates = pd.date_range("2021-01-01", periods=90, freq="D")
    daily = pd.DataFrame({"A": 0.97, "B": 1.0}, index=dates)
    daily.loc[dates[80:], "B"] = 0.95
    dev = deviation_from_own_norm(daily)
    assert dev["A"].abs().max() < 1e-9             # static offset is not an anomaly
    assert np.isclose(dev["B"].iloc[85], -0.05)    # the change against its own norm is


def test_daily_features_count_only_valid_steps():
    inputs, ac, t_cell = synthetic()
    ac["p_ac_inv03_kw"] *= 0.90                    # rel = 0.90 on every valid step
    ac.loc[ac.index[88:], "p_ac_inv03_kw"] = np.nan    # 200 of 288 steps unknown; 88 left on 06-01
    feat = daily_features(relative_residual(inputs, ac, t_cell, CONFIG), CONFIG)
    row = feat.loc[(pd.Timestamp("2021-06-01"), "p_ac_inv03_kw")]
    assert row["n_steps"] == 88                    # NaN steps are not counted
    assert row["frac_below95"] == 1.0              # and do not dilute the share
