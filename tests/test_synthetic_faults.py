"""Tests of the SYNTHETIC fault injection (all data here are synthetic, for testing the code)."""

import copy

import numpy as np
import pandas as pd

from pv_loss_auditor.config import load_config
from pv_loss_auditor.synthetic_faults import eligible_starts, inject_faults, plan_faults

CONFIG = load_config("configs/system_9069.yaml")


def valid_frame(days=1400, n_inv=20):
    dates = pd.date_range("2021-01-01", periods=days, freq="D")
    cols = [f"p_ac_inv{i:02d}_kw" for i in range(1, n_inv + 1)]
    return pd.DataFrame(True, index=dates, columns=cols)


def test_plan_is_reproducible_and_respects_the_rules():
    valid = valid_frame()
    a, b = plan_faults(valid, CONFIG), plan_faults(valid, CONFIG)
    pd.testing.assert_frame_equal(a, b)                          # same seed, same plan
    s = CONFIG["anomalies"]["synthetic"]
    assert len(a) == len(s["durations_days"]) * len(s["levels_strings"]) * s["events_per_level"]
    assert a["is_synthetic"].all()
    assert ((a["end"] - a["start"]).dt.days + 1 == a["duration_days"]).all()
    assert set(a["duration_days"]) == set(s["durations_days"])
    for _, g in a.groupby("inverter"):                           # no overlap within an inverter
        g = g.sort_values("start")
        assert (g["start"].iloc[1:].to_numpy() > g["end"].iloc[:-1].to_numpy()).all()


def test_excluded_and_invalid_days_are_avoided():
    valid = valid_frame()
    valid.loc["2021-03-01":"2021-03-31", "p_ac_inv02_kw"] = False   # no data for inv02
    cfg = copy.deepcopy(CONFIG)
    cfg["anomalies"]["synthetic"]["exclude_plant"] = [{"start": "2021-06-01", "end": "2021-06-10"}]
    cfg["anomalies"]["synthetic"]["exclude_inverter"] = []
    ok = eligible_starts(valid, cfg, 21)
    w, b = 21, cfg["anomalies"]["synthetic"]["buffer_days"]
    # no start whose window, widened by the buffer, touches the excluded block
    blocked_starts = ok.loc[pd.Timestamp("2021-06-01") - pd.Timedelta(days=w - 1 + b):
                            pd.Timestamp("2021-06-10") + pd.Timedelta(days=b)]
    assert not blocked_starts.any().any()
    assert ok.loc["2021-04-01"].all()                              # far away: allowed
    assert not ok.loc["2021-03-02", "p_ac_inv02_kw"]               # window contains missing days
    assert ok.loc["2021-03-02", "p_ac_inv01_kw"]                   # other inverters unaffected


def test_inject_scales_only_the_window_of_one_inverter():
    index = pd.date_range("2021-06-01 04:00", periods=288 * 3, freq="5min", tz="UTC")
    ac = pd.DataFrame({"p_ac_inv01_kw": 500.0, "p_ac_inv02_kw": 500.0}, index=index)
    ac.iloc[10, 0] = np.nan
    ev = pd.DataFrame([{"inverter": "p_ac_inv01_kw", "start": pd.Timestamp("2021-06-02"),
                        "end": pd.Timestamp("2021-06-02"), "strings": 17, "loss_fraction": 0.1}])
    out = inject_faults(ac, ev)
    day = index.tz_convert("America/New_York").normalize().tz_localize(None)
    hit = day == pd.Timestamp("2021-06-02")
    assert np.allclose(out.loc[hit, "p_ac_inv01_kw"], 450.0)
    assert np.allclose(out.loc[~hit, "p_ac_inv01_kw"].dropna(), 500.0)
    assert np.allclose(out["p_ac_inv02_kw"], 500.0)
    assert np.isnan(out.iloc[10, 0])                               # NaN stays NaN
    assert np.allclose(ac["p_ac_inv01_kw"].dropna(), 500.0)        # input untouched
