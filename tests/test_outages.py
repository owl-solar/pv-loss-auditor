"""Tests of the outage helpers on SYNTHETIC data (only for testing the code)."""

import numpy as np
import pandas as pd

from pv_loss_auditor.outages import episode_table, pair_overlap


def frame(dates, cols):
    return pd.DataFrame(0.0, index=dates, columns=cols)


def test_pair_overlap_finds_the_pair_that_fails_together():
    dates = pd.date_range("2021-01-01", periods=60, freq="D")
    d = frame(dates, ["p_ac_inv01_kw", "p_ac_inv02_kw", "p_ac_inv03_kw"])
    d.loc[dates[10:30], ["p_ac_inv01_kw", "p_ac_inv02_kw"]] = 1.0     # 01 and 02 stop together
    d.loc[dates[40:55], "p_ac_inv03_kw"] = 1.0                        # 03 alone
    out = pair_overlap(d, min_union=10)
    top = out.iloc[0]
    assert {top["a"], top["b"]} == {"p_ac_inv01_kw", "p_ac_inv02_kw"} and top["jaccard"] == 1.0
    pair_03 = out[(out.a == "p_ac_inv01_kw") & (out.b == "p_ac_inv03_kw")]
    assert pair_03["jaccard"].iloc[0] == 0.0


def test_pair_overlap_ignores_unknown_days():
    dates = pd.date_range("2021-01-01", periods=30, freq="D")
    d = pd.DataFrame({"p_ac_inv01_kw": 1.0, "p_ac_inv02_kw": np.nan}, index=dates)
    d.loc[dates[:15], "p_ac_inv02_kw"] = 1.0                          # known and down on 15 days
    out = pair_overlap(d, min_union=10)
    assert out.iloc[0]["union"] == 15 and out.iloc[0]["jaccard"] == 1.0


def test_episode_table_separates_stopped_from_partial_and_reads_the_meter_ratio():
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    d = frame(dates, ["p_ac_inv01_kw", "p_ac_inv02_kw", "p_ac_inv03_kw"])
    d["p_ac_inv01_kw"] = 1.0                                          # down all days
    d.loc[dates[:4], "p_ac_inv02_kw"] = 1.0                           # down 40 % of days
    ratio = pd.Series(0.8, index=dates)
    t = episode_table(d, ratio, [{"start": "2021-01-01", "end": "2021-01-10"}])
    assert t["stopped"].iloc[0] == "01" and t["partly"].iloc[0] == "02"
    assert np.isclose(t["ratio_median"].iloc[0], 0.8)
