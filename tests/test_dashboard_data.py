"""Tests of the dashboard aggregations on a small hand-made monthly waterfall (SYNTHETIC numbers)."""

import pandas as pd
import pytest

from pv_loss_auditor import dashboard_data as dd
from pv_loss_auditor.losses import STEPS


def monthly_table():
    """Three months in two years. Every step is 1 000 kWh except: downtime 5 000 (2020-06),
    running_shortfall 2 000 (2021-01), channel_credit -1 000 (a credit is negative)."""
    index = pd.Index(["2020-05", "2020-06", "2021-01"], name="period")
    t = pd.DataFrame({step: 1000.0 for step in STEPS}, index=index)
    t.loc["2020-06", "downtime"] = 5000.0
    t.loc["2021-01", "running_shortfall"] = 2000.0
    t["channel_credit"] = -1000.0
    t["nominal_kwh"] = 100000.0
    t["expected_kwh"] = 60000.0
    t["metered_kwh"] = t["nominal_kwh"] - t[STEPS].sum(axis=1)  # closes by construction
    return t


def test_yearly_sums_months_per_calendar_year():
    y = dd.yearly(monthly_table())
    assert list(y.index) == ["2020", "2021"]
    assert y.loc["2020", "downtime"] == 6000.0 and y.loc["2021", "downtime"] == 1000.0
    assert y.loc["2020", "nominal_kwh"] == 200000.0


def test_causes_follow_the_presentation_groups():
    c = dd.causes_table(monthly_table(), per="year")
    # Downtime group = downtime + unknown_ac + channel_credit; 2020: (1000+5000) + 2000 - 2000
    assert c.loc["2020", "Downtime"] == 6000.0
    assert c.loc["2021", "Running shortfall"] == 2000.0
    # Design group = temperature + dc_losses + inverter + clipping = 4 x 1000 per month
    assert c.loc["2020", "Design"] == 8000.0
    # After inverters group = after_inverters + reconciliation = 2 x 1000 per month
    assert c.loc["2021", "After inverters"] == 2000.0


def test_causes_add_up_to_nominal_minus_metered():
    m = monthly_table()
    for per in ("year", "total"):
        c = dd.causes_table(m, per=per)
        losses = c[["Design", "Downtime", "Running shortfall", "After inverters"]].sum(axis=1)
        assert (losses + c["metered_kwh"]).round(6).eq(c["nominal_kwh"]).all()


def test_audit_total_prices_downtime_and_shortfall_only():
    r = dd.audit_total(monthly_table(), usd_per_mwh=30.0)
    # downtime group: 7000 over three months; running shortfall: 1000 + 1000 + 2000 = 4000
    assert r["gwh"] == pytest.approx(11000 / 1e6)
    assert r["usd"] == pytest.approx(11000 / 1000 * 30.0)  # 11 MWh x 30 USD/MWh = 330 USD


def test_waterfall_steps_close_and_keep_the_order():
    steps = dd.waterfall_steps(monthly_table())
    assert list(steps["label"]) == ["nominal"] + STEPS + ["metered"]
    assert steps.loc[steps.label == "channel_credit", "kwh"].iloc[0] == -3000.0
    assert dd.waterfall_steps(monthly_table(), year=2021)["kwh"].iloc[0] == 100000.0


def test_waterfall_steps_refuse_an_open_balance():
    m = monthly_table()
    m.loc["2020-05", "metered_kwh"] += 50.0
    with pytest.raises(ValueError):
        dd.waterfall_steps(m)


def test_scale_units():
    kwh = pd.DataFrame({"Downtime": [2_000_000.0, 1_000_000.0]}, index=["2020", "2021"])
    nominal = pd.Series([40_000_000.0, 50_000_000.0], index=kwh.index)
    gwh, title, _ = dd.scale(kwh, nominal, "GWh", 30.0)
    assert gwh["Downtime"].tolist() == [2.0, 1.0] and title == "GWh"
    usd, _, _ = dd.scale(kwh, nominal, "USD thousand", 30.0)
    assert usd["Downtime"].tolist() == [60.0, 30.0]  # 2 000 MWh x 30 USD = 60 000 USD
    share, _, _ = dd.scale(kwh, nominal, "% of nominal energy", 30.0)
    assert share["Downtime"].tolist() == [5.0, 2.0]
    with pytest.raises(ValueError):
        dd.scale(kwh, nominal, "EUR", 30.0)


def test_waterfall_bars_walk_down_and_up_and_end_at_metered():
    steps = pd.DataFrame(
        {
            "label": ["nominal", "temperature", "channel_credit", "downtime", "metered"],
            "kind": ["total", "loss", "loss", "loss", "total"],
        }
    )
    # 100 - 10 (loss) + 2 (a credit is negative) - 5 = 87
    bars = dd.waterfall_bars(steps, [100.0, 10.0, -2.0, 5.0, 87.0])
    assert bars["start"].tolist() == [0.0, 100.0, 90.0, 92.0, 0.0]
    assert bars["end"].tolist() == [100.0, 90.0, 92.0, 87.0, 87.0]
    assert bars["base"].tolist() == [0.0, 90.0, 90.0, 87.0, 0.0]
    assert bars["length"].tolist() == [100.0, 10.0, 2.0, 5.0, 87.0]
    assert bars["cause"].tolist() == ["Total", "Design", "Downtime", "Downtime", "Total"]
    with pytest.raises(ValueError):
        dd.waterfall_bars(steps, [100.0, 10.0, -2.0, 5.0, 80.0])  # does not end at metered


def test_every_step_has_a_cause_and_a_label():
    assert set(dd.STEP_CAUSE) == set(STEPS) == set(dd.STEP_LABELS)


def test_monthly_cause_share_is_percent_of_nominal():
    share = dd.monthly_cause_share(monthly_table())
    # 2020-05: downtime group = 1000 + 1000 - 1000 = 1000 kWh of 100 000 nominal = 1 %
    assert share.loc["2020-05", "Downtime"] == pytest.approx(1.0)
    assert share.loc["2020-06", "Downtime"] == pytest.approx(5.0)


def test_failure_groups_join_pairs_into_components():
    pairs = pd.DataFrame(
        {
            "a": ["p_ac_inv17_kw", "p_ac_inv12_kw", "p_ac_inv35_kw", "p_ac_inv05_kw"],
            "b": ["p_ac_inv18_kw", "p_ac_inv17_kw", "p_ac_inv36_kw", "p_ac_inv06_kw"],
            "jaccard": [1.0, 0.97, 0.99, 0.90],
        }
    )  # 05+06 is below the 0.95 cut
    days = pd.date_range("2022-01-01", periods=4, freq="D")
    cols = ["p_ac_inv12_kw", "p_ac_inv17_kw", "p_ac_inv18_kw", "p_ac_inv35_kw", "p_ac_inv36_kw"]
    down = pd.DataFrame(0.0, index=days, columns=cols)
    down.loc[days[:3], ["p_ac_inv12_kw", "p_ac_inv17_kw", "p_ac_inv18_kw"]] = 1.0  # 3 days together
    down.loc[days[0], ["p_ac_inv35_kw", "p_ac_inv36_kw"]] = 1.0
    g = dd.failure_groups(pairs, down)
    assert g["group"].tolist() == ["12+17+18", "35+36"]
    assert g["inverters"].tolist() == [3, 2]
    assert g["min_jaccard"].tolist() == [0.97, 0.99]
    assert g["days_all_down"].tolist() == [3, 1]


def test_meter_verdict_has_a_gap_between_the_two_readings():
    assert dd.meter_verdict(1.006) == "stops confirmed by the meter"
    assert dd.meter_verdict(0.80) == "partly lost data channels"
    assert dd.meter_verdict(0.967) == "mixed"


def test_rolling_median_is_centred_and_needs_min_periods():
    days = pd.date_range("2021-01-01", periods=9, freq="D")
    s = pd.Series([1.0, 1.0, 1.0, 9.0, 1.0, 1.0, 1.0, 5.0, 5.0], index=days)
    out = dd.rolling_median(s, window=5, min_periods=3)
    assert out.iloc[3] == 1.0  # the outlier 9 does not move the median of 5 days
    assert out.iloc[0] == 1.0  # edge: 3 values (1, 1, 1) satisfy min_periods
    assert out.iloc[7] == 3.0  # window cut by the end of the series: [1, 1, 5, 5] -> (1+5)/2
    gappy = s.copy()
    gappy.iloc[:3] = float("nan")  # window of position 1 holds one valid value (9) < 3
    assert pd.isna(dd.rolling_median(gappy, window=5, min_periods=3).iloc[1])


def test_detection_rate_pools_durations_and_keeps_the_event_counts():
    by_cell = pd.DataFrame(
        {
            "strings": [5, 5, 5, 5],
            "duration_days": [21, 60, 21, 60],
            "detected": [0, 2, 1, 1],
            "events": [2, 2, 2, 2],
            "criterion": ["far"] * 4,
            "detector": ["rule", "rule", "forest", "forest"],
        }
    )
    rate, n = dd.detection_rate(by_cell)
    assert rate.loc[5, "rule"] == 0.5 and rate.loc[5, "forest"] == 0.5  # 2 of 4 and 2 of 4
    assert n.loc[5, "rule"] == 4
    rate60, n60 = dd.detection_rate(by_cell, duration=60)
    assert rate60.loc[5, "rule"] == 1.0 and rate60.loc[5, "forest"] == 0.5 and n60.loc[5, "rule"] == 2
    assert dd.detection_rate(by_cell, criterion="f1")[0].empty  # no rows for another criterion


def test_dashboard_modules_do_not_import_the_heavy_stack():
    # The deployed app installs only app/requirements.txt (no pvlib, pvanalytics, boto3, scipy ...).
    # If a dashboard module started to import them, the public app would fail on start.
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "import sys, pv_loss_auditor.dashboard_data, pv_loss_auditor.config, pv_loss_auditor.loss_groups; "
        "print([m for m in ('pvlib', 'pvanalytics', 'boto3', 'scipy', 'sklearn', 'rdtools') if m in sys.modules])"
    )
    root = Path(__file__).resolve().parent.parent
    env = {**os.environ, "PYTHONPATH": str(root / "src")}  # works without `pip install -e .`
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
