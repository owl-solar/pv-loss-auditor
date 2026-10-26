"""Tests of the peer-normalised residuals on SYNTHETIC data (only for testing the code)."""

import copy

import numpy as np
import pandas as pd
import pytest

from pv_loss_auditor.anomalies import daily_features, deviation_from_own_norm, relative_residual
from pv_loss_auditor.config import load_config
from pv_loss_auditor.model import expected_power

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


# ---- Isolation Forest part (synthetic data) --------------------------------------------------

def _labelled_rows():
    dates = pd.date_range("2021-03-01", periods=10, freq="D")
    index = pd.MultiIndex.from_product([dates, ["p_ac_inv01_kw", "p_ac_inv02_kw"]],
                                       names=["date", "inverter"])
    events = pd.DataFrame([{"inverter": "p_ac_inv01_kw", "start": dates[2], "end": dates[5],
                            "strings": 10, "loss_fraction": 0.06}])
    return index, events


def test_day_labels_and_event_metrics():
    from pv_loss_auditor.anomalies import day_labels, evaluate_flags
    cfg = copy.deepcopy(CONFIG)
    cfg["anomalies"]["synthetic"]["exclude_plant"] = [{"start": "2021-03-09", "end": "2021-03-09"}]
    cfg["anomalies"]["synthetic"]["exclude_inverter"] = []
    index, events = _labelled_rows()
    labels = day_labels(index, events, cfg)
    assert (labels == "fault").sum() == 4                       # 4 days x inverter 01
    assert (labels == "excluded").sum() == 2                    # one plant-wide day x 2 inverters
    assert (labels == "clean").sum() == 20 - 4 - 2

    flag = pd.Series(False, index=index)
    flag[(index.get_level_values("date") <= "2021-03-05") & (index.get_level_values("inverter") == "p_ac_inv01_kw")] = True  # days 1-5
    flag[(index.get_level_values("date") == "2021-03-01") & (index.get_level_values("inverter") == "p_ac_inv02_kw")] = True   # false alarm
    flag[(index.get_level_values("date") == "2021-03-09")] = True                                                  # excluded: ignored
    summary, per_event = evaluate_flags(flag, labels, events, cfg)
    # flagged inside the fault: days 3, 4, 5 of 4 (03-03 .. 03-06) -> 3 true positives
    assert per_event["n_flagged"].iloc[0] == 3 and per_event["detected"].iloc[0]
    assert np.isclose(summary["recall_days"], 3 / 4)
    assert summary["false_alarm_days"] == 3                     # inv01 03-01, 03-02 and inv02 03-01
    assert np.isclose(summary["precision"], 3 / 6)


def test_forest_ranks_an_obvious_outlier_first():
    from pv_loss_auditor.anomalies import forest_scores
    rng = np.random.default_rng(0)
    x = pd.DataFrame(rng.normal(0, 0.01, size=(500, 4)), columns=["a", "b", "c", "d"])
    x.iloc[123] = [-0.3, -0.2, 0.5, 0.4]               # one point far from the cloud
    scores = forest_scores(x, CONFIG)
    assert scores.idxmax() == 123


def test_own_norm_features_removes_static_offsets():
    from pv_loss_auditor.anomalies import FEATURES, own_norm_features
    dates = pd.date_range("2021-01-01", periods=300, freq="D")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "inverter"])
    feat = pd.DataFrame({f: 0.0 for f in FEATURES}, index=index)
    feat.loc[(slice(None), "A"), "rel_median"] = 0.97     # static offset of inverter A
    feat.loc[(slice(None), "B"), "rel_median"] = 1.00
    dev = own_norm_features(feat, CONFIG)
    assert np.allclose(dev["dev_rel_median"], 0.0)
    assert list(dev.columns) == ["dev_" + f for f in FEATURES]


def test_tune_and_report_chooses_on_tuning_rows_and_reports_on_test_rows():
    from pv_loss_auditor.anomalies import tune_and_report
    dates = pd.date_range("2021-11-01", periods=120, freq="D")
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "inverter"])
    d = index.get_level_values("date")
    events = pd.DataFrame([
        {"inverter": "A", "start": pd.Timestamp("2021-12-01"), "end": pd.Timestamp("2021-12-10"), "strings": 10},
        {"inverter": "A", "start": pd.Timestamp("2022-02-01"), "end": pd.Timestamp("2022-02-10"), "strings": 10}])
    labels, score = pd.Series("clean", index=index), pd.Series(0.0, index=index)
    for e in events.itertuples():
        in_fault = (d >= e.start) & (d <= e.end)
        labels[in_fault], score[in_fault] = "fault", 1.0
    tune, test = d <= pd.Timestamp("2021-12-31"), d > pd.Timestamp("2021-12-31")
    table, best, summary, per_event, flag = tune_and_report(
        score, {"low": 0.5, "high": 2.0}, labels, tune, test, events.iloc[:1], events.iloc[1:], CONFIG)
    assert best == "low"                                  # "high" flags nothing, F1 = 0
    assert summary["precision"] == 1.0 and summary["recall_days"] == 1.0
    assert per_event["detected"].all()


def test_far_criterion_respects_the_false_alarm_budget():
    from pv_loss_auditor.anomalies import tune_and_report
    dates = pd.date_range("2021-01-01", periods=100, freq="D")
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "inverter"])
    d = index.get_level_values("date")
    labels = pd.Series("clean", index=index)
    event = pd.DataFrame([{"inverter": "A", "start": dates[10], "end": dates[19], "strings": 5}])
    labels[(d >= dates[10]) & (d <= dates[19])] = "fault"
    score = pd.Series(0.0, index=index)
    score[(d >= dates[10]) & (d <= dates[19])] = 1.0       # the fault scores 1.0
    score[(d >= dates[40]) & (d <= dates[59])] = 0.6       # 20 clean days with a medium score
    tune = np.ones(len(index), dtype=bool)
    thr = {"loose": 0.5, "strict": 0.9}
    # budget 10 % of 90 clean days = 9: "loose" flags 20 clean days -> too many; "strict" fits
    _, best, _, _, _ = tune_and_report(score, thr, labels, tune, tune, event, event, CONFIG,
                                       criterion="far", far_max=0.10)
    assert best == "strict"


def test_rule_score_uses_the_given_sigma_and_only_the_low_side():
    from pv_loss_auditor.anomalies import rule_score
    x = pd.DataFrame({"dev_rel_median": [0.0, -0.02, 0.02, -0.04]})
    s = rule_score(x, sigma=0.01)
    assert np.allclose(s.to_numpy(), [0.0, 2.0, -2.0, 4.0])      # loss = positive score, gain = negative
    own = rule_score(x)                                          # sigma from x itself
    sigma = 1.4826 * (x["dev_rel_median"] - x["dev_rel_median"].median()).abs().median()
    assert np.allclose(own.to_numpy(), -x["dev_rel_median"].to_numpy() / sigma)


# --- detector comparison (robust_sigma ... flags_in_windows), SYNTHETIC panel -----------------

def _panel(seed=0):
    """10 inverters x 730 days (2021-2022) of deviations ~ N(0, 0.007); each of 4 events (2 in
    each year) drops dev_rel_median by 0.15 for 30 days. Returns x, labels, events."""
    from pv_loss_auditor.anomalies import FEATURES
    dates = pd.date_range("2021-01-01", periods=730, freq="D")
    inv = [f"p_ac_inv{i:02d}_kw" for i in range(1, 11)]
    index = pd.MultiIndex.from_product([dates, inv], names=["date", "inverter"])
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({f"dev_{f}": rng.normal(0, 0.007, len(index)) for f in FEATURES}, index=index)
    events = pd.DataFrame([
        {"inverter": inv[i], "start": pd.Timestamp(s), "end": pd.Timestamp(s) + pd.Timedelta(days=29),
         "strings": 20}
        for i, s in enumerate(["2021-03-01", "2021-08-01", "2022-03-01", "2022-08-01"])])
    labels = pd.Series("clean", index=index)
    d, i = index.get_level_values("date"), index.get_level_values("inverter")
    for e in events.itertuples():
        hit = (i == e.inverter) & (d >= e.start) & (d <= e.end)
        labels[hit] = "fault"
        x.loc[hit, "dev_rel_median"] -= 0.15
        x.loc[hit, "dev_frac_below95"] += 0.5
    return x, labels, events


def test_robust_sigma_ignores_rare_outliers():
    from pv_loss_auditor.anomalies import robust_sigma
    d = pd.Series(np.random.default_rng(1).normal(0, 0.01, 2000))
    clean = robust_sigma(d)
    d.iloc[:40] = -0.5                                    # 2 % huge outliers
    assert clean == pytest.approx(0.01, rel=0.1)          # 1.4826 x MAD recovers the std
    assert robust_sigma(d) == pytest.approx(clean, rel=0.1)
    assert d.std() > 3 * clean                            # the plain std would explode


def test_run_detectors_finds_obvious_faults_and_returns_all_tables():
    from pv_loss_auditor.anomalies import (
        DETECTORS,
        detection_by_cell,
        flags_in_windows,
        robust_sigma,
        run_detectors,
        summary_table,
    )
    x, labels, events = _panel()
    sigma = robust_sigma(x["dev_rel_median"])
    results, split = run_detectors(x, labels, events, CONFIG, sigma, criteria=("far",))
    assert list(results["far"]) == DETECTORS
    assert len(split["ev_tune"]) == 2 and len(split["ev_test"]) == 2
    table = summary_table(results)
    assert len(table) == 3 and (table["far"] <= CONFIG["anomalies"]["forest"]["far_max"]).all()
    rule = table.set_index("detector").loc["rule (baseline)"]
    # -0.15 is 21 sigma: every fault day is found. Precision is not asserted at a fixed level: the
    # 1 % false-alarm budget allows up to ~36 false days against 60 fault days (precision >= 0.62).
    assert rule["recall_events"] == 1.0 and rule["recall_days"] == 1.0
    assert rule["precision"] >= 60 / (60 + CONFIG["anomalies"]["forest"]["far_max"] * rule["clean_days"])
    cells = detection_by_cell(results)
    assert set(cells.columns) == {"strings", "duration_days", "detected", "events", "criterion", "detector"}
    assert cells[cells.detector == "rule (baseline)"][["detected", "events"]].sum().tolist() == [2, 2]

    flags = {n: v[4] for n, v in results["far"].items()}
    win = flags_in_windows(flags, [("p_ac_inv03_kw", "2022-03-01", "2022-03-30")])
    assert win.loc[0, "days"] == 30 and win.loc[0, "rule (baseline)"] == 30


def test_flags_in_windows_counts_only_the_named_inverter_and_dates():
    from pv_loss_auditor.anomalies import flags_in_windows
    dates = pd.date_range("2021-01-01", periods=10, freq="D")
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "inverter"])
    flag = pd.Series(True, index=index)
    flag[(index.get_level_values("inverter") == "B")] = False
    out = flags_in_windows({"d": flag}, [("A", "2021-01-03", "2021-01-06"), ("B", "2021-01-03", "2021-01-06")])
    assert out["days"].tolist() == [4, 4] and out["d"].tolist() == [4, 0]
