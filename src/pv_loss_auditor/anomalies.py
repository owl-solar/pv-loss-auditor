"""Residuals of each inverter against the model and against its peers (Phase 5).

r = measured AC / expected AC is ~0.9 for a healthy inverter (the model is not calibrated, see
model.py). Plant-wide effects (season, soiling, the 2023 step, sensor bias) move r of ALL
inverters together, so r is divided by the median r of the running inverters at the same step:

    rel = r_i / median_j(r_j)        ~1.0 for a healthy inverter, 0.97 if it lost 3 % to its peers.

This removes everything the inverters share and keeps what is specific to one inverter, which is
what a failed string or combiner does. The price: a loss that hits every inverter equally is
invisible here (it is covered by the waterfall and Phase 4).
"""

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from pv_loss_auditor.kpi import inverter_states
from pv_loss_auditor.model import expected_power
from pv_loss_auditor.validation import check_poa, require_aligned, require_nonempty, require_positive

log = logging.getLogger(__name__)

Config = Mapping[str, Any]


def relative_residual(inputs: pd.DataFrame, ac: pd.DataFrame, t_cell: pd.Series, config: Config) -> pd.DataFrame:
    """Peer-normalised residual rel per inverter-step (DataFrame like `ac`, NaN = not usable).

    A step counts if it is daylight, POA > min_poa_wm2, T_cell known and the model is below
    max_expected_fraction x rating (no clipping). Stopped inverters (down) are excluded: they are
    downtime, already in the waterfall. The peer median needs at least min_peers inverters.
    """
    require_nonempty(inputs, "inputs")
    require_aligned(inputs, ac, ("inputs", "ac"))
    require_aligned(inputs, t_cell, ("inputs", "t_cell"))
    check_poa(inputs["poa_wm2"], config)
    a = config["anomalies"]
    rating = require_positive(config["inverters"]["ac_kw"][0]["ac_kw"], "inverter AC rating")
    poa = inputs["poa_wm2"]
    expected = expected_power(poa, t_cell, config)["p_ac_kw"]
    step_ok = (inputs["daylight"] & (poa > a["min_poa_wm2"]) & t_cell.notna()
               & (expected < a["max_expected_fraction"] * rating))
    _, down = inverter_states(ac, poa, config)
    usable = ac.notna() & ~down & step_ok.to_numpy()[:, None]

    r = ac.div(expected, axis=0).where(usable)
    peer = r.median(axis=1).where(r.notna().sum(axis=1) >= a["min_peers"])
    return r.div(peer, axis=0)


def daily_features(rel: pd.DataFrame, config: Config) -> pd.DataFrame:
    """One row per (local date, inverter) with summary features of rel.

    rel_median   typical level of the day (a lost string shifts it);
    rel_p10      lower tail (partial-day problems);
    frac_below95 share of steps with rel < 0.95 (persistent vs short);
    rel_spread   p90 - p10 (unstable behaviour);
    n_steps      number of valid steps; rows below min_day_steps are dropped.
    """
    day = rel.index.tz_convert("America/New_York").normalize().tz_localize(None)
    # pandas >= 3 keeps NaN in stack(); drop them, or n_steps and frac_below95 count unusable steps
    long = rel.stack().dropna().rename("rel").reset_index()
    long.columns = ["time", "inverter", "rel"]
    long["date"] = day[rel.index.get_indexer(long["time"])]
    g = long.groupby(["date", "inverter"])["rel"]
    out = pd.DataFrame({
        "rel_median": g.median(),
        "rel_p10": g.quantile(0.10),
        "frac_below95": g.apply(lambda s: (s < 0.95).mean()),
        "rel_spread": g.quantile(0.90) - g.quantile(0.10),
        "n_steps": g.size(),
    })
    return out[out["n_steps"] >= config["anomalies"]["min_day_steps"]]


def deviation_from_own_norm(daily_rel_median: pd.DataFrame, window_days: int = 61,
                            min_days: int = 20) -> pd.DataFrame:
    """Daily rel_median minus the inverter's own rolling median (centred, +-30 days).

    Healthy inverters sit at fixed offsets from their peers (0.96 ... 1.05, stable over years:
    different mismatch, shading, sensor position). Those offsets are not faults, so the detector
    looks at the change against the inverter's own recent level. Input: DataFrame, rows =
    local dates, columns = inverters (as `daily_features(...)["rel_median"].unstack()`).
    Limit: a slow real degradation of one inverter is absorbed into its norm.
    """
    daily = daily_rel_median.asfreq("D")           # regular grid, so 61 rows = 61 days
    norm = daily.rolling(window_days, center=True, min_periods=min_days).median()
    return daily - norm


# ---------------------------------------------------------------------------------------------
# Isolation Forest on the daily features (Phase 5)
# ---------------------------------------------------------------------------------------------

FEATURES = ["rel_median", "rel_p10", "frac_below95", "rel_spread"]


def own_norm_features(feat: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Daily features as deviations from each inverter's own rolling norm.

    The norm window comes from the config (anomalies.norm_window_days).

    feat: output of daily_features (index = date, inverter). Returns a frame with the same index
    and columns dev_<feature>; rows where the own norm is not defined (too few days) are dropped.
    """
    out = {}
    for name in FEATURES:
        wide = feat[name].unstack()                       # dates x inverters
        dev = deviation_from_own_norm(wide, config["anomalies"]["norm_window_days"],
                                      config["anomalies"]["norm_min_days"])
        out["dev_" + name] = dev.stack().dropna()                              # pandas 3 keeps NaN
    return pd.DataFrame(out).dropna()


def forest_scores(x: pd.DataFrame, config: Config) -> pd.Series:
    """Anomaly score per row of x (higher = more unusual); unsupervised, no labels used.

    Isolation Forest isolates a point by random splits; unusual points need fewer splits. Both
    directions count (a loss and a gain are both unusual). The score is -score_samples of
    scikit-learn, so that larger = more anomalous.
    """
    from sklearn.ensemble import IsolationForest
    require_nonempty(x, "feature table")
    f = config["anomalies"]["forest"]
    log.info("Isolation Forest: %d rows x %d features, %d trees", len(x), x.shape[1], f["n_estimators"])
    model = IsolationForest(n_estimators=f["n_estimators"], random_state=f["random_state"])
    model.fit(x)
    return pd.Series(-model.score_samples(x), index=x.index, name="score")


def day_labels(index: pd.MultiIndex, events: pd.DataFrame, config: Config) -> pd.Series:
    """Label of each (date, inverter) row: 'fault' (inside a synthetic window), 'excluded'
    (inside a real event excluded from the injection: not clean, not a label) or 'clean'."""
    from pv_loss_auditor.synthetic_faults import _blocked_days
    dates = index.get_level_values("date")
    inverters = index.get_level_values("inverter")
    lab = pd.Series("clean", index=index)
    blocked = _blocked_days(pd.DatetimeIndex(sorted(set(dates))), sorted(set(inverters)), config)
    # Row-wise lookup of the blocked frame
    pos_d = blocked.index.get_indexer(dates)
    pos_i = blocked.columns.get_indexer(inverters)
    lab[blocked.to_numpy()[pos_d, pos_i]] = "excluded"
    for e in events.itertuples():
        hit = (inverters == e.inverter) & (dates >= e.start) & (dates <= e.end)
        lab[hit] = "fault"
    return lab


def evaluate_flags(flag, labels, events, config, min_flag_days=None):
    """Day-level and event-level metrics for a boolean flag Series (same index as labels).

    Day level: precision / recall over 'fault' and 'clean' rows ('excluded' rows are ignored).
    Event level: an event is detected if at least min_flag_days of its days are flagged.
    Returns (summary dict, per-event DataFrame with n_flagged, first_flag_day and detected).
    """
    n_min = min_flag_days or config["anomalies"]["forest"]["min_flag_days"]
    use = labels != "excluded"
    fault = labels == "fault"
    tp = int((flag & fault & use).sum())
    fp = int((flag & ~fault & use).sum())
    fn = int((~flag & fault & use).sum())
    dates = labels.index.get_level_values("date")
    inverters = labels.index.get_level_values("inverter")
    rows = []
    for e in events.itertuples():
        hit = (inverters == e.inverter) & (dates >= e.start) & (dates <= e.end)
        flagged_dates = dates[(flag & hit).to_numpy()]
        first = (flagged_dates.min() - e.start).days if len(flagged_dates) else np.nan
        rows.append({"inverter": e.inverter, "start": e.start, "strings": e.strings,
                     "duration_days": (e.end - e.start).days + 1,
                     "n_days": int(hit.sum()), "n_flagged": int((flag & hit).sum()),
                     "first_flag_day": first})   # days from the fault start to the first flag
    per_event = pd.DataFrame(rows)
    per_event["detected"] = per_event["n_flagged"] >= n_min
    summary = {
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall_days": tp / (tp + fn) if tp + fn else float("nan"),
        "recall_events": float(per_event["detected"].mean()) if len(per_event) else float("nan"),
        "false_alarm_days": fp, "clean_days": int((~fault & use).sum()),
    }
    p, r = summary["precision"], summary["recall_days"]
    summary["f1_days"] = 2 * p * r / (p + r) if p + r else 0.0
    return summary, per_event


def rule_score(x: pd.DataFrame, sigma: float | None = None) -> pd.Series:
    """Baseline without ML: how many robust sigmas dev_rel_median lies BELOW zero (higher = worse).

    sigma = 1.4826 x MAD of dev_rel_median. Pass the sigma of the REAL data (no injected faults):
    when 16 % of the rows carry synthetic faults, the MAD of the injected table is inflated
    (0.0094 instead of 0.0068 here) and every threshold in sigmas silently becomes stricter.
    Without the argument the sigma of `x` itself is used. Labels are never used. A loss of
    strings lowers the median of rel, so only the low side counts; a rise is not flagged.
    """
    d = x["dev_rel_median"]
    if sigma is None:
        sigma = 1.4826 * (d - d.median()).abs().median()
    require_positive(sigma, "robust sigma of the residual (flat residual series?)")
    return (-d / sigma).rename("score")


def split_by_year(index, events, labels, tune_until):
    """Masks for tuning (date <= tune_until) and test rows; events by where they lie entirely.

    Events that straddle tune_until belong to neither set and their fault days are relabelled
    'excluded', so neither the tuning nor the report sees half an event.
    Returns (tune_mask, test_mask, events_tune, events_test, labels_clean_split).
    """
    cut = pd.Timestamp(tune_until)
    dates = index.get_level_values("date")
    events_tune = events[events["end"] <= cut]
    events_test = events[events["start"] > cut]
    straddle = events[(events["start"] <= cut) & (events["end"] > cut)]
    lab = labels.copy()
    inverters = index.get_level_values("inverter")
    for e in straddle.itertuples():
        lab[(inverters == e.inverter) & (dates >= e.start) & (dates <= e.end)] = "excluded"
    return dates <= cut, dates > cut, events_tune, events_test, lab


def tune_and_report(score, thresholds, labels, tune, test, events_tune, events_test, config,
                    criterion="f1", far_max=0.01):
    """Pick a threshold on the tuning rows, report on the test rows.

    thresholds: {parameter value: score threshold} (for the forest the parameter is the
    contamination and the threshold its score quantile; for the rule k and k itself).
    criterion:
      "f1"  best day-level F1 (rewards recall of long faults; precision depends on how many
            faults were injected, which is not the real fault rate);
      "far" false-alarm budget: among thresholds with false_alarm_days / clean_days <= far_max
            take the best day-level recall; if none fits, the strictest threshold.
    Returns (tuning table, chosen parameter, test summary, per-event test table, flag Series).
    """
    rows = []
    for param, thr in thresholds.items():
        summary, _ = evaluate_flags(score[tune] >= thr, labels[tune], events_tune, config)
        rows.append({"param": param, "threshold": thr, **summary})
    table = pd.DataFrame(rows)
    table["far"] = table["false_alarm_days"] / table["clean_days"]
    if criterion == "f1":
        best = table.loc[table["f1_days"].idxmax()]
    elif criterion == "far":
        within = table[table["far"] <= far_max]
        best = (within.loc[within["recall_days"].idxmax()] if len(within)
                else table.loc[table["far"].idxmin()])
    else:
        raise ValueError(f"unknown criterion {criterion!r}")
    flag = score >= best["threshold"]
    summary_test, per_event = evaluate_flags(flag[test], labels[test], events_test, config)
    return table, best["param"], summary_test, per_event, flag


# ---------------------------------------------------------------------------------------------
# Detector comparison on the daily features (notebook 05, used by the dashboard export)
# ---------------------------------------------------------------------------------------------

DETECTORS = ["forest, 4 features", "forest, 2 features", "rule (baseline)"]


def robust_sigma(d: pd.Series) -> float:
    """1.4826 x MAD: the standard deviation of a normal sample, insensitive to rare large values.

    Compute it on the REAL data (no injected faults), see rule_score.
    """
    return 1.4826 * (d - d.median()).abs().median()


def detector_scores(x, sigma, config):
    """Score of every row of x for the three detectors (higher = more anomalous).

    The two forests differ in the features they see (all four deviations, or the two named in
    anomalies.forest.features_small); the rule uses the sigma of the real data.
    """
    small = config["anomalies"]["forest"]["features_small"]
    return {
        DETECTORS[0]: forest_scores(x, config),
        DETECTORS[1]: forest_scores(x[small], config),
        DETECTORS[2]: rule_score(x, sigma=sigma),
    }


def run_detectors(x, labels, events, config, sigma, criteria=("far", "f1")):
    """Tune every detector on the early years and report on the later years, per criterion.

    Returns (results, split): results[criterion][detector] = the tuple of tune_and_report
    (tuning table, chosen parameter, test summary, per-event test table, flag Series);
    split = dict with the relabelled labels `lab` and the event tables `ev_tune`, `ev_test`.
    """
    fo = config["anomalies"]["forest"]
    tune, test, ev_tune, ev_test, lab = split_by_year(x.index, events, labels, fo["tune_until"])
    scores = detector_scores(x, sigma, config)
    results = {}
    for criterion in criteria:
        results[criterion] = {}
        for name, score in scores.items():
            if name.startswith("forest"):
                thresholds = {c: score.quantile(1 - c) for c in fo["contamination_grid"]}
            else:
                thresholds = {k: float(k) for k in config["anomalies"]["rule"]["k_grid"]}
            results[criterion][name] = tune_and_report(
                score, thresholds, lab, tune, test, ev_tune, ev_test, config,
                criterion=criterion, far_max=fo["far_max"])
    return results, {"lab": lab, "ev_tune": ev_tune, "ev_test": ev_test}


def summary_table(results):
    """Test-year summary, one row per (criterion, detector): the chosen parameter, precision,
    recall (days, events), false-alarm days and rate, and the median day of the first flag."""
    rows = []
    for criterion, by_detector in results.items():
        for name, (_, best, s_te, per, _) in by_detector.items():
            rows.append({
                "criterion": criterion, "detector": name, "chosen": best,
                "precision": s_te["precision"], "recall_days": s_te["recall_days"],
                "recall_events": s_te["recall_events"], "false_alarm_days": s_te["false_alarm_days"],
                "clean_days": s_te["clean_days"], "far": s_te["false_alarm_days"] / s_te["clean_days"],
                "median_first_flag_day": per[per.detected].first_flag_day.median()})
    return pd.DataFrame(rows)


def detection_by_cell(results):
    """Detected events per (criterion, detector, lost strings, fault duration) on the test years."""
    parts = []
    for criterion, by_detector in results.items():
        for name, (_, _, _, per, _) in by_detector.items():
            g = per.groupby(["strings", "duration_days"]).detected.agg(["sum", "size"])
            g = g.rename(columns={"sum": "detected", "size": "events"}).astype(int).reset_index()
            parts.append(g.assign(criterion=criterion, detector=name))
    return pd.concat(parts, ignore_index=True)


def flags_in_windows(flags, windows):
    """Flagged days of each detector inside given (inverter, start, end) windows (REAL events).

    flags: {detector: boolean Series on the (date, inverter) index}. One row per window with the
    number of rows (days) of x inside it and one column per detector.
    """
    rows = []
    for inverter, start, end in windows:
        row = {"inverter": inverter, "start": pd.Timestamp(start), "end": pd.Timestamp(end)}
        for name, flag in flags.items():
            dates = flag.index.get_level_values("date")
            inside = ((flag.index.get_level_values("inverter") == inverter)
                      & (dates >= start) & (dates <= end))
            row["days"] = int(inside.sum())
            row[name] = int(flag[inside].sum())
        rows.append(row)
    return pd.DataFrame(rows)
