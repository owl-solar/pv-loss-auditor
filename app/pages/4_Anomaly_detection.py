"""Anomaly detection on residuals: how small a string fault can be found (SYNTHETIC) and what it does on real events."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # `common` lives in app/
import common  # noqa: E402
from pv_loss_auditor import dashboard_data as dd  # noqa: E402

st.set_page_config(page_title="Anomalies · PV Loss Auditor", page_icon="☀️", layout="wide")
summary = common.load("anomaly_summary")
by_cell = common.load("anomaly_by_cell")
tuning = common.load("anomaly_tuning")
events = common.load("anomaly_events")
real = common.load("anomaly_real_events")
excluded = common.load("anomaly_excluded_flag_share")
inv30 = common.load("anomaly_inv30")
noise = common.load("anomaly_noise").iloc[0]

BLUE, ORANGE, AQUA = (common.CAUSE_COLORS[k] for k in ("Downtime", "Running shortfall", "After inverters"))
DETECTOR_COLORS = {"forest, 4 features": BLUE, "forest, 2 features": AQUA, "rule (baseline)": ORANGE}
RULE, FOREST4 = "rule (baseline)", "forest, 4 features"

st.title("Anomaly detection on residuals")
common.synthetic_banner()
st.markdown(
    "**Residual:** measured / expected AC of each inverter, divided by the median of the running inverters at the "
    "same step. This removes season, weather, soiling and plant-wide steps (including the 2023 step) and keeps what is "
    "specific to one inverter. Healthy inverters sit at fixed offsets (0.96 … 1.05) from their peers, so the "
    "features are **deviations from the inverter's own 121-day norm**.  \n"
    "**Detectors:** Isolation Forest on four features, Isolation Forest on two, and a plain rule (daily median below "
    "*k* robust sigmas, sigma taken from the REAL data). **Threshold:** chosen on 2020-2021 so that at most 1 % of "
    "clean inverter-days are false alarms, **reported on 2022-2023**. An event counts as detected with ≥ 3 flagged days."
)

test_summary = summary[summary["criterion"] == "far"].set_index("detector")
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Rule: events found",
    f"{test_summary.loc[RULE, 'recall_events']:.0%}",
    help=f"Precision {test_summary.loc[RULE, 'precision']:.2f} at a false-alarm rate "
    f"{test_summary.loc[RULE, 'far']:.2%} of clean inverter-days (test years, SYNTHETIC).",
)
forest_lo = test_summary.loc[["forest, 4 features", "forest, 2 features"], "recall_events"]
c2.metric(
    "Forests: events found",
    f"{forest_lo.min():.0%} … {forest_lo.max():.0%}",
    help="Precision "
    + " … ".join(
        f"{test_summary.loc[d, 'precision']:.2f}" for d in ("forest, 4 features", "forest, 2 features")
    )
    + ".",
)
c3.metric("Detection limit", "≈ 10 strings", help="About 6 % of an inverter's 170 strings.")
c4.metric(
    "Noise of the real data",
    f"{noise['robust_sigma'] * 100:.2f} %",
    help=f"Robust sigma of the daily deviation; one string is {noise['one_string_pct']:.2f} % of an inverter, "
    f"so 3 sigma = {noise['three_sigma_strings']:.1f} strings.",
)
st.caption(
    f"Test years: {int(noise['events_test'])} SYNTHETIC events and {int(noise['clean_days_test']):,} clean inverter-days "
    f"(of {int(noise['events'])} events in all years)."
)

# --- detection by size -----------------------------------------------------------------------
st.header("How small a fault can be found")
duration = st.radio(
    "Fault duration (SYNTHETIC)",
    ["all", 21, 60],
    horizontal=True,
    format_func=lambda v: v if v == "all" else f"{v} days",
)
rate, counts = dd.detection_rate(by_cell, "far", None if duration == "all" else duration)
fig = go.Figure()
for name, color in DETECTOR_COLORS.items():
    fig.add_scatter(
        x=[str(s) for s in rate.index],
        y=rate[name],
        mode="lines+markers",
        name=name,
        line=dict(color=color, width=2.5),
        marker=dict(color=color, size=10, line=dict(color=common.SURFACE, width=2)),
        customdata=counts[name],
        hovertemplate="%{x} strings: %{y:.0%} of %{customdata} events<extra>" + name + "</extra>",
    )
fig.update_yaxes(title="share of events detected", tickformat=".0%", range=[-0.03, 1.05])
fig.update_xaxes(title="lost strings of 170 per inverter (SYNTHETIC)", type="category")
st.plotly_chart(
    common.style(fig, "Detection rate by fault size, test years 2022-2023 (SYNTHETIC faults)", height=380),
    width="stretch",
)
small = (
    by_cell[(by_cell["criterion"] == "far") & (by_cell["strings"] <= 5)]
    .groupby("detector")[["detected", "events"]]
    .sum()
)
st.caption(
    "Each point rests on 1-7 events per fault size and duration, pooled here; read the curve, not single "
    f"values. Of the {int(small.loc[RULE, 'events'])} test events with 1-5 lost strings the rule found "
    f"{int(small.loc[RULE, 'detected'])}, the forests {int(small.loc[FOREST4, 'detected'])} and "
    f"{int(small.loc['forest, 2 features', 'detected'])}. A single string ({noise['one_string_pct']:.2f} % of an "
    f"inverter) is below the noise of the real data ({noise['robust_sigma'] * 100:.2f} %). The idea that noisy "
    "features dilute the forest is **not supported**: the forest on two features is no better."
)

with st.expander("Detected / events per fault size and duration (chosen detector)"):
    name = st.selectbox("Detector", list(DETECTOR_COLORS), index=2)
    t = by_cell[(by_cell["criterion"] == "far") & (by_cell["detector"] == name)]
    table = (
        (t["detected"].astype(str) + " / " + t["events"].astype(str))
        .set_axis(pd.MultiIndex.from_frame(t[["strings", "duration_days"]]))
        .unstack("duration_days")
    )
    table.columns = [f"{c} days" for c in table.columns]
    st.dataframe(table.rename_axis("lost strings"), width="stretch")

# --- why this criterion --------------------------------------------------------------------------
st.header("Why a false-alarm budget")
view = summary.assign(
    criterion=summary["criterion"].map({"far": "false-alarm budget ≤ 1 %", "f1": "best day-level F1"})
)
view = view[
    [
        "criterion",
        "detector",
        "chosen",
        "precision",
        "recall_events",
        "recall_days",
        "false_alarm_days",
        "far",
    ]
]
view = view.rename(
    columns={
        "chosen": "parameter",
        "recall_events": "events found",
        "recall_days": "fault days found",
        "false_alarm_days": "false-alarm days",
        "far": "false-alarm rate",
    }
)
st.dataframe(
    view.style.format(
        {
            "precision": "{:.2f}",
            "events found": "{:.0%}",
            "fault days found": "{:.0%}",
            "false-alarm rate": "{:.1%}",
        }
    ),
    hide_index=True,
    width="stretch",
)
fault_share = events["duration_days"].sum() / noise["rows"]
f1 = summary[summary["criterion"] == "f1"]
grid = tuning[tuning["criterion"] == "f1"].groupby("detector")["param"].agg(["min", "max"])
at_edge = [d for d, p in zip(f1["detector"], f1["chosen"]) if p in (grid.loc[d, "min"], grid.loc[d, "max"])]
st.caption(
    f"Maximising F1 picked the edge of the parameter grid for {len(at_edge)} of {len(f1)} detectors and gave "
    f"{f1['far'].min():.0%} … {f1['far'].max():.0%} false alarms: it rewards finding long faults and ignores "
    "the cost of a false alarm. The budget criterion keeps the false alarms at or below 1 %. Precision also "
    f"depends on the share of injected days ({fault_share:.0%} of all inverter-days here), which is not the "
    "real fault rate of a plant."
)
with st.expander("Tuning table (2020-2021, SYNTHETIC) of the chosen criterion"):
    t = tuning[tuning["criterion"] == "far"][
        ["detector", "param", "precision", "recall_days", "recall_events", "false_alarm_days", "far"]
    ]
    st.dataframe(
        t.style.format(
            {"precision": "{:.3f}", "recall_days": "{:.3f}", "recall_events": "{:.3f}", "far": "{:.2%}"}
        ),
        hide_index=True,
        width="stretch",
    )

# --- real events -----------------------------------------------------------------------------
st.header("The same detectors on real events")
st.markdown(
    "These are **not** injected: two events found in the daily residuals of inverters 30 and 13, cause unknown "
    "(the data cannot say whether strings, a combiner or a sensor)."
)
fig = go.Figure()
fig.add_scatter(
    x=inv30.index,
    y=inv30["dev_rel_median"],
    mode="lines+markers",
    name="deviation from own norm",
    line=dict(color=common.INK_SECONDARY, width=1),
    marker=dict(size=4),
    hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
)
for col, name, y in (("flag rule (baseline)", RULE, -0.32), ("flag forest, 4 features", FOREST4, -0.35)):
    flagged = inv30.index[inv30[col].astype(bool)]
    fig.add_scatter(
        x=flagged,
        y=[y] * len(flagged),
        mode="markers",
        name="flagged: " + name,
        marker=dict(
            symbol="line-ns-open",
            size=12,
            color=DETECTOR_COLORS[name],
            line=dict(color=DETECTOR_COLORS[name], width=2),
        ),
        hovertemplate="%{x|%Y-%m-%d}<extra>flagged: " + name + "</extra>",
    )
fig.add_vrect(
    x0="2020-09-22",
    x1="2020-11-04",
    fillcolor=ORANGE,
    opacity=0.12,
    line_width=0,
    annotation_text="real event, 45 days",
    annotation_position="top left",
)
fig.update_yaxes(title="daily median rel, minus own norm")
st.plotly_chart(
    common.style(fig, "Inverter 30, real event 2020-09-22 … 2020-11-04", height=360), width="stretch"
)
show = real.assign(
    event=real["inverter"].map(dd.inverter_number)
    + ": "
    + real["start"].dt.strftime("%Y-%m-%d")
    + " … "
    + real["end"].dt.strftime("%Y-%m-%d")
)
show = show[["event", "days", RULE, "forest, 4 features", "forest, 2 features"]].rename(
    columns={"days": "days with a feature"}
)
st.dataframe(show, hide_index=True, width="stretch")
inside = (inv30.index >= "2020-09-22") & (inv30.index <= "2020-11-04")
outside_forest = inv30.index[~inside & inv30["flag forest, 4 features"].astype(bool)]
outside_rule = inv30.index[~inside & inv30["flag rule (baseline)"].astype(bool)]
before = inv30.loc["2020-08-21":"2020-09-21", "dev_rel_median"]
clean_days_tuning = int(
    tuning[(tuning["criterion"] == "far") & (tuning["detector"] == FOREST4)]["clean_days"].iloc[0]
)
st.info(
    f"**Outside the labelled window.** Before the event, from 2020-08-21 to 2020-09-21, the deviation of inverter 30 is "
    f"{before.min():+.2f} … {before.max():+.2f} (median {before.median():+.2f}), a month of *over*-performance against its "
    f"peers. The forest flags {len(outside_forest)} days outside the event window "
    f"({outside_forest.min():%Y-%m-%d} … {outside_forest.max():%Y-%m-%d}) and only "
    f"{int(inv30.loc[inside, 'flag forest, 4 features'].sum())} inside it; the rule flags {len(outside_rule)} days outside. "
    "Those days are labelled *clean*, so they count as false alarms. The cause is unknown. The own-norm window is "
    "unlikely to be the reason (121 days against a 45-day event: at most 37 % of its days lie in the event, so its "
    "median stays at the healthy level), and the event window in the config, "
    f"which starts on 2020-09-22, is probably too narrow. The effect on the reported numbers is small: "
    f"{len(outside_forest)} days are {len(outside_forest) / clean_days_tuning:.1%} of the {clean_days_tuning:,} clean "
    "inverter-days of the tuning years, against a budget of 1 %.",
    icon="🔍",
)
lines = [
    f"inverter {dd.inverter_number(r.inverter)}: rule {r.rule_baseline} of {r.days} days, forests "
    f"{r.forest_4_features} / {r.forest_2_features}"
    for r in real.rename(
        columns=lambda c: c.replace(", ", "_").replace(" ", "_").replace("(", "").replace(")", "")
    ).itertuples()
]
st.caption(
    "Flagged days inside each real event: " + "; ".join(lines) + ". A 61-day norm window absorbed the 45-day "
    "event of inverter 30 (the window was widened to 121 days for this reason). The two steps of this event "
    "(-14 % from 2020-09-22, -21 % from 2020-10-28) are listed in the config; the detectors do not know them."
)

# --- limits ----------------------------------------------------------------------------------
st.header("Limits to quote with any number")
st.warning(
    "- The faults are **SYNTHETIC**, abrupt, constant and on the AC side; real faults can be gradual or "
    "intermittent.\n"
    f"- The test has {int(noise['events_test'])} events in 14 cells (1-7 events each): rates are coarse.\n"
    "- Precision depends on the share of injected days, not on a real fault rate.\n"
    "- The own norm absorbs slow drift and faults longer than about 60 days.\n"
    "- The 5-minute localisation level was not built; the detectors work on daily features.\n"
    "- The 2023 step is common to the whole plant and cannot appear in a peer-normalised residual.\n"
    "- Plant-wide stops, real events and a buffer of 30 days around them are excluded from the injection.",
    icon="⚠️",
)
