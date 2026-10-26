"""Downtime and outages: which inverters were down when, who fails together, what the meter says."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # `common` lives in app/
import common  # noqa: E402
from pv_loss_auditor import dashboard_data as dd  # noqa: E402

st.set_page_config(page_title="Downtime · PV Loss Auditor", page_icon="☀️", layout="wide")
down = common.load("down_days")
ratio = common.load("inverter_meter_ratio")["ratio"]
pairs = common.load("down_pairs")
episodes = common.load("outage_episodes")

st.title("Downtime and outages")
st.markdown(
    "An inverter is **down** on a local day if its AC power is below 1 % of its rating on more than half of "
    "the daylight steps with irradiance above 50 W/m². A day with too few known steps is *unknown*. "
    "A zero in the files is not always a stop: the **meter** decides (see the third panel)."
)

# --- the map -------------------------------------------------------------------------------
episodes = episodes.assign(label=lambda e: e["start"].astype(str) + " ... " + e["end"].astype(str))
zoom = st.selectbox("Zoom to an episode", ["whole period"] + episodes["label"].tolist())
x_range = None
if zoom != "whole period":
    row = episodes[episodes["label"] == zoom].iloc[0]
    pad = pd.Timedelta(days=10)
    x_range = [pd.Timestamp(row["start"]) - pad, pd.Timestamp(row["end"]) + pad]

numbers = [dd.inverter_number(c) for c in down.columns]
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.18, 0.22])
fig.add_trace(
    go.Heatmap(
        x=down.index,
        y=numbers,
        z=down.T.to_numpy(),
        zmin=0,
        zmax=1,
        showscale=False,
        colorscale=[[0, "#e3e2de"], [1, common.CAUSE_COLORS["Downtime"]]],
        xgap=0,
        ygap=1,
        hovertemplate="%{x|%Y-%m-%d}, inverter %{y}: %{z}<extra>1 = down, 0 = running</extra>",
    ),
    row=1,
    col=1,
)
n_down = down.sum(axis=1)
fig.add_trace(
    go.Scatter(
        x=down.index,
        y=n_down,
        mode="lines",
        line=dict(color=common.CAUSE_COLORS["Downtime"], width=2),
        fill="tozeroy",
        fillcolor="rgba(42,120,214,0.18)",
        name="inverters down",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.0f} of 40 down<extra></extra>",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=ratio.index,
        y=ratio,
        mode="lines",
        line=dict(color=common.INK_SECONDARY, width=1.5),
        name="sum(inverters) / meter",
        hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>",
    ),
    row=3,
    col=1,
)
fig.add_hline(
    y=0.97, line=dict(color=common.CAUSE_COLORS["Running shortfall"], width=1.5, dash="dash"), row=3, col=1
)
fig.update_yaxes(
    title_text="inverter", type="category", autorange="reversed", row=1, col=1, tickvals=numbers[::3]
)
fig.update_yaxes(title_text="down", row=2, col=1)
fig.update_yaxes(title_text="inverters / meter", range=[0.4, 1.2], row=3, col=1)
fig = common.style(fig, "Down-state map of the 40 inverters, one cell per local day", height=720)
fig.update_layout(showlegend=False, margin=dict(t=60))
if x_range:
    fig.update_xaxes(range=x_range)
st.plotly_chart(fig, width="stretch")
st.caption(
    "Gray = running, blue = down, blank = unknown (too few valid steps). Third panel: daily "
    "sum of inverter energy over (meter × 1.022); the dashed line at 0.97 marks a channel-loss day, and the axis is "
    "cut at 0.4 (the plant-wide stop of 2022-05-23 falls below it). About 1.00 means the zeros are real stops; clearly below "
    "means the meter sees energy that the inverter channels do not report, so part of the zeros are **lost "
    "data channels**. Which inverter lies cannot be told from these files, so the correction is plant-level."
)

# --- episodes ------------------------------------------------------------------------------
st.header("Episodes and what the meter says")
view = pd.DataFrame(
    {
        "start": episodes["start"],
        "end": episodes["end"],
        "days": episodes["days"],
        "inverters stopped": episodes["n_stopped"],
        "median sum / meter": episodes["ratio_median"],
        "p5": episodes["ratio_p5"],
        "reading": episodes["ratio_median"].map(dd.meter_verdict),
    }
)
st.dataframe(
    view.style.format({"median sum / meter": "{:.3f}", "p5": "{:.3f}"}), hide_index=True, width="stretch"
)
st.caption(
    "Reading: median ≥ 0.97 stops confirmed by the meter, < 0.95 part of the zeros are lost data channels, "
    "in between mixed (the thresholds are mine; 0.97 is one minus the channel tolerance of the waterfall). "
    "The plant-wide stops of 2020-12-12, 2022-05-23, 2022-10-20 and the partial 2021-01-02 cannot be "
    "attributed to the grid or to the plant from the data; after each one whole groups stayed down. "
    "Inverter 01 reports 0 kW from late February to mid-April 2023 (inverters 01-06 down 2022-12 ... 2023-04)."
)
with st.expander("Which inverters (stopped / partly)"):
    st.dataframe(
        episodes[["label", "stopped", "partly"]].rename(columns={"label": "episode"}),
        hide_index=True,
        width="stretch",
    )

# --- groups --------------------------------------------------------------------------------
st.header("Inverters fail in groups, not alone")
groups = dd.failure_groups(pairs, down)
left, right = st.columns([3, 2])
fig = go.Figure(
    go.Bar(
        x=groups["days_all_down"],
        y=groups["group"],
        orientation="h",
        marker_color=common.CAUSE_COLORS["Downtime"],
        marker_line=dict(color=common.SURFACE, width=2),
        text=groups["days_all_down"],
        textposition="outside",
        hovertemplate="inverters %{y}: all down on %{x} days<extra></extra>",
    )
)
fig.update_yaxes(autorange="reversed", title="inverters", type="category")
fig.update_xaxes(title="days with all members down")
left.plotly_chart(common.style(fig, "Groups with Jaccard ≥ 0.95", height=300), width="stretch")
right.dataframe(
    groups.rename(
        columns={"group": "group", "min_jaccard": "weakest pair (Jaccard)", "days_all_down": "days all down"}
    ).style.format({"weakest pair (Jaccard)": "{:.3f}"}),
    hide_index=True,
    width="stretch",
)
st.caption(
    "Jaccard = days both inverters are down / days at least one is down (1.0 = always together). A shared "
    "step-up transformer skid or feeder is a **hypothesis**: it is not in the files."
)
