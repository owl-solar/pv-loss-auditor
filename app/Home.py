"""Summary page: how much energy and money the plant lost, and the KPIs per year.

Run from the repository root:  streamlit run app/Home.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import common
from pv_loss_auditor import dashboard_data as dd

st.set_page_config(page_title="PV Loss Auditor", page_icon="☀️", layout="wide")
price = common.price_control()
monthly = common.load("waterfall_monthly")
kpi_year = common.load("kpi_yearly")
kpi_month = common.load("kpi_monthly")
info = common.meta()

st.title("PV Loss Auditor")
st.markdown(
    f"**{info['plant']}** · {info['dc_kw'] / 1000:.1f} MWp DC / {info['ac_kw'] / 1000:.0f} MW AC · "
    f"measured data 2020-05 ... 2023-11 (PVDAQ)"
)

# --- headline ------------------------------------------------------------------------------
audit = dd.audit_total(monthly, price)
low, base, high = dd.price_band(info["price_usd_per_mwh"])
band = (dd.audit_total(monthly, low)["usd"], dd.audit_total(monthly, high)["usd"])
total = dd.causes_table(monthly, per="total").iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Downtime + running shortfall",
    f"{audit['gwh']:.1f} GWh",
    help="The part of the gap an operator can act on. Design losses are not included.",
)
c2.metric(
    "Value at the chosen price",
    f"{audit['usd'] / 1e6:.2f} M USD",
    help=f"Band at {low:.2f} ... {high:.2f} USD/MWh: {band[0] / 1e6:.2f} ... {band[1] / 1e6:.2f} M USD",
)
y = kpi_year
c3.metric(
    "Availability 2023",
    f"{y.loc['2023', 'availability_energy']:.2f}",
    delta=f"{y.loc['2023', 'availability_energy'] - y.loc['2022', 'availability_energy']:+.2f} vs 2022",
    help="Energy-based, net of data-channel loss. Range: raw "
    f"{y.loc['2023', 'availability_time_raw']:.2f} (lower bound) ... "
    f"{y.loc['2023', 'availability_time_upper']:.2f} (upper bound).",
)
c4.metric(
    "PR 2023 (plant meter)",
    f"{y.loc['2023', 'pr_wc_meter']:.3f}",
    delta=f"{y.loc['2023', 'pr_wc_meter'] - y.loc['2020', 'pr_wc_meter']:+.3f} vs 2020",
    help="Temperature-corrected PR (variant C) at the revenue meter, including downtime.",
)
st.caption(
    f"Whole period, price {price:.2f} USD/MWh: band {band[0] / 1e6:.2f} ... {band[1] / 1e6:.2f} M USD "
    f"at {low:.2f} ... {high:.2f} USD/MWh. Total gap to nominal energy: "
    f"{(total['Design'] + total['Downtime'] + total['Running shortfall'] + total['After inverters']) / 1e6:.1f} GWh, "
    f"of which the plant as built (design) is {total['Design'] / 1e6:.1f} GWh."
)
common.price_caveat()

# --- losses by cause and year --------------------------------------------------------------
st.header("Where the energy went")
unit = st.radio(
    "Unit",
    dd.UNITS,
    horizontal=True,
    key="unit_summary",
    help="Years are not full years (2020 = May-Dec, 2023 = Jan-Nov): compare them as % of nominal energy.",
)
causes = dd.causes_table(monthly, per="year")
order = ["Downtime", "Running shortfall", "After inverters", "Design"]
values, axis_title, fmt = dd.scale(causes[order], causes["nominal_kwh"], unit, price)

fig = go.Figure()
min_label = 0.06 * values.sum(axis=1).max()  # a segment thinner than this gets no inside label
for cause in order:
    labels = [f"{v:{fmt}}" if v >= min_label else "" for v in values[cause]]
    fig.add_bar(
        x=values.index,
        y=values[cause],
        name=cause,
        marker_color=common.CAUSE_COLORS[cause],
        marker_line=dict(color=common.SURFACE, width=2),  # 2 px gap between segments
        text=labels,
        textposition="inside",
        insidetextanchor="middle",
        textangle=0,
        textfont=dict(color="white" if cause != "Design" else common.INK),
        hovertemplate="%{x}: %{y:" + fmt + "} " + axis_title + "<extra>" + cause + "</extra>",
    )
fig.update_layout(barmode="stack", yaxis_title=axis_title, xaxis_title=None)
st.plotly_chart(common.style(fig, "Loss by cause and calendar year"), width="stretch")
with st.expander("Values as a table (thin segments have no label in the chart)"):
    common.table(values.style.format("{:" + fmt + "}"), width="stretch")
common.design_loss_caveat()

# --- KPI trend -----------------------------------------------------------------------------
st.header("KPIs")
left, right = st.columns(2)
months = kpi_month.index

fig = go.Figure()
fig.add_scatter(
    x=months,
    y=kpi_month["availability_time_upper"],
    mode="lines",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip",
)
fig.add_scatter(
    x=months,
    y=kpi_month["availability_time_raw"],
    mode="lines",
    line=dict(width=0),
    fill="tonexty",
    fillcolor="rgba(42,120,214,0.18)",
    name="range raw ... upper bound",
    hoverinfo="skip",
)
fig.add_scatter(
    x=months,
    y=kpi_month["availability_energy"],
    mode="lines",
    name="energy-based (central)",
    line=dict(color=common.CAUSE_COLORS["Downtime"], width=2),
    hovertemplate="%{x}: %{y:.3f}<extra>availability</extra>",
)
fig.update_yaxes(range=[0.2, 1.02], title="availability")
left.plotly_chart(common.style(fig, "Availability, monthly", height=340), width="stretch")

fig = go.Figure()
fig.add_scatter(
    x=months,
    y=kpi_month["pr_wc_meter"],
    mode="lines",
    name="plant meter (with downtime)",
    line=dict(color=common.CAUSE_COLORS["Downtime"], width=2),
    hovertemplate="%{x}: %{y:.3f}<extra>PR meter</extra>",
)
fig.add_scatter(
    x=months,
    y=kpi_month["pr_wc_excl_down"],
    mode="lines",
    name="running inverters only",
    line=dict(color=common.CAUSE_COLORS["Running shortfall"], width=2),
    hovertemplate="%{x}: %{y:.3f}<extra>PR running</extra>",
)
fig.update_yaxes(title="PR (variant C)")
right.plotly_chart(
    common.style(fig, "Performance Ratio (temperature-corrected), monthly", height=340), width="stretch"
)

table = (
    kpi_year[
        [
            "pr_wc_meter",
            "pr_wc_excl_down",
            "availability_energy",
            "availability_time_raw",
            "availability_time_upper",
            "channel_days",
            "specific_yield_kwh_kwp",
        ]
    ]
    .rename(
        columns={
            "pr_wc_meter": "PR (C), meter",
            "pr_wc_excl_down": "PR (C), running inverters",
            "availability_energy": "availability (energy-based)",
            "availability_time_raw": "availability, lower bound",
            "availability_time_upper": "availability, upper bound",
            "channel_days": "channel-loss days",
            "specific_yield_kwh_kwp": "specific yield, kWh/kWp",
        }
    )
    .T
)  # years as columns: fits the page
whole = ["channel-loss days", "specific yield, kWh/kWp"]
common.table(table.style.format("{:.3f}").format("{:.0f}", subset=pd.IndexSlice[whole, :]), width="stretch")
st.caption(
    "Quote 2023 availability as **0.79**: the raw 0.69 counts lost data channels as stops (lower "
    "bound); from 2023-03 every day is a channel-loss day, so the upper bound is uninformative. "
    "PR with downtime comes from the plant meter; the inverter-based PR is understated by about "
    "7 pp in 2023. The specific yield of the inverters covers only the known channels."
)
