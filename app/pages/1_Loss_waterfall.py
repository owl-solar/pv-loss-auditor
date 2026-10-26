"""Loss waterfall: nameplate energy at the measured irradiance -> energy at the plant meter."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # `common` lives in app/
import common  # noqa: E402
from pv_loss_auditor import dashboard_data as dd

st.set_page_config(page_title="Loss waterfall · PV Loss Auditor", page_icon="☀️", layout="wide")
price = common.price_control()
monthly = common.load("waterfall_monthly")
sens = common.load("dc_loss_sensitivity")

st.title("Loss waterfall")
st.markdown(
    "Each step is the difference of two neighbouring cumulative quantities, so the steps add up to "
    "*nominal − metered* by construction (the export stops if any month does not close). "
    "Scope: 5-minute steps with daylight, irradiance, cell temperature and meter known."
)

years = sorted(set(monthly.index.str[:4]))
col_a, col_b = st.columns([1, 2])
period = col_a.selectbox(
    "Period", ["2020-05 ... 2023-11 (all)"] + years, help="2020 = May-December, 2023 = January-November."
)
unit = col_b.radio("Unit", dd.UNITS, horizontal=True, key="unit_waterfall")

year = None if period.startswith("2020-05") else period
steps = dd.waterfall_steps(monthly, year)
nominal = steps["kwh"].iloc[0]
scaled, axis_title, fmt = dd.scale(steps[["kwh"]], pd.Series(nominal, index=steps.index), unit, price)
bars = dd.waterfall_bars(steps, scaled["kwh"])
bars["name"] = bars["label"].map(dd.STEP_LABELS).fillna(bars["label"].str.capitalize())

fig = go.Figure()
for cause in ["Total", "Design", "Downtime", "Running shortfall", "After inverters"]:
    b = bars[bars["cause"] == cause]
    color = common.INK_SECONDARY if cause == "Total" else common.CAUSE_COLORS[cause]
    fig.add_bar(
        y=b["name"],
        x=b["length"],
        base=b["base"],
        orientation="h",
        name=cause,
        marker_color=color,
        marker_line=dict(color=common.SURFACE, width=2),
        customdata=b["value"],
        hovertemplate="%{y}: %{customdata:" + fmt + "} " + axis_title + "<extra></extra>",
        text=[f"{v:{fmt}}" for v in b["value"]],
        textposition="outside",
        cliponaxis=False,
        textfont=dict(color=common.INK),
    )
# Category order set explicitly: plotly otherwise orders rows by first appearance across the traces
# (the "Total" trace holds both ends), which put "Metered" in the second row.
fig.update_yaxes(autorange="reversed", title=None, categoryorder="array", categoryarray=list(bars["name"]))
# Room on the right for the outside labels (else the longest bar's label runs into the legend).
ends = bars[["start", "end"]]
fig.update_xaxes(title=axis_title, range=[min(0.0, ends.min().min() * 1.1), ends.max().max() * 1.15])
fig.update_layout(barmode="overlay")
fig = common.style(fig, f"Loss waterfall, {period}", height=520)
# After style(), which sets a horizontal legend that would cover the first bar: legend to the right.
fig.update_layout(legend=dict(orientation="v", x=1.02, y=1.0, yanchor="top"), margin=dict(r=170, t=60))
st.plotly_chart(fig, width="stretch")
st.caption(
    "Bars start at zero on the energy axis. A negative step is a gain: temperature below 25 °C, and the "
    "**channel credit**, which takes back energy that the meter saw but the inverter channels did not "
    "report (zeros written while the inverter ran), so it is not a loss. After the channel credit, "
    "downtime is only the stop energy the meter confirms."
)
if unit == "USD thousand":
    common.price_caveat()
common.design_loss_caveat()

with st.expander("Values as a table"):
    table = bars[["name", "cause", "value"]].rename(columns={"name": "step", "value": axis_title})
    common.table(table.style.format({axis_title: "{:" + fmt + "}"}), hide_index=True, width="stretch")

# --- how the picture changes with the assumed design losses ---------------------------------
st.header("How much depends on the assumed design losses")
st.markdown(
    "The 6.34 % design DC losses (LID 1.5, mismatch 2, wiring 1.5, connections 0.5, nameplate 1) are an "
    "assumption. Changing the total moves energy between *design* and *running shortfall*; downtime "
    "hardly moves, because it is valued at the expected power."
)
sens_view = (sens / 1e6).rename(
    columns={
        "dc_losses": "Design DC losses, GWh",
        "downtime": "Downtime (gross), GWh",
        "running_shortfall": "Running shortfall, GWh",
    }
)
sens_view.index = [f"{v:g} %" for v in sens.index]
sens_view.index.name = "assumed design DC losses"
common.table(sens_view.style.format("{:.1f}"), width="stretch")
st.caption(
    "Downtime here is gross (before the channel credit); the whole-period total of 43.2 GWh in the "
    "summary is downtime net of channel loss plus running shortfall at 6.34 %."
)

# --- losses by cause month by month ---------------------------------------------------------
st.header("Losses by cause, month by month")
share = dd.monthly_cause_share(monthly)
fig = go.Figure()
for cause in ["Downtime", "Running shortfall", "After inverters", "Design"]:
    fig.add_bar(
        x=share.index,
        y=share[cause],
        name=cause,
        marker_color=common.CAUSE_COLORS[cause],
        marker_line=dict(color=common.SURFACE, width=1),
        hovertemplate="%{x}: %{y:.1f} %<extra>" + cause + "</extra>",
    )
fig.update_layout(barmode="stack", yaxis_title="% of nominal energy")
st.plotly_chart(
    common.style(fig, "Share of nominal energy lost, by cause and month", height=380), width="stretch"
)
st.caption(
    "Downtime is the story of 2022-2023 (3 % of nominal energy in 2020-2021, 14 % in 2022, 18 % in 2023, "
    "net of channel loss). The running shortfall stays at 7.6-8.7 % of the plant's nominal energy in every "
    "year, but it is counted only on inverters that run, so in 2022-2023 the same share hides a worse "
    "ratio: measured / expected of the running inverters is about 3 pp lower in 2023 (not soiling, AOI or "
    "sensors: unexplained, see Degradation)."
)
