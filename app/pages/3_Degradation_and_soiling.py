"""Degradation, soiling and external weather: what explains the shortfall of the running inverters."""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # `common` lives in app/
import common  # noqa: E402
from src import dashboard_data as dd  # noqa: E402

st.set_page_config(page_title="Degradation · PV Loss Auditor", page_icon="☀️", layout="wide")
daily = common.load("daily_series")
yoy = common.load("degradation_yoy")
grid = common.load("degradation_filters")
soil = common.load("soiling")
ext_stats = common.load("external_poa_stats")
ext_ratio = common.load("external_poa_ratio")
pr_src = common.load("pr_by_poa_source")
split = common.load("shortfall_split")
aoi = common.load("aoi_by_b0")

BLUE, ORANGE, AQUA = (common.CAUSE_COLORS[k] for k in ("Downtime", "Running shortfall", "After inverters"))
GRAY = common.CAUSE_COLORS["Design"]

st.title("Degradation, soiling and external weather")
st.markdown("Everything here is about the **running inverters** only (stopped inverters are downtime, already "
            "in the waterfall). The yardstick is *normalized energy* = measured AC / PVWatts expected AC, "
            "and the temperature-corrected PR (C); the two series agree within 0.1 pp.")

kept = int(daily["energy_norm"].notna().sum())
pre = grid[grid["period"].str.startswith("to")]["rd_pct_yr"]
full = grid[grid["period"] == "full"]["rd_pct_yr"]
s2 = soil[(soil["cleaning_info"] == "rain >= 2 mm + jumps") & (soil["period"] == "full")].iloc[0]
b05 = float(aoi.loc[aoi["b0"] == 0.05, "aoi_loss_pct"].iloc[0])
c1, c2, c3, c4 = st.columns(4)
c1.metric("Degradation before 2023", "≈ −1 %/yr (± 0.5)",
          help=f"YoY over a grid of day filters: {pre.min():.2f} ... {pre.max():.2f} %/yr. RdTools' own interval "
               "(± 0.3) does not include the choice of filters. Never quote a single decimal.")
c2.metric("Including 2023", f"{full.max():.2f} … {full.min():.2f} %/yr",
          help="This is a step in 2023, not ageing (see the open question below).")
c3.metric("Soiling", f"≈ {s2['loss_pct']:.1f} %",
          help=f"RdTools SRR with rain ≥ 2 mm as cleaning; only {int(s2['valid'])} of {int(s2['intervals'])} "
               "intervals are valid: an order of magnitude, not a measurement.")
c4.metric("Reflection (AOI)", f"≤ {b05:.1f} %", help="Upper bound; ASHRAE b0 = 0.05.")

# --- normalized energy ---------------------------------------------------------------------
st.header("Normalized energy of the running inverters")
roll = dd.rolling_median(daily["energy_norm"], window=61, min_periods=20)
fig = go.Figure()
fig.add_scatter(x=daily.index, y=daily["energy_norm"], mode="markers", name="daily",
                marker=dict(color=GRAY, size=4, opacity=0.6), hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra></extra>")
fig.add_scatter(x=roll.index, y=roll, mode="lines", name="61-day median", line=dict(color=BLUE, width=2.5),
                hovertemplate="%{x|%Y-%m-%d}: %{y:.3f}<extra>61-day median</extra>")
fig.add_vline(x=pd.Timestamp("2023-01-01"), line=dict(color=common.INK_SECONDARY, width=1, dash="dot"))
fig.update_yaxes(title="measured / expected")
st.plotly_chart(common.style(fig, "Daily measured / expected energy, running inverters", height=360), width="stretch")
medians = daily["energy_norm"].groupby(daily.index.year).median()
st.caption(f"Days kept: {kept} of {len(daily)} (≥ 90 % usable inverter-steps, ≥ 50 % of them running, ≥ 2 kWh/m² "
           "POA; these thresholds are my assumptions). Yearly medians: "
           + " / ".join(f"{v:.3f}" for v in medians) + " for 2020 / 2021 / 2022 / 2023. Dotted line: 1 January 2023.")

# --- degradation ---------------------------------------------------------------------------
st.header("Degradation (RdTools year-on-year)")
left, right = st.columns(2)
fig = go.Figure()
for period, label, color in (("to 2022-12-31", "to 2022-12", BLUE), ("full", "with 2023", ORANGE)):
    g = grid[grid["period"] == period]
    fig.add_scatter(x=g["rd_pct_yr"], y=[label] * len(g), mode="markers", name=label,
                    marker=dict(color=color, size=11, line=dict(color=common.SURFACE, width=2)),
                    customdata=g[["min_poa", "min_running"]],
                    hovertemplate="%{x:.2f} %/yr<br>min POA %{customdata[0]} kWh/m², min running %{customdata[1]}"
                                  "<extra></extra>")
fig.update_yaxes(title=None)
fig.update_xaxes(title="YoY degradation, %/yr")
left.plotly_chart(common.style(fig, "Same data, 12 day-filter settings each", height=300), width="stretch")
view = yoy.assign(series=yoy["series"].map({"energy_norm": "normalized energy", "pr_wc": "PR (C)"}))
view = view.rename(columns={"rd_pct_yr": "rate, %/yr", "ci_low": "CI low", "ci_high": "CI high"})
right.dataframe(view.style.format({"rate, %/yr": "{:.2f}", "CI low": "{:.2f}", "CI high": "{:.2f}"}),
                hide_index=True, width="stretch")
st.caption(f"Before 2023 the rate depends on the day filters ({pre.min():.2f} … {pre.max():.2f} %/yr here) by more "
           "than RdTools' own interval, so state it as **about −1 %/yr ± 0.5**. Including 2023 it is "
           f"{full.max():.2f} … {full.min():.2f} %/yr: a step, not ageing. Cells 04 / 10 against pyranometer 02B "
           "show −0.6 / −1.0 %/yr (notebook 02; not separable from 0); ±1 %/yr stays in the uncertainty of the rate.")

# --- soiling -------------------------------------------------------------------------------
st.header("Soiling (RdTools SRR)")
sv = soil[soil["period"] == "full"].copy()
sv["err_minus"] = sv["loss_pct"] - (1 - sv["ci_high"]) * 100
sv["err_plus"] = (1 - sv["ci_low"]) * 100 - sv["loss_pct"]
no_rain = sv["cleaning_info"] == "upward jumps only"
fig = go.Figure(go.Bar(
    y=sv["cleaning_info"], x=sv["loss_pct"], orientation="h", marker_color=[GRAY if n else BLUE for n in no_rain],
    marker_line=dict(color=common.SURFACE, width=2),
    error_x=dict(type="data", symmetric=False, array=sv["err_plus"], arrayminus=sv["err_minus"], color=common.INK_SECONDARY),
    text=[f"{v:.1f} %  ({int(a)} of {int(b)} intervals valid)" for v, a, b in zip(sv["loss_pct"], sv["valid"], sv["intervals"])],
    textposition="outside", cliponaxis=False,
    hovertemplate="%{y}: %{x:.2f} %<extra></extra>"))
fig.update_yaxes(autorange="reversed", title=None)
fig.update_xaxes(title="average energy lost to soiling, %", range=[0, 4.8])
st.plotly_chart(common.style(fig, "Soiling loss by the cleaning information RdTools gets", height=320), width="stretch")
rain_rows = sv[sv["cleaning_info"].str.startswith("rain >= ") & sv["cleaning_info"].str.contains(r"^rain >= (?:1|2|5) mm")]
rain_days = (daily["rain_mm"] >= 2).groupby(daily.index.year).sum()
st.caption(f"Gray: no rain data, cleanings are inferred from upward jumps: {sv.loc[no_rain, 'loss_pct'].iloc[0]:.1f} % "
           "is an **overestimate** (noise taken as cleanings). Blue: ERA5 rain above the threshold counts as a "
           f"cleaning; the rain threshold is an assumption, so it is varied ({rain_rows['loss_pct'].min():.1f} … "
           f"{rain_rows['loss_pct'].max():.1f} % for 1 … 5 mm). Georgia has {rain_days.loc[2021:2023].mean():.0f} rain "
           "days ≥ 2 mm a year (2021-2023 mean). The value before 2023 is the same, so soiling does **not** explain "
           "the 2023 step.")

# --- external weather ----------------------------------------------------------------------
st.header("Plant sensors against Open-Meteo reanalysis")
st.markdown("ERA5 and ECMWF IFS (archive API, explicit `models=`) transposed to 20°/180° with Perez, against the mean "
            "of reference cells 04 / 10. These are **reanalysis, not satellite data** (the satellite API has no "
            "coverage at this site). Open-Meteo hourly values are means of the *preceding* hour.")
st.dataframe(ext_stats.rename(columns={"source": "source", "hours": "hours > 200 W/m²", "mbe_pct": "MBE, %",
                                       "nrmse_pct": "hourly nRMSE, %", "r": "r"}).style.format(
    {"MBE, %": "{:+.2f}", "hourly nRMSE, %": "{:.1f}", "r": "{:.2f}", "hours > 200 W/m²": "{:.0f}"}),
    hide_index=True, width="stretch")
left, right = st.columns(2)
fig = go.Figure()
for name, color, label in (("era5", BLUE, "ERA5"), ("ifs", ORANGE, "IFS")):
    fig.add_scatter(x=ext_ratio.index, y=ext_ratio[name], mode="markers", name=label + " monthly",
                    marker=dict(color=color, size=5, opacity=0.5), hovertemplate="%{x}: %{y:.3f}<extra>" + label + "</extra>")
    fig.add_scatter(x=ext_ratio.index, y=ext_ratio[name + "_12m"], mode="lines", name=label + " 12-month",
                    line=dict(color=color, width=2.5), hovertemplate="%{x}: %{y:.3f}<extra>" + label + " 12-month</extra>")
fig.update_yaxes(title="external / cells (insolation)")
left.plotly_chart(common.style(fig, "Ratio to the reference cells", height=340), width="stretch")
fig = go.Figure()
# Dots, not bars: the axis does not start at zero (it would hide 3-4 pp differences), which is
# fine for positions but would distort the lengths of bars.
for name, color, label in (("plant", common.INK_SECONDARY, "plant cells"), ("era5", BLUE, "ERA5"), ("ifs", ORANGE, "IFS")):
    fig.add_scatter(x=pr_src.index.astype(str), y=pr_src[name], mode="markers+text", name=label,
                    marker=dict(color=color, size=12, line=dict(color=common.SURFACE, width=2)),
                    text=pr_src[name], texttemplate="%{text:.3f}", textposition="middle right",
                    textfont=dict(color=common.INK), cliponaxis=False,
                    hovertemplate="%{x}: %{y:.3f}<extra>" + label + "</extra>")
fig.update_yaxes(title="raw PR (variant A), running inverters", range=[0.64, 0.79])
fig.update_xaxes(type="category", range=[-0.4, 3.9])
right.plotly_chart(common.style(fig, "PR with each irradiance source", height=340), width="stretch")
months = ext_ratio.index.str[5:].astype(int)
winter, summer = ext_ratio[months.isin([11, 12, 1, 2])], ext_ratio[months.isin([6, 7, 8])]
gap = pr_src[["era5", "ifs"]].sub(pr_src["plant"], axis=0) * 100
st.caption(
    f"Open-Meteo reads POA {ext_stats['mbe_pct'].min():.1f} … {ext_stats['mbe_pct'].max():.1f} % above the cells at "
    f"hours > 200 W/m², hourly nRMSE {ext_stats['nrmse_pct'].min():.0f} %: too noisy for hour-level anomaly work. "
    f"Its raw PR is {-gap.max().max():.0f} … {-gap.min().min():.0f} pp below the cells, more than any year-to-year "
    "change: a plant without sensors would look worse by that much. The 12-month ERA5 ratio is flat, so the "
    "cells did **not** drift and the 2023 step is not a sensor effect. IFS rises, probably ECMWF model versions "
    f"(hypothesis). By season the monthly ERA5 ratio is {winter['era5'].min():.2f} … {winter['era5'].max():.2f} in "
    f"Nov-Feb and {summer['era5'].min():.2f} … {summer['era5'].max():.2f} in Jun-Aug: either the reanalysis or the "
    "cells misread low sun, the data cannot say which.")

# --- shortfall split -----------------------------------------------------------------------
st.header("Where the shortfall of the running inverters goes")
yrs = split.index.astype(str)
fig = go.Figure()
parts = [("soiling_pct", "Soiling (estimate)", BLUE, None),
         ("unexplained_min_pct", "Unexplained (at least)", GRAY, None),
         ("aoi_upper_bound_pct", "AOI upper bound (may be unexplained)", GRAY, "/")]
for col, label, color, pattern in parts:
    fig.add_bar(x=yrs, y=split[col], name=label, marker_color=color,
                marker_pattern_shape=pattern or "", marker_line=dict(color=common.SURFACE, width=2),
                text=split[col], texttemplate="%{text:.1f}", textposition="inside", insidetextanchor="middle",
                textfont=dict(color="white" if col == "soiling_pct" else common.INK),
                hovertemplate="%{x}: %{y:.1f} % of expected<extra>" + label + "</extra>")
fig.update_layout(barmode="stack", yaxis_title="% of expected energy")
st.plotly_chart(common.style(fig, "Shortfall = 1 − median daily measured / expected", height=380), width="stretch")
st.dataframe(split.rename(columns={"shortfall_pct": "shortfall, %", "soiling_pct": "soiling, %",
                                   "aoi_upper_bound_pct": "AOI ≤, %", "unexplained_min_pct": "unexplained min, %",
                                   "unexplained_max_pct": "unexplained max, %"}).style.format("{:.1f}"), width="stretch")
st.caption("The hatched part is the AOI **upper bound**: the reference cells also lose to reflection, so only the "
           "difference between module and cell enters measured / model. If it is zero, the unexplained share is the "
           "*max* column. The unexplained part mixes the error of the assumed 6.34 % design DC losses, low-light "
           "behaviour that linear PVWatts does not have, degradation since commissioning (age unknown) and sensor "
           "bias; the data cannot separate them, so it is shown as a range.")

# --- open questions ------------------------------------------------------------------------
st.header("What I could not explain")
st.info(
    "**The 2023 step, about 3 pp.** Measured / expected of the running inverters is about 3 percentage points lower "
    "in 2023 than in 2020-2022. It is not soiling (same value before 2023), not AOI (flat over years), not a sensor "
    "effect (cells against ERA5 flat), and downtime is excluded. It is common to the whole plant, so it disappears "
    "by construction in the peer-normalised residuals of the anomaly detection. Candidates: failed strings or "
    "combiner boxes in the running inverters, or plant-level changes. This is a hypothesis list, not a conclusion.\n\n"
    "**The PR dip in November-December** (PR (C) 0.71-0.73): AOI does not explain it (the winter loss is only "
    "0.3-0.5 pp above summer). What is left: row shading at low sun and low-light behaviour (PVWatts is linear in "
    "irradiance, while measured / expected is only 0.71-0.83 below 200 W/m²). Not resolved.", icon="❓")
