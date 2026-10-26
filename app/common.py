"""Shared pieces of the dashboard pages: data loading, price control, caveat banners, chart style.

Every number comes from the tables exported by src/pv_loss_auditor/export_dashboard.py (app/data/), so the app
needs neither the raw data nor the model. Caveats live here so that no page can show a USD figure
or a synthetic result without its warning.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))   # `pv_loss_auditor` is imported from src/ (no install needed)

DATA_DIR = Path(__file__).resolve().parent / "data"

# One colour per entity, the same on every page (data-viz skill: colour follows the entity).
# Blue / orange / aqua are slots 1-3 of the reference palette (validated, all-pairs). Design is
# neutral gray on purpose: it is the plant as built, not a loss to recover.
CAUSE_COLORS = {
    "Design": "#9a9a95",
    "Downtime": "#2a78d6",
    "Running shortfall": "#eb6834",
    "After inverters": "#1baf7a",
}
SURFACE = "#fcfcfb"
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e3e2de"


@st.cache_data
def load(name):
    """One exported table (parquet) by name."""
    return pd.read_parquet(DATA_DIR / f"{name}.parquet")


@st.cache_data
def meta():
    return json.loads((DATA_DIR / "meta.json").read_text(encoding="utf8"))


def price_control():
    """Sidebar slider for the flat energy price; returns USD/MWh.

    The range is the +-30 % band around the Georgia Power projection: sliding it shows how much
    of every USD figure is the price assumption and how little of it is the ranking of causes.
    """
    p = meta()["price_usd_per_mwh"]
    with st.sidebar:
        st.subheader("Price assumption")
        price = st.slider("Energy price, USD/MWh", min_value=round(p["low"], 2),
                          max_value=round(p["high"], 2), value=round(p["base"], 2), step=0.01,
                          key="price", help="Base 30.19 USD/MWh, band +-30 %.")
        st.caption("USD figures use this flat price. The ranking of causes in kWh does not depend on it.")
    return price


def price_caveat():
    """The limits of every USD figure, stated where the figure is shown."""
    p = meta()["price_usd_per_mwh"]
    st.caption(
        f"**Price:** a flat projection of {p['base']:.2f} USD/MWh (Georgia Power avoided energy cost "
        f"for 2026, solar ground-mounted fixed tilt; band {p['low']:.2f} ... {p['high']:.2f}) applied to "
        "2020-2023. Realised wholesale prices in 2021-2022 were higher, and this is the **utility's "
        "avoided cost, not this plant's PPA price** (unknown).")


def synthetic_banner():
    """Shown on every result that depends on injected faults."""
    st.warning("**SYNTHETIC faults.** The string faults in this section were injected into the real "
               "AC series by `src/pv_loss_auditor/synthetic_faults.py` (seed 9069); they are not events of the plant. "
               "Faults are abrupt, constant and on the AC side.", icon="⚠️")


def design_loss_caveat():
    st.caption("**Design losses (6.34 %)** are an assumption (LID, mismatch, wiring, connections, "
               "nameplate): there is no PVsyst report for this plant. At 4 % / 8 % the running "
               "shortfall changes from 19.9 to 24.2 / 16.8 GWh; downtime stays within 0.7 GWh.")


def style(fig, title=None, height=380):
    """Common Plotly look: thin recessive grid, ink-coloured text, legend on top."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK), x=0) if title else None,
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(color=INK_SECONDARY, size=12), margin=dict(l=10, r=10, t=90 if title else 40, b=10),
        # traceorder "normal": the legend reads in the order of the stack (bottom to top)
        legend=dict(orientation="h", y=1.14, x=0, font=dict(color=INK), traceorder="normal"),
        hoverlabel=dict(bgcolor="white", font_color=INK))
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID)
    return fig
