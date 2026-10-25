"""Aggregations behind the dashboard (app/): monthly waterfall tables -> years, causes, USD.

No Streamlit here, so every number the app shows can be tested. Input is the exported table
`waterfall_monthly` (kWh per local month, see src/export_dashboard.py).
"""

import pandas as pd

from src.losses import CAUSES, STEPS, by_cause

# Short names of the presentation groups in src/losses.CAUSES (same order).
CAUSE_NAMES = dict(zip(CAUSES, ["Design", "Downtime", "Running shortfall", "After inverters"]))
# Causes the audit points at: what an operator can act on. "Design" is the plant as built.
AUDIT_CAUSES = ["Downtime", "Running shortfall"]


def with_year(monthly):
    """Add the calendar year (from the 'YYYY-MM' index) and keep months as they are."""
    out = monthly.copy()
    out["year"] = out.index.str[:4]
    return out


def yearly(monthly):
    """Sum of the monthly waterfall per calendar year (kWh). 2020 starts in May, 2023 ends in
    November: the years are NOT full years, compare them as shares of nominal energy."""
    return with_year(monthly).groupby("year").sum()


def causes_table(monthly, per="year"):
    """Loss causes in kWh with the short names. per = "year" (rows = years) or "total" (one row).

    Also carries nominal_kwh, expected_kwh and metered_kwh for shares and checks.
    """
    base = yearly(monthly) if per == "year" else monthly.sum().to_frame("total").T
    causes = by_cause(base).rename(columns=CAUSE_NAMES)
    return causes.join(base[["nominal_kwh", "expected_kwh", "metered_kwh"]])


def to_usd(kwh, usd_per_mwh):
    """kWh -> USD at a flat price (USD/MWh)."""
    return kwh * usd_per_mwh / 1000


def audit_total(monthly, usd_per_mwh):
    """Downtime + running shortfall over the whole period: GWh and USD at the given price."""
    total = causes_table(monthly, per="total").iloc[0]
    kwh = float(total[AUDIT_CAUSES].sum())
    return {"gwh": kwh / 1e6, "usd": to_usd(kwh, usd_per_mwh)}


def price_band(prices):
    """(low, base, high) USD/MWh from the exported meta dict."""
    return prices["low"], prices["base"], prices["high"]


def waterfall_steps(monthly, year=None):
    """Steps for the waterfall chart: nominal, each loss step, metered (kWh), with a check.

    year = None gives the whole period. The returned frame has columns label, kwh, kind
    ("total" for the two ends, "loss" for steps; a negative loss is a gain, e.g. the channel
    credit) and is ordered as in src/losses.STEPS.
    """
    m = monthly if year is None else monthly[monthly.index.str[:4] == str(year)]
    s = m.sum()
    rows = [{"label": "nominal", "kwh": s["nominal_kwh"], "kind": "total"}]
    rows += [{"label": step, "kwh": s[step], "kind": "loss"} for step in STEPS]
    rows += [{"label": "metered", "kwh": s["metered_kwh"], "kind": "total"}]
    frame = pd.DataFrame(rows)
    gap = frame.loc[0, "kwh"] - frame.loc[frame.kind == "loss", "kwh"].sum() - frame.loc[len(frame) - 1, "kwh"]
    if abs(gap) > 1.0:
        raise ValueError(f"waterfall steps do not add up: off by {gap:.1f} kWh")
    return frame


UNITS = ["GWh", "USD thousand", "% of nominal energy"]


def scale(kwh_table, nominal_kwh, unit, usd_per_mwh):
    """Convert a table in kWh to the unit chosen by the reader.

    Returns (table, axis title, number format). "% of nominal energy" divides each row by its own
    nominal energy: the right unit to compare 2020 (May-Dec) with 2023 (Jan-Nov).
    """
    if unit == "GWh":
        return kwh_table / 1e6, "GWh", ".1f"
    if unit == "USD thousand":
        return to_usd(kwh_table, usd_per_mwh) / 1000, "USD thousand", ",.0f"
    if unit == "% of nominal energy":
        return kwh_table.div(nominal_kwh, axis=0) * 100, "% of nominal energy", ".1f"
    raise ValueError(f"unknown unit {unit!r}")
