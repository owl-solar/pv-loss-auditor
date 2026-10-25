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


# Which presentation group each waterfall step belongs to, and a reader-friendly name.
STEP_CAUSE = {step: CAUSE_NAMES[cause] for cause, steps in CAUSES.items() for step in steps}
STEP_LABELS = {
    "temperature": "Temperature", "dc_losses": "Design DC losses (assumed)",
    "inverter": "Inverter efficiency", "clipping": "Clipping",
    "downtime": "Downtime", "unknown_ac": "Unknown AC (no channel value)",
    "channel_credit": "Channel credit (not a loss)", "running_shortfall": "Running shortfall",
    "reconciliation": "Meter reconciliation", "after_inverters": "After inverters (transformer, cables)",
}


def waterfall_bars(steps, values):
    """Geometry of a horizontal waterfall: where every bar starts and how long it is.

    steps: frame from waterfall_steps (label, kind); values: the same rows in the unit the reader
    chose (any unit, same order). The first row is the nominal total, the last the metered total.
    A loss moves the running level down, a negative loss (gain: cold weather, channel credit) up.
    Returns label, cause ("Total" for the two ends), value, start, end, base (= min) and length (= |end - start|); the level
    after the last step must equal the metered total, else ValueError.
    """
    values = pd.Series(list(values), index=steps.index, dtype=float)
    rows, level = [], 0.0
    for label, kind, value in zip(steps["label"], steps["kind"], values):
        if kind == "total":
            start, end = 0.0, value
            level = value if label == "nominal" else level
        else:
            start, end = level, level - value
            level = end
        rows.append({"label": label, "cause": STEP_CAUSE.get(label, "Total"), "value": value, "start": start,
                     "end": end, "base": min(start, end), "length": abs(end - start)})
    bars = pd.DataFrame(rows)
    if abs(level - bars["value"].iloc[-1]) > 1e-6 * max(1.0, abs(bars["value"].iloc[0])):
        raise ValueError("waterfall bars do not end at the metered total")
    return bars


def monthly_cause_share(monthly):
    """Loss causes per month as % of that month's nominal energy (the groups of losses.CAUSES)."""
    c = by_cause(monthly).rename(columns=CAUSE_NAMES)
    return c.div(monthly["nominal_kwh"], axis=0) * 100


def inverter_number(column):
    """'p_ac_inv17_kw' -> '17'."""
    return column.split("inv")[1][:2]


def failure_groups(pairs, down_days, min_jaccard=0.95):
    """Inverters that fail together: connected groups of pairs with Jaccard >= min_jaccard.

    pairs: exported `down_pairs` (a, b, jaccard); down_days: exported frame (dates x inverters,
    1.0 = down, 0.0 = up, NaN = unknown). One row per group with its members ("11+12+17+18"),
    the weakest Jaccard among its pairs and the days on which ALL members were down.
    The cause (shared transformer skid, feeder) is not in the data: a hypothesis only.
    """
    strong = pairs[pairs["jaccard"] >= min_jaccard]
    parent = {}

    def find(i):
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(strong["a"], strong["b"]):
        parent[find(a)] = find(b)
    members = {}
    for inverter in parent:
        members.setdefault(find(inverter), []).append(inverter)
    rows = []
    for group in members.values():
        group = sorted(group)
        inside = strong[strong["a"].isin(group) & strong["b"].isin(group)]
        rows.append({"group": "+".join(inverter_number(c) for c in group), "inverters": len(group),
                     "min_jaccard": inside["jaccard"].min(),
                     "days_all_down": int((down_days[group] == 1).all(axis=1).sum())})
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def meter_verdict(ratio_median, agree=0.97, disagree=0.95):
    """What the plant meter says about an outage episode, from the median of the daily
    sum(inverters) / (meter x 1.022): >= agree the stops are real (the meter sees them too),
    < disagree part of the zeros are lost data channels, in between mixed. Thresholds are mine:
    0.97 is 1 - channel_tolerance of the config, 0.95 leaves a gap so that borderline episodes
    are not forced into one of the two readings."""
    if ratio_median >= agree:
        return "stops confirmed by the meter"
    if ratio_median < disagree:
        return "partly lost data channels"
    return "mixed"


def rolling_median(series, window=61, min_periods=20):
    """Centred rolling median on a regular daily grid (NaN days stay in, so `window` rows = days)."""
    return series.asfreq("D").rolling(window, center=True, min_periods=min_periods).median()

