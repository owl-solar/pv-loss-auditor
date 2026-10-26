"""Capture the key numbers of the pipeline as a golden reference (JSON).

Two runs, two sections of tests/fixtures/golden_values.json:
- "fixture": 14 real days (tests/fixtures/9069_clean/), model + KPIs + waterfall + down-days.
  Fast, needs no external data: this is what CI checks.
- "full": the whole analysis period 2020-05 ... 2023-11 (needs data/processed/9069 and the cached
  Open-Meteo files), adds degradation, soiling and the anomaly detectors on SYNTHETIC faults.

It calls the same table builders as the dashboard export, so the golden numbers are the numbers
the project reports. Run from the project root:

    python -m scripts.capture_golden --run fixture
    python -m scripts.capture_golden --run full
    python -m scripts.capture_golden --run full --out some_other.json    # compare, do not overwrite
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pv_loss_auditor import export_dashboard as ex
from pv_loss_auditor import prices
from pv_loss_auditor.config import load_config
from pv_loss_auditor.model import cell_temperature, load_inputs, weighted_gamma

GOLDEN = Path("tests/fixtures/golden_values.json")
FIXTURE_DIR = "tests/fixtures/9069_clean"


def dump(df, max_rows=None):
    """Small table -> {"index": [...], "columns": {name: [values]}}; non-numeric kept as text."""
    df = df if max_rows is None else df.head(max_rows)
    cols = {}
    for c in df.columns:
        col = df[c]
        if pd.api.types.is_bool_dtype(col):
            cols[str(c)] = [bool(v) for v in col]
        elif pd.api.types.is_numeric_dtype(col):
            cols[str(c)] = [None if pd.isna(v) else float(v) for v in col]
        else:
            cols[str(c)] = [None if pd.isna(v) else str(v) for v in col]
    return {"index": [str(i) for i in df.index], "columns": cols}


def pipeline_section(config):
    """Model, KPIs, waterfall, down-days: everything that needs only the clean dataset."""
    inputs, ac = load_inputs(config)
    t_cell = cell_temperature(inputs, config)
    gamma = weighted_gamma(config)
    out = {"gamma_per_c": float(gamma)}

    wf = ex.waterfall_tables(inputs, ac, t_cell, config, gamma)
    monthly = wf["waterfall_monthly"]
    monthly.index = monthly.index.astype(str)
    yearly = monthly.groupby(monthly.index.str[:4]).sum()
    out["waterfall_yearly_kwh"] = dump(yearly)
    out["waterfall_total_kwh"] = {k: float(v) for k, v in monthly.sum().items()}
    out["waterfall_max_closure_kwh"] = float(monthly["closure_kwh"].abs().max())
    out["dc_loss_sensitivity"] = dump(wf["dc_loss_sensitivity"])

    kpis, t_ref = ex.kpi_tables(inputs, ac, t_cell, config, gamma)
    out["t_ref_c"] = float(t_ref)
    out["kpi_yearly"] = dump(kpis["kpi_yearly"])
    out["kpi_monthly"] = dump(kpis["kpi_monthly"])

    down = ex.outage_tables(inputs, ac, config)["down_days"]
    # NaN = day not known for that inverter; only True counts as a down-day.
    is_down = down == 1
    out["down_inverter_days"] = int(is_down.to_numpy().sum())
    out["unknown_inverter_days"] = int(down.isna().to_numpy().sum())
    out["down_days_per_inverter"] = {c: int(v) for c, v in is_down.sum().items()}

    # Value of the audit losses (downtime net of channel loss + running shortfall) at the base price.
    price = prices.price_usd_per_mwh(config)[0] if Path(config["price"]["file"]).exists() else None
    if price is not None:
        lost_kwh = monthly["downtime"].sum() + monthly["channel_credit"].sum() + monthly["running_shortfall"].sum()
        out["price_usd_per_mwh"] = float(price)
        out["audit_loss_kwh"] = float(lost_kwh)
        out["audit_loss_usd"] = float(lost_kwh * price / 1000)
    return out, (inputs, ac, t_cell)


def full_extra(config, parts):
    """Degradation, soiling, anomalies: need years of data and the Open-Meteo cache."""
    inputs, ac, t_cell = parts
    out = {}
    deg = ex.degradation_tables(inputs, ac, t_cell, config)
    daily = deg["daily_series"]
    out["daily_series_summary"] = {c: {"n": int(daily[c].notna().sum()), "sum": float(daily[c].sum()),
                                       "mean": float(daily[c].mean())}
                                   for c in daily.columns if pd.api.types.is_numeric_dtype(daily[c])}
    for name in ("degradation_yoy", "degradation_filters", "soiling", "external_poa_stats",
                 "pr_by_poa_source", "aoi_by_b0", "shortfall_split"):
        out[name] = dump(deg[name])

    an = ex.anomaly_tables(inputs, ac, t_cell, config)
    for name in ("anomaly_summary", "anomaly_real_events", "anomaly_excluded_flag_share", "anomaly_noise",
                 "anomaly_by_cell"):
        out[name] = dump(an[name])
    ev = an["anomaly_events"]
    out["anomaly_events"] = {"n": len(ev), "by_set": {k: int(v) for k, v in ev["set"].value_counts().items()},
                             "start_sum_days": int((ev["start"] - ev["start"].min()).dt.days.sum())}
    out["outage_episodes"] = dump(ex.outage_tables(inputs, ac, config)["outage_episodes"])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", choices=["fixture", "full"], required=True)
    parser.add_argument("--config", default="configs/system_9069.yaml")
    parser.add_argument("--out", default=str(GOLDEN))
    args = parser.parse_args()

    config = load_config(args.config)
    if args.run == "fixture":
        config["clean"]["out_dir"] = FIXTURE_DIR
    section, parts = pipeline_section(config)
    if args.run == "full":
        section.update(full_extra(config, parts))

    out = Path(args.out)
    golden = json.loads(out.read_text(encoding="utf8")) if out.exists() else {}
    golden[args.run] = section
    out.write_text(json.dumps(golden, indent=1, sort_keys=True), encoding="utf8")
    print(f"{args.run}: written to {out} ({out.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
