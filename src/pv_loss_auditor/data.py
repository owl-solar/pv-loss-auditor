"""Clean 9069 dataset: all files on one UTC grid, QA flags, one wide parquet file per month.

Principle: flag, do not delete. Measured values stay as they are; each problem gets a boolean
column `flag_<test>__<channel>` (True = suspicious). Later phases pick their own filters, e.g.
PR needs clean irradiance, while downtime analysis must keep the zeros of a stopped inverter.
Only two values are changed, both recorded:
- logger placeholders (<= -250, e.g. -297) -> NaN (not measurements; counted in the report);
- small negative irradiance at night (zero offset of the sensor) -> 0, flagged.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pvanalytics.quality.gaps import stale_values_diff
from pvanalytics.quality.irradiance import check_irradiance_limits_qcrad

from pv_loss_auditor import timealign as ta
from pv_loss_auditor.coverage import SENTINEL_MAX
from pv_loss_auditor.geometry import Site

FILES = ["irradiance", "environment", "electrical_ac", "meter"]


def _select(columns, patterns):
    return [c for c in columns if any(re.search(p, c) for p in patterns)]


def load_utc(config):
    """Selected channels of all files -> one DataFrame on a regular 5-min UTC grid.

    Returns (data, sentinel_counts). Each file is converted with its own clock rule
    (config clock.rules); steps missing in a file become NaN.
    """
    cl = config["clean"]
    parts, sentinels = [], {}
    for file in FILES:
        folder = Path(config["period"]["out_dir"]) / file
        paths = sorted(folder.glob(f"{file}_*.parquet"))
        cols = _select(pd.read_parquet(paths[0]).columns, cl["keep"][file])
        raw = pd.concat([pd.read_parquet(p, columns=cols) for p in paths])
        raw = raw[~raw.index.duplicated()]
        sentinels.update((raw <= SENTINEL_MAX).sum().to_dict())
        df = raw.mask(raw <= SENTINEL_MAX)
        df.index = ta.to_utc(df.index, config["clock"]["rules"][file])
        parts.append(df[df.index.notna()])
    start = min(p.index.min() for p in parts).floor("D")
    end = max(p.index.max() for p in parts)
    grid = pd.date_range(start, end, freq="5min", name="time_utc")
    data = pd.concat([p.reindex(grid) for p in parts], axis=1)
    return data, pd.Series(sentinels, name="sentinel_values")


def add_sun(data, config):
    """Sun elevation, clear-sky GHI and clear-sky POA of the fixed plane; returns the Site."""
    s = config["system"]
    site = Site(s["latitude"], s["longitude"], s["elevation_m"], data.index)
    m = config["mount"]
    data["sun_elevation_deg"] = site.solpos["apparent_elevation"]
    data["ghi_clearsky_wm2"] = site.clearsky["ghi"]
    data["poa_clearsky_wm2"] = site.poa(m["tilt_deg"], m["azimuth_deg"], ghi=None)
    data["flag_daylight"] = data["sun_elevation_deg"] > 0
    return site


def flag_stale(data, columns, cl, ac_rating_kw=None):
    """flag_stale__<col>: >= window identical values in daylight (pvanalytics stale_values_diff).

    For AC power, exact 0 (inverter off = downtime, a real loss) and the clipping plateau
    (>= ac_clip_fraction x rating) are not 'stuck'; they are handled in Phase 3.
    """
    day = data["sun_elevation_deg"] > cl["stale_min_elevation_deg"]
    flags = {}
    for col in columns:
        x = data[col]
        stale = stale_values_diff(x, window=cl["stale_window_steps"]) & day & x.notna()
        if ac_rating_kw is not None:
            stale &= (x != 0) & (x < cl["ac_clip_fraction"] * ac_rating_kw)
        flags[f"flag_stale__{col}"] = stale
    return pd.DataFrame(flags)


def flag_irradiance(data, site, ghi_cols, poa_cols, cl):
    """Out-of-range irradiance; zero offset at night set to 0 (flag_night_zeroed__<col>).

    GHI: QCRAD physical limits (upper bound from extraterrestrial irradiance and sun zenith).
    POA: above poa_cs_factor x clear-sky POA + offset, or below irradiance_min_day_wm2 by day.
    """
    day = data["flag_daylight"]
    flags = {}
    for col in ghi_cols + poa_cols:
        x = data[col]
        night_offset = ~day & (x < 0) & (x >= cl["night_offset_floor_wm2"])
        data.loc[night_offset, col] = 0.0
        flags[f"flag_night_zeroed__{col}"] = night_offset
        if col in ghi_cols:
            ok = check_irradiance_limits_qcrad(site.solpos["zenith"], site.dni_extra, ghi=x)[0]
            out = ~ok
        else:
            upper = cl["poa_cs_factor"] * data["poa_clearsky_wm2"] + cl["poa_cs_offset_wm2"]
            out = (x > upper) | (day & (x < cl["irradiance_min_day_wm2"]))
        # At night only values below the zero-offset band are implausible (e.g. -181 W/m2).
        out |= ~day & (x < cl["night_offset_floor_wm2"])
        flags[f"flag_range__{col}"] = out & x.notna()
    return pd.DataFrame(flags)


def flag_sensor_limits(data, limits):
    """Temperature and wind outside their physical range: {regex: [min, max]} from the config."""
    flags = {}
    for pattern, (lo, hi) in limits.items():
        for col in [c for c in data.columns if re.search(pattern, c)]:
            x = data[col]
            flags[f"flag_range__{col}"] = ((x < lo) | (x > hi)) & x.notna()
    return pd.DataFrame(flags)


def flag_ac(data, columns, cl, ac_rating_kw):
    """AC power above ac_max_fraction x rating or below ac_min_kw."""
    return pd.DataFrame({
        f"flag_range__{c}": (data[c] > cl["ac_max_fraction"] * ac_rating_kw) | (data[c] < cl["ac_min_kw"])
        for c in columns
    })


def build_clean(config):
    """Full cleaning run. Returns (data with flag columns, sentinel counts)."""
    cl = config["clean"]
    data, sentinels = load_utc(config)
    site = add_sun(data, config)
    rating = config["inverters"]["ac_kw"][0]["ac_kw"]  # 825 kW, verified for all 40 in QA
    ghi_cols = [c for c in data.columns if c.startswith("ghi_") and "clearsky" not in c]
    poa_cols = [c for c in data.columns if c.startswith("poa_") and "clearsky" not in c]
    inv_cols = [c for c in data.columns if re.match(r"p_ac_inv\d{2}_kw$", c)]
    sensor_cols = ghi_cols + poa_cols + [c for c in data.columns if c.startswith(("t_", "wind_"))]
    flags = [
        flag_irradiance(data, site, ghi_cols, poa_cols, cl),   # first: it zeroes night offsets
        flag_stale(data, sensor_cols, cl),
        flag_sensor_limits(data, cl["sensor_limits"]),
        flag_stale(data, inv_cols, cl, ac_rating_kw=rating),
        flag_ac(data, inv_cols, cl, rating),
        # The plant meter is checked against plant AC (40 x rating), with the same rules.
        flag_stale(data, ["p_ac_meter2_kw"], cl, ac_rating_kw=40 * rating),
        flag_ac(data, ["p_ac_meter2_kw"], {**cl, "ac_min_kw": 40 * cl["ac_min_kw"]}, 40 * rating),
    ]
    return pd.concat([data, *flags], axis=1), sentinels


def save_monthly(data, out_dir):
    """One parquet file per UTC month: <out_dir>/9069_clean_<YYYY-MM>.parquet."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for month, part in data.groupby(data.index.to_period("M")):
        path = out / f"9069_clean_{month}.parquet"
        part.to_parquet(path)
        paths.append(path)
    return paths


def load_clean(config, columns=None):
    """Read the clean dataset back (all months)."""
    paths = sorted(Path(config["clean"]["out_dir"]).glob("9069_clean_*.parquet"))
    return pd.concat([pd.read_parquet(p, columns=columns) for p in paths])


def flag_summary(data):
    """Per channel: % valid in daylight and % of daylight steps carrying each flag."""
    day = data["flag_daylight"]
    channels = [c for c in data.columns if not c.startswith("flag_")
                and c not in ("sun_elevation_deg", "ghi_clearsky_wm2", "poa_clearsky_wm2")]
    rows = []
    for ch in channels:
        row = {"channel": ch, "valid_daylight_pct": 100 * data.loc[day, ch].notna().mean()}
        for kind in ("stale", "range", "night_zeroed"):
            col = f"flag_{kind}__{ch}"
            if col in data:
                base = ~day if kind == "night_zeroed" else day
                row[f"{kind}_pct"] = 100 * data.loc[base, col].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("channel")


def compact_summary(summary):
    """flag_summary with the 40 inverters folded into one row (median and worst inverter)."""
    inv = summary[summary.index.str.match(r"p_ac_inv\d{2}_kw$")]
    rest = summary.drop(inv.index)
    folded = pd.DataFrame({
        "valid_daylight_pct": [inv["valid_daylight_pct"].median(), inv["valid_daylight_pct"].min()],
        "stale_pct": [inv["stale_pct"].median(), inv["stale_pct"].max()],
        "range_pct": [inv["range_pct"].median(), inv["range_pct"].max()],
    }, index=["p_ac_inv01..40_kw (median)", "p_ac_inv01..40_kw (worst)"])
    return pd.concat([rest, folded]).rename_axis("channel")


def monthly_valid(data, channels):
    """% of daylight steps with a value and no stale/range flag, per UTC month and channel."""
    day = data["flag_daylight"]
    out = {}
    for ch in channels:
        bad = data[ch].isna()
        for kind in ("stale", "range"):
            if f"flag_{kind}__{ch}" in data:
                bad |= data[f"flag_{kind}__{ch}"]
        good = (~bad)[day]
        out[ch] = 100 * good.groupby(good.index.to_period("M")).mean()
    return pd.DataFrame(out).rename_axis("month")


def write_report(data, sentinels, config, path, findings=""):
    """Markdown QA report: settings, flag shares per channel, monthly usable share, findings."""
    cl = config["clean"]
    summary = compact_summary(flag_summary(data))
    key = ["ghi_pyr12_wm2", "poa_ref01_wm2", "poa_ref04_wm2", "poa_ref10_wm2", "poa_pyr02b_wm2",
           "t_mod_tc01s1_c", "t_mod_tc03s1_c", "t_mod_tc06s1_c", "t_mod_tc10s1_c",
           "t_amb_ws01s1_c", "p_ac_meter2_kw"]
    monthly = monthly_valid(data, [k for k in key if k in data])
    monthly.columns = [c.replace("_wm2", "").replace("_kw", "").replace("_c", "") for c in monthly.columns]
    text = f"""# QA report — PVDAQ 9069 (Simon Solar Farm), {data.index.min():%Y-%m-%d} … {data.index.max():%Y-%m-%d} UTC

Generated by `src/pv_loss_auditor/data.py` (notebook `01_data_qa.ipynb`, section 4). Clean data:
`{cl['out_dir']}/9069_clean_<YYYY-MM>.parquet`, one file per UTC month, 5-min steps,
{data.shape[1]} columns ({sum(c.startswith('flag_') for c in data.columns)} of them flags).

## Rules
- All files converted to UTC with their own clock rule (irradiance, environment: EU DST dates on
  UTC−5; electrical_ac, meter: America/New_York). The skipped / repeated hour at a switch is dropped.
- Values are kept as measured. Only changes: logger placeholders (≤ −250) → NaN
  ({int(sentinels.sum())} values, all in {', '.join(sentinels[sentinels > 0].index) or '–'});
  night irradiance in [{cl['night_offset_floor_wm2']}, 0) W/m² → 0 (`flag_night_zeroed__*`).
- `flag_stale__*`: ≥ {cl['stale_window_steps']} identical consecutive values (pvanalytics
  `stale_values_diff`) with sun > {cl['stale_min_elevation_deg']}°. For AC power, exact 0 (inverter
  off) and the clipping plateau (≥ {cl['ac_clip_fraction']:.0%} of rating) are not flagged.
- `flag_range__*`: GHI outside QCRAD physical limits; POA > {cl['poa_cs_factor']} × clear-sky POA
  + {cl['poa_cs_offset_wm2']} W/m² or < {cl['irradiance_min_day_wm2']} W/m² by day; AC > {cl['ac_max_fraction']:.0%}
  of rating (825 kW per inverter, 33 000 kW for the meter) or < {cl['ac_min_kw']} kW per inverter;
  irradiance at night below {cl['night_offset_floor_wm2']} W/m²; temperature and wind outside
  {', '.join(f'{p.strip("^_")} {lo}…{hi}' for p, (lo, hi) in cl['sensor_limits'].items())}.
- Not included: inverter heatsink temperatures (clock not established) and thermocouple 03 s1
  (implausible for the whole period, see findings).

## Flag shares per channel (% of daylight steps; night_zeroed: % of night steps)
{to_markdown(summary.round(2))}

## Usable share per month (% of daylight steps with a value and no stale / range flag)
{to_markdown(monthly.round(0), floatfmt="{:.0f}")}

## Findings
{findings}
"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return path


def main():
    import argparse

    from pv_loss_auditor.config import load_config

    parser = argparse.ArgumentParser(description="Build the clean 9069 dataset (monthly parquet).")
    parser.add_argument("--config", default="configs/system_9069.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data, sentinels = build_clean(config)
    paths = save_monthly(data, config["clean"]["out_dir"])
    print(f"{len(paths)} monthly files written to {config['clean']['out_dir']}, "
          f"{data.shape[0]} steps x {data.shape[1]} columns")


def to_markdown(df, floatfmt="{:.2f}"):
    """Minimal markdown table (avoids an extra dependency); NaN -> '–'."""
    head = "| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    lines = [head, sep]
    for idx, row in df.iterrows():
        cells = ["–" if pd.isna(v) else (floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
                 for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
