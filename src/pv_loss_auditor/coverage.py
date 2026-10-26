"""Full-period data coverage map of wide PVDAQ CSV files, using point reads via byte ranges.

Downloading 9069_electrical_ac.csv (1.8 GB) just to see which months and sensors have data
is too expensive on mobile internet. Instead we read small chunks at evenly spaced byte
positions. Each chunk contains at least one complete row = a snapshot of ALL channels at one
timestamp. With ~20 snapshots per month we get:
1. which periods have rows at all (large time jumps between neighbouring snapshots = gaps);
2. which channels report valid values in daylight (share of non-empty values per month).
Precision of (2) is coarse: ~10 daylight snapshots per month -> about +-15 percentage points.
Good enough to choose an analysis period, not to compute availability KPIs.
"""

import logging
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from pv_loss_auditor import pvdaq
from pv_loss_auditor.config import load_config
from pv_loss_auditor.logs import setup_logging

log = logging.getLogger(__name__)

TS_LEN = pvdaq.TS_LEN
SENTINEL_MAX = -250  # values like -297 / -299 / -300 are logger placeholders, not measurements
REQUEST_OVERHEAD_BYTES = 700  # rough size of HTTP request + response headers, estimate only


def _read_row(s3, key, offset, read_bytes, size):
    """Return the first complete CSV row that starts after `offset` (as str), or None at EOF.

    A random offset usually lands inside a row, so we skip to the next newline and take the
    row after it. If the chunk is too short to contain a complete row, double it and retry.
    """
    n = read_bytes
    while True:
        chunk = pvdaq._get_range(s3, key, offset, min(offset + n, size))
        nl = chunk.find(b"\n")
        end = chunk.find(b"\n", nl + 1) if nl != -1 else -1
        if end != -1:
            return chunk[nl + 1:end].rstrip(b"\r").decode()
        if offset + n >= size:
            return None
        n *= 2


def _file_span(s3, key, header, read_bytes):
    """First and last timestamp of a file (two tiny reads)."""
    # data_start - 1 is the newline that ends the header, so the next row is the first data row.
    first = _read_row(s3, key, header["data_start"] - 1, read_bytes, header["size"])
    tail = pvdaq._get_range(s3, key, header["size"] - 2 * read_bytes, header["size"])
    last = tail.rstrip().split(b"\n")[-1].decode()
    return first[:TS_LEN], last[:TS_LEN]


def plan_points(s3, config):
    """For each file: byte offsets of the point reads (points_per_month x months in the file)."""
    cov = config["coverage"]
    plan = []
    for name, spec in config["data"]["files"].items():
        header = pvdaq._cached_header(s3, spec["key"], config["data"]["header_cache"])
        read_bytes = cov["read_bytes"][name]
        first_ts, last_ts = _file_span(s3, spec["key"], header, read_bytes)
        n_months = len(pd.period_range(first_ts[:7], last_ts[:7], freq="M"))
        n_points = n_months * cov["points_per_month"]
        span = header["size"] - header["data_start"]
        # Stratified random sampling in BYTES (we cannot jump to a timestamp without searching):
        # split the file into n_points equal strata and read at a random position in each.
        # A fixed step would alias with the daily cycle: in the meter file one step was ~25 h,
        # so consecutive points fell at almost the same time of day (mostly night for weeks).
        rng = np.random.default_rng(cov["random_seed"])
        offsets = [header["data_start"] + int((i + u) * span / n_points)
                   for i, u in enumerate(rng.random(n_points))]
        plan.append({"name": name, "key": spec["key"], "header": header, "read_bytes": read_bytes,
                     "first_ts": first_ts, "last_ts": last_ts, "n_months": n_months,
                     "offsets": offsets})
    return plan


def estimate(plan, gap_probes_reserve=200):
    """Expected traffic in MB per file: point reads + HTTP overhead + reserve for gap search."""
    rows = []
    for p in plan:
        n = len(p["offsets"])
        rows.append({"file": p["name"], "period": f"{p['first_ts'][:10]} .. {p['last_ts'][:10]}",
                     "months": p["n_months"], "points": n,
                     "data_mb": n * p["read_bytes"] / 1e6,
                     "overhead_mb": n * REQUEST_OVERHEAD_BYTES / 1e6,
                     "gap_search_mb": gap_probes_reserve * p["read_bytes"] / 1e6})
    table = pd.DataFrame(rows)
    table["total_mb"] = table[["data_mb", "overhead_mb", "gap_search_mb"]].sum(axis=1)
    return table


def _parse_rows(rows, offsets, columns, patterns):
    """Rows (CSV strings) -> DataFrame: timestamp, row length, byte offset, selected channels."""
    keep = [c for c in columns if any(re.search(pt, c) for pt in patterns)]
    split = pd.Series(rows).str.split(",")
    df = pd.DataFrame({"measured_on": pd.to_datetime(split.str[0], format="%Y-%m-%d %H:%M:%S"),
                       "offset": offsets, "row_bytes": pd.Series(rows).str.len() + 1})
    for c in keep:
        # Empty field -> NaN. errors="coerce" also turns any non-numeric text into NaN.
        df[c] = pd.to_numeric(split.str[columns.index(c)], errors="coerce")
    return df


def _refine_gap(s3, p, lo, ts_lo, hi, ts_hi):
    """Locate a time gap between two byte offsets: (last timestamp before, first after).

    Bisection: the gap lies in the half where time advances more per byte than in the other.
    Then scan the remaining small window row by row and take the largest jump.
    """
    size, rb = p["header"]["size"], p["read_bytes"]
    while hi - lo > 4 * rb:
        mid = (lo + hi) // 2
        ts_mid = pd.Timestamp(_read_row(s3, p["key"], mid, rb, size)[:TS_LEN])
        if (ts_mid - ts_lo) / (mid - lo) > (ts_hi - ts_mid) / (hi - mid):
            hi, ts_hi = mid, ts_mid
        else:
            lo, ts_lo = mid, ts_mid
    chunk = pvdaq._get_range(s3, p["key"], lo, min(hi + 2 * rb, size))
    lines = chunk.split(b"\n")[1:-1]  # drop the partial first and last line
    ts = pd.to_datetime([x[:TS_LEN].decode() for x in lines], format="%Y-%m-%d %H:%M:%S")
    j = int(np.argmax(np.diff(ts.values)))
    return ts[j], ts[j + 1]


def find_gaps(s3, p, samples, min_gap_days):
    """Time gaps longer than min_gap_days.

    Neighbouring snapshots are a fixed number of bytes apart; dividing by the row length gives
    the expected number of 5-min rows between them. A time jump 4x larger than expected marks
    a candidate, which is then located exactly with _refine_gap.
    """
    s = samples.sort_values("offset").reset_index(drop=True)
    gaps = []
    for i in range(len(s) - 1):
        a, b = s.iloc[i], s.iloc[i + 1]
        dt = b["measured_on"] - a["measured_on"]
        expected_rows = (b["offset"] - a["offset"]) / ((a["row_bytes"] + b["row_bytes"]) / 2)
        expected_dt = pd.Timedelta(minutes=5) * expected_rows
        if dt > pd.Timedelta(days=min_gap_days) and dt > 4 * expected_dt:
            start, end = _refine_gap(s3, p, a["offset"], a["measured_on"],
                                     b["offset"], b["measured_on"])
            if end - start > pd.Timedelta(days=min_gap_days):
                gaps.append({"file": p["name"], "last_before": start, "first_after": end,
                             "days": round((end - start) / pd.Timedelta(days=1), 1)})
    return gaps


def build_coverage(config, dry_run=False, assume_yes=False):
    """Sample all files, find gaps, cache results in coverage.out_dir. Skips cached files."""
    cov = config["coverage"]
    out_dir = Path(cov["out_dir"])
    s3 = pvdaq.get_s3_client()
    todo = [p for p in plan_points(s3, config)
            if not (out_dir / f"{p['name']}_samples.parquet").exists()]
    table = estimate(todo)
    log.info(table.round(2).to_string(index=False))
    total_mb = table["total_mb"].sum() if len(table) else 0.0
    log.info(f"Estimated total: {total_mb:.1f} MB "
          f"(spent on headers/first-last rows so far: {pvdaq.bytes_transferred / 1e6:.2f} MB)")
    if dry_run or not todo:
        return
    if total_mb > config["data"]["max_download_mb"]:
        raise SystemExit("Estimate exceeds max_download_mb; needs explicit approval.")
    if not assume_yes and input("Proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("Cancelled.")

    out_dir.mkdir(parents=True, exist_ok=True)
    for p in todo:
        size, rb = p["header"]["size"], p["read_bytes"]
        with ThreadPoolExecutor(max_workers=cov["workers"]) as ex:
            rows = list(ex.map(lambda off: _read_row(s3, p["key"], off, rb, size), p["offsets"]))
        ok = [(r, o) for r, o in zip(rows, p["offsets"]) if r]
        samples = _parse_rows([r for r, _ in ok], [o for _, o in ok], p["header"]["columns"],
                              cov["channel_patterns"][p["name"]])
        gaps = find_gaps(s3, p, samples, cov["min_gap_days"])
        samples.to_parquet(out_dir / f"{p['name']}_samples.parquet")
        (out_dir / f"{p['name']}_gaps.json").write_text(json.dumps(gaps, default=str, indent=1))
        log.info(f"{p['name']}: {len(samples)} snapshots, {len(gaps)} gaps > {cov['min_gap_days']} d, "
              f"traffic so far {pvdaq.bytes_transferred / 1e6:.1f} MB")


def load_coverage(config):
    """Read cached snapshots (all files) and gaps. No network access."""
    out_dir = Path(config["coverage"]["out_dir"])
    samples = {p.stem.replace("_samples", ""): pd.read_parquet(p)
               for p in sorted(out_dir.glob("*_samples.parquet"))}
    gaps = pd.DataFrame([g for p in sorted(out_dir.glob("*_gaps.json"))
                         for g in json.loads(p.read_text())])
    return samples, gaps


def availability_table(samples, config):
    """Channel x month table: % of daylight snapshots with a valid value (not empty, not sentinel).

    Daylight = sun elevation > min_sun_elevation_deg, with timestamps read as local time in
    local_tz_assumed. Months without daylight snapshots stay NaN.
    """
    cov, site = config["coverage"], config["system"]
    parts, counts = [], {}
    for name, df in samples.items():
        local = df["measured_on"].dt.tz_localize(cov["local_tz_assumed"],
                                                 ambiguous="NaT", nonexistent="NaT")
        df = df[local.notna()]
        local = local[local.notna()]
        elev = pvlib.solarposition.get_solarposition(pd.DatetimeIndex(local), site["latitude"],
                                                     site["longitude"])["apparent_elevation"]
        day = df[elev.to_numpy() > cov["min_sun_elevation_deg"]]
        month = day["measured_on"].dt.to_period("M")
        channels = [c for c in day.columns if c not in ("measured_on", "offset", "row_bytes")]
        valid = day[channels].notna() & (day[channels] > SENTINEL_MAX)
        parts.append(valid.groupby(month).mean().mul(100).T)
        counts[name] = month.value_counts().sort_index()
    table = pd.concat(parts)
    return table, pd.DataFrame(counts)


def short_label(col):
    """Readable channel label for plots, e.g. 'reference_cell_05_poa_irradiance_...' -> 'ref_05_poa'."""
    s = re.sub(r"_(o|inv|meter)_\d+$", "", col)
    for old, new in [("pyranometer_(class_a)_", "pyr_"), ("reference_cell_", "ref_"),
                     ("_poa_irradiance_(w/m2)", "_poa"), ("_ghi_irradiance_(w/m2)", "_ghi"),
                     ("thermocouple_", "tc_"), ("_back-of-module_temperature_(sensor_", "_s"),
                     (")_(c)", ""), ("inverter_", "inv_"), ("_ac_power_(kw)", "_p_ac")]:
        s = s.replace(old, new)
    return s


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Full-period coverage map via point reads.")
    parser.add_argument("--config", required=True, help="e.g. configs/system_9069.yaml")
    parser.add_argument("--dry-run", action="store_true", help="only print the traffic estimate")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()
    build_coverage(load_config(args.config), dry_run=args.dry_run, assume_yes=args.yes)


if __name__ == "__main__":
    main()
