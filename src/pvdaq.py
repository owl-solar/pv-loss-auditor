"""Access to NREL PVDAQ data in the public OEDI S3 bucket (no AWS credentials).

Bucket layout (s3://oedi-data-lake/pvdaq/), as found on inspection:
- csv/systems_YYYYMMDD.csv           one-row-per-system summary, several dated versions
- parquet/{site,system,mount,...}/   normalised metadata tables, one parquet file each
- parquet/metrics/                   one file per system: list of measured channels
- 2023-solar-data-prize/<id>_OEDI/   six detailed systems, each with metadata/*.json
"""

import argparse
import io
import json
import threading
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from src.config import load_config

BUCKET = "oedi-data-lake"
PREFIX = "pvdaq/"

# Latest dated version of the summary table at the time of writing.
SYSTEMS_CSV_KEY = "pvdaq/csv/systems_20250729.csv"
PARQUET_META_TABLES = [
    "site", "system", "mount", "inverters", "modules", "meters", "other-instruments",
]
DATA_PRIZE_PREFIX = "pvdaq/2023-solar-data-prize/"


def get_s3_client():
    """S3 client that sends unsigned requests, i.e. anonymous public read access."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def list_keys(s3, prefix):
    """Return (key, size_bytes) for every object under a prefix (handles pagination)."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            # Skip zero-byte "folder marker" objects (keys ending with '/').
            if not obj["Key"].endswith("/"):
                keys.append((obj["Key"], obj["Size"]))
    return keys


def _download(s3, key, out_root):
    """Download one object, mirroring its S3 path under out_root. Skips existing files."""
    local_path = Path(out_root) / key
    if not local_path.exists():
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(BUCKET, key, str(local_path))
    return local_path


def download_metadata(out_root="data/raw"):
    """Download all PVDAQ metadata files (no time series). Returns list of local paths."""
    s3 = get_s3_client()
    keys = [SYSTEMS_CSV_KEY]
    for table in PARQUET_META_TABLES:
        keys += [k for k, _ in list_keys(s3, f"{PREFIX}parquet/{table}/")]
    keys += [k for k, _ in list_keys(s3, f"{PREFIX}parquet/metrics/")]
    # Data Prize systems: only files inside the metadata/ subfolder.
    keys += [k for k, _ in list_keys(s3, DATA_PRIZE_PREFIX) if "/metadata/" in k]
    return [_download(s3, k, out_root) for k in keys]


# ---------------------------------------------------------------------------
# Time slices of large wide CSV files via HTTP byte ranges
# ---------------------------------------------------------------------------
# Data Prize CSVs (e.g. 9069_electrical_ac.csv, 1.8 GB) hold one row per timestamp, sorted by
# time, one column per channel. S3 can return any byte range of a file, so we binary-search
# the byte offset where a month starts and download only that slice, not the whole file.
# CSV is row-oriented: a slice still contains all columns; we keep only the needed ones.

PROBE_BYTES = 8192  # one binary-search read; longest 9069 row is ~1 KB, so it holds full rows
TS_LEN = len("YYYY-MM-DD HH:MM:SS")  # ISO timestamps compare correctly as plain strings

bytes_transferred = 0  # running total in this Python session, for traffic reporting
_traffic_lock = threading.Lock()  # src/coverage.py reads in parallel threads


def _get_range(s3, key, start, end):
    """Download bytes [start, end) of an S3 object and add them to the traffic counter."""
    global bytes_transferred
    if end <= start:
        return b""
    body = s3.get_object(Bucket=BUCKET, Key=key, Range=f"bytes={start}-{end - 1}")["Body"].read()
    with _traffic_lock:  # "+=" is not atomic across threads
        bytes_transferred += len(body)
    return body


def read_header(s3, key):
    """Return (column names, byte offset of the first data row, file size in bytes)."""
    size = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
    buf = b""
    # The header can be long (~18 KB for 401 columns), so read growing chunks until a newline.
    while b"\n" not in buf and len(buf) < size:
        buf += _get_range(s3, key, len(buf), min(2 * len(buf) + 4096, size))
    line = buf.split(b"\n", 1)[0]
    columns = line.decode("utf-8").rstrip("\r").split(",")
    return columns, len(line) + 1, size


def _cached_header(s3, key, cache_path):
    """read_header() with a local JSON cache, so headers are fetched only once."""
    cache_path = Path(cache_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if key not in cache:
        columns, data_start, size = read_header(s3, key)
        cache[key] = {"columns": columns, "data_start": data_start, "size": size}
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    return cache[key]


def find_offset(s3, key, data_start, size, ts):
    """Byte offset of the first row whose timestamp is >= ts ('YYYY-MM-DD HH:MM:SS').

    Binary search on byte positions. A position usually falls inside a row, so each probe
    skips to the next newline and reads the timestamp of the first complete row after it.
    Requires rows sorted by time; download_sample() checks this on the downloaded slice.
    """
    lo, hi = data_start, size
    while hi - lo > PROBE_BYTES:
        mid = (lo + hi) // 2
        chunk = _get_range(s3, key, mid, min(mid + PROBE_BYTES, size))
        nl = chunk.find(b"\n")
        row_ts = chunk[nl + 1:nl + 1 + TS_LEN].decode() if nl != -1 else ""
        if len(row_ts) < TS_LEN and mid + len(chunk) < size:
            raise RuntimeError(f"Row longer than PROBE_BYTES in {key}; increase PROBE_BYTES.")
        if len(row_ts) < TS_LEN or row_ts >= ts:
            hi = mid  # target row starts before the first complete row after mid (or EOF)
        else:
            lo = mid  # first complete row after mid is still earlier than ts

    # Linear scan of the small remaining window [lo, hi + PROBE_BYTES).
    chunk = _get_range(s3, key, lo, min(hi + PROBE_BYTES, size))
    # lo is a row start only if it is still data_start; otherwise skip the partial row at lo.
    pos = 0 if lo == data_start else chunk.find(b"\n") + 1
    while pos < len(chunk):
        if chunk[pos:pos + TS_LEN].decode() >= ts:
            return lo + pos
        nl = chunk.find(b"\n", pos)
        if nl == -1:
            break
        pos = nl + 1
    if lo + len(chunk) < size:
        raise RuntimeError(f"Row for {ts} not found in the final window of {key}.")
    return size  # no row >= ts: ts is after the end of the file


def _month_bounds(month):
    """'2019-06' -> ('2019-06-01 00:00:00', '2019-07-01 00:00:00')."""
    period = pd.Period(month, freq="M")
    fmt = "%Y-%m-%d %H:%M:%S"
    return period.start_time.strftime(fmt), (period + 1).start_time.strftime(fmt)


def plan_sample(s3, config):
    """Find the byte range of every (file, month) part that is not cached yet.

    Costs only headers and binary-search probes (a few hundred KB), no data rows.
    """
    data = config["data"]
    out_dir = Path(data["out_dir"])
    parts = []
    for name, spec in data["files"].items():
        header = _cached_header(s3, spec["key"], data["header_cache"])
        for month in data["sample_months"]:
            out_path = out_dir / f"{name}_{month}.parquet"
            if out_path.exists():
                print(f"cached, skip: {out_path}")
                continue
            ts_start, ts_end = _month_bounds(month)
            start = find_offset(s3, spec["key"], header["data_start"], header["size"], ts_start)
            end = find_offset(s3, spec["key"], header["data_start"], header["size"], ts_end)
            parts.append({"name": name, "month": month, "key": spec["key"], "start": start,
                          "end": end, "ts_start": ts_start, "ts_end": ts_end,
                          "columns": header["columns"], "channels": spec["channels"],
                          "out_path": out_path})
    return parts


def _parse_slice(raw, part, ts_col):
    """Parse a downloaded CSV slice (whole rows, no header); keep and rename needed columns."""
    df = pd.read_csv(io.BytesIO(raw), header=None, names=part["columns"],
                     usecols=[ts_col, *part["channels"]])
    df[ts_col] = pd.to_datetime(df[ts_col], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index(ts_col).rename(columns=part["channels"])
    return df[list(part["channels"].values())]


def download_sample(config, dry_run=False, assume_yes=False):
    """Download the configured months x channels of each file and save them as parquet.

    Safety for mobile internet: prints the exact size before downloading, refuses anything
    above data.max_download_mb, asks for confirmation unless assume_yes, skips cached parts.
    Timestamps are kept naive as in the source; the time zone is decided in QA.
    """
    s3 = get_s3_client()
    data = config["data"]
    parts = plan_sample(s3, config)
    total_mb = sum(p["end"] - p["start"] for p in parts) / 1e6
    for p in parts:
        print(f"{p['name']:14s} {p['month']}  {(p['end'] - p['start']) / 1e6:7.1f} MB")
    print(f"To download: {total_mb:.1f} MB "
          f"(probes and headers so far: {bytes_transferred / 1e6:.2f} MB)")

    if dry_run or not parts:
        return []
    if total_mb > data["max_download_mb"]:
        raise SystemExit(f"{total_mb:.1f} MB exceeds max_download_mb={data['max_download_mb']}. "
                         "Raise the limit in the config only after explicit approval.")
    if not assume_yes and input("Proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("Cancelled.")

    saved = []
    for p in parts:
        raw = _get_range(s3, p["key"], p["start"], p["end"])
        df = _parse_slice(raw, p, data["timestamp_column"])
        # Sanity checks: sorted rows and all timestamps inside the requested month.
        if not df.index.is_monotonic_increasing:
            raise RuntimeError(f"{p['name']} {p['month']}: timestamps not sorted.")
        if df.index.min() < pd.Timestamp(p["ts_start"]) or df.index.max() >= pd.Timestamp(p["ts_end"]):
            raise RuntimeError(f"{p['name']} {p['month']}: rows outside the month.")
        p["out_path"].parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p["out_path"])
        saved.append(p["out_path"])
        print(f"saved {p['out_path']}  rows={len(df)}  "
              f"{df.index.min()} .. {df.index.max()}  duplicates={df.index.duplicated().sum()}")
    print(f"Total transferred this run: {bytes_transferred / 1e6:.1f} MB")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Download configured PVDAQ months x channels.")
    parser.add_argument("--config", required=True, help="e.g. configs/system_9069.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="only estimate the size (reads headers and binary-search probes)")
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()
    download_sample(load_config(args.config), dry_run=args.dry_run, assume_yes=args.yes)


if __name__ == "__main__":
    main()
