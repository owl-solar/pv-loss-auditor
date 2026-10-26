"""Access to NREL PVDAQ data in the public OEDI S3 bucket (no AWS credentials).

Bucket layout (s3://oedi-data-lake/pvdaq/), as found on inspection:
- csv/systems_YYYYMMDD.csv           one-row-per-system summary, several dated versions
- parquet/{site,system,mount,...}/   normalised metadata tables, one parquet file each
- parquet/metrics/                   one file per system: list of measured channels
- 2023-solar-data-prize/<id>_OEDI/   six detailed systems, each with metadata/*.json
"""

import logging
import argparse
import io
import json
import os
import re
import threading
import time
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from pv_loss_auditor.config import load_config
from pv_loss_auditor.logs import setup_logging

log = logging.getLogger(__name__)

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
_traffic_lock = threading.Lock()  # src/pv_loss_auditor/coverage.py reads in parallel threads


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
                log.info(f"cached, skip: {out_path}")
                continue
            ts_start, ts_end = _month_bounds(month)
            start = find_offset(s3, spec["key"], header["data_start"], header["size"], ts_start)
            end = find_offset(s3, spec["key"], header["data_start"], header["size"], ts_end)
            parts.append({"name": name, "month": month, "key": spec["key"], "start": start,
                          "end": end, "ts_start": ts_start, "ts_end": ts_end,
                          "columns": header["columns"], "channels": spec["channels"],
                          "out_path": out_path})
    return parts


def _parse_rows(raw, columns, channels, ts_col):
    """Parse CSV rows (bytes, whole rows, no header line).

    columns : all column names of the file, in file order (the slice has no header)
    channels: {raw column name: project name}, the columns to keep
    Returns a DataFrame indexed by the (naive) timestamp with the renamed channel columns.
    """
    df = pd.read_csv(io.BytesIO(raw), header=None, names=columns, usecols=[ts_col, *channels])
    df[ts_col] = pd.to_datetime(df[ts_col], format="%Y-%m-%d %H:%M:%S")
    df = df.set_index(ts_col).rename(columns=channels)
    return df[list(channels.values())]


def _parse_slice(raw, part, ts_col):
    """Parse a downloaded CSV slice of a sample part (see plan_sample)."""
    return _parse_rows(raw, part["columns"], part["channels"], ts_col)


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
        log.info(f"{p['name']:14s} {p['month']}  {(p['end'] - p['start']) / 1e6:7.1f} MB")
    log.info(f"To download: {total_mb:.1f} MB "
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
        log.info(f"saved {p['out_path']}  rows={len(df)}  "
              f"{df.index.min()} .. {df.index.max()}  duplicates={df.index.duplicated().sum()}")
    log.info(f"Total transferred this run: {bytes_transferred / 1e6:.1f} MB")
    return saved


# ---------------------------------------------------------------------------
# Resumable download of a whole period, month by month
# ---------------------------------------------------------------------------
# Idea: read the CSV from the start of the period to its end in fixed-size chunks and cut the
# stream into months by the timestamp at the start of each row. A month is written to its own
# parquet file as soon as the first row of the NEXT month appears. A manifest (JSON) records the
# byte range of every saved month, so after an interruption we continue exactly where the last
# saved month ended, and nothing that is already saved is downloaded again.

MONTH_RE = re.compile(r"^\d{4}-\d{2}$")


def resolve_channels(columns, rules):
    """Map raw CSV column names to project names with the regex rules of the config.

    Fails loudly if a rule matches no column or two raw columns get the same project name,
    so a wrong pattern cannot silently drop or merge channels.
    """
    mapping = {}
    for rule in rules:
        pattern = re.compile(rule["pattern"])
        hits = [(c, pattern.match(c)) for c in columns if pattern.match(c)]
        if not hits:
            raise ValueError(f"Rule {rule['pattern']!r} matches no column.")
        for column, match in hits:
            if column in mapping:
                raise ValueError(f"Column {column!r} matched by two rules.")
            mapping[column] = rule["name"].format(*match.groups())
    names = list(mapping.values())
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Several raw columns map to the same name: {duplicates}")
    return mapping


def _write_json_atomic(path, obj):
    """Write JSON to a temp file, then replace the target: a crash never leaves a half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)  # atomic on the same drive


def _manifest_path(period):
    return Path(period["out_dir"]) / "manifest.json"


def _load_manifest(period):
    path = _manifest_path(period)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"files": {}}


def _get_range_retry(s3, key, start, end, retries):
    """_get_range with retries and growing pauses (mobile connections drop).

    Also treats a short read as an error: S3 must return exactly end - start bytes.
    """
    for attempt in range(retries + 1):
        try:
            body = _get_range(s3, key, start, end)
            if len(body) != end - start:
                raise OSError(f"short read: {len(body)} of {end - start} bytes")
            return body
        except (BotoCoreError, ClientError, OSError) as exc:
            if attempt == retries:
                raise
            wait = 2 ** (attempt + 1)
            log.info(f"  network error ({type(exc).__name__}), retry {attempt + 1}/{retries} "
                  f"in {wait} s", flush=True)
            time.sleep(wait)


def _read_chunks(s3, key, start, size, chunk_bytes, retries):
    """Yield (offset, block) from byte `start` to the end of the file.

    Every block consists of WHOLE rows only: it starts at the beginning of a row and ends right
    after a newline. A chunk boundary usually falls in the middle of a row, so the unfinished
    tail of each chunk (everything after its last newline) is kept in `carry` and put in front
    of the next chunk. `start` must be the beginning of a row.
    """
    carry = b""
    pos = start  # next byte to download
    while pos < size:
        end = min(pos + chunk_bytes, size)
        chunk = _get_range_retry(s3, key, pos, end, retries)
        data = carry + chunk
        data_offset = pos - len(carry)  # file offset of data[0], which is the start of a row
        pos = end
        cut = data.rfind(b"\n") + 1  # bytes up to and including the last newline (0 if none)
        carry = data[cut:]
        if cut:
            yield data_offset, data[:cut]
    if carry:  # last row of a file that does not end with a newline
        yield pos - len(carry), carry


def plan_period(s3, config, files=None):
    """Prepare the download: per file the start offset, the resume offset and the exact size.

    New file -> binary search for the first row of period.start (a few hundred KB).
    File in the manifest -> resume right after its last saved month, no search.
    Returns (plans, manifest); also writes the manifest, so a started plan is never repeated.
    """
    period, data = config["period"], config["data"]
    manifest = _load_manifest(period)
    start_ts = pd.Period(period["start"], freq="M").start_time.strftime("%Y-%m-%d %H:%M:%S")
    plans = []
    for name in files or period["order"]:
        key = data["files"][name]["key"]
        header = _cached_header(s3, key, data["header_cache"])
        size_now = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"]
        if size_now != header["size"]:
            raise RuntimeError(f"{key}: size on S3 ({size_now}) differs from cached header "
                               f"({header['size']}); delete {data['header_cache']} and check.")
        mapping = resolve_channels(header["columns"], period["rules"][name])
        entry = manifest["files"].get(name)
        if entry is None:
            before = bytes_transferred
            start_offset = find_offset(s3, key, header["data_start"], size_now, start_ts)
            entry = {"key": key, "size": size_now, "start_month": period["start"],
                     "start_offset": start_offset, "plan_bytes": bytes_transferred - before,
                     "traffic_bytes": 0, "channels": mapping, "months": []}
            manifest["files"][name] = entry
        elif (entry["size"], entry["channels"], entry["start_month"]) != \
                (size_now, mapping, period["start"]):
            raise SystemExit(f"{name}: file size, channels or start month differ from the "
                             f"manifest. Delete {Path(period['out_dir']) / name} and its entry "
                             "in the manifest to start this file over.")
        resume = entry["months"][-1]["end"] if entry["months"] else entry["start_offset"]
        plans.append({"name": name, "entry": entry, "header": header, "mapping": mapping,
                      "size": size_now, "resume": resume})
    _write_json_atomic(_manifest_path(period), manifest)
    return plans, manifest


class _MonthSaver:
    """Turns the raw rows of one month into a parquet file and records it in the manifest."""

    def __init__(self, label, plan, config, manifest, progress):
        self.label, self.plan, self.manifest, self.progress = label, plan, manifest, progress
        self.name, self.entry = plan["name"], plan["entry"]
        self.ts_col = config["data"]["timestamp_column"]
        self.out_dir = Path(config["period"]["out_dir"]) / self.name
        self.manifest_path = _manifest_path(config["period"])
        self.traffic_mark = bytes_transferred  # traffic already counted for this file

    def save(self, month, lines, start, end, at_eof):
        df = _parse_rows(b"".join(lines), self.plan["header"]["columns"],
                         self.plan["mapping"], self.ts_col)
        # Sanity checks: sorted rows, all inside the month (stop instead of saving bad data).
        if not df.index.is_monotonic_increasing:
            raise RuntimeError(f"{self.name} {month}: timestamps not sorted.")
        if (df.index.to_period("M") != pd.Period(month, freq="M")).any():
            raise RuntimeError(f"{self.name} {month}: rows outside the month.")

        path = self.out_dir / f"{self.name}_{month}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        df.to_parquet(tmp)
        os.replace(tmp, path)  # parquet first, manifest second: a crash in between is harmless

        self.entry["traffic_bytes"] += bytes_transferred - self.traffic_mark
        self.traffic_mark = bytes_transferred
        self.entry["months"].append({
            "month": month, "start": start, "end": end, "rows": len(df),
            "first_ts": str(df.index.min()), "last_ts": str(df.index.max()),
            "duplicates": int(df.index.duplicated().sum()), "at_eof": at_eof})
        _write_json_atomic(self.manifest_path, self.manifest)
        self._print_progress(month, end - start, end)

    def _print_progress(self, month, month_bytes, end):
        p = self.progress
        p["done"] += month_bytes
        file_total = self.plan["size"] - self.entry["start_offset"]
        file_done = end - self.entry["start_offset"]
        speed = (bytes_transferred - p["traffic0"]) / max(time.time() - p["t0"], 1e-9)
        eta_min = (p["total"] - p["done"]) / max(speed, 1) / 60
        log.info(f"{self.label} {self.name:13s} {month}  {month_bytes / 1e6:6.1f} MB | "
              f"file {file_done / 1e6:6.1f}/{file_total / 1e6:.1f} MB "
              f"({100 * file_done / file_total:3.0f}%) | "
              f"total {p['done'] / 1e6:7.1f}/{p['total'] / 1e6:.1f} MB | "
              f"{speed / 1e6:.2f} MB/s | ETA {eta_min:.0f} min", flush=True)


def _download_file(s3, plan, label, config, manifest, progress):
    """Download one file from its resume offset to the end, saving each month when complete."""
    period = config["period"]
    saver = _MonthSaver(label, plan, config, manifest, progress)
    chunk_bytes = int(period["chunk_mb"] * 1024 * 1024)
    key = plan["entry"]["key"]
    cur_month, month_start, lines = None, None, []
    offset = plan["resume"]  # file offset of the row being looked at
    for block_offset, block in _read_chunks(s3, key, plan["resume"], plan["size"],
                                            chunk_bytes, period["retries"]):
        offset = block_offset
        for line in block.splitlines(keepends=True):
            if line.strip():  # skip blank lines
                month = line[:7].decode("ascii", errors="replace")  # 'YYYY-MM' of the row
                if not MONTH_RE.match(month):
                    raise ValueError(f"{key}: unexpected row at byte {offset}: {line[:60]!r}")
                if month != cur_month:
                    if cur_month is not None:
                        if month < cur_month:
                            raise ValueError(f"{key}: time goes backwards at byte {offset}.")
                        # First row of the next month: the current month is complete and ends
                        # exactly here. The next run would resume at this very byte.
                        saver.save(cur_month, lines, month_start, offset, at_eof=False)
                        lines = []
                    cur_month, month_start = month, offset
                lines.append(line)
            offset += len(line)
    if cur_month is not None:  # end of file: the last month ends with the file
        saver.save(cur_month, lines, month_start, offset, at_eof=True)


def period_report(config):
    """One row per file from the manifest: months, rows, MB, first/last timestamp, gaps."""
    manifest = _load_manifest(config["period"])
    rows = []
    for name, entry in manifest["files"].items():
        months = entry["months"]
        if not months:
            continue
        first = min(m["first_ts"] for m in months)
        last = max(m["last_ts"] for m in months)
        n_rows = sum(m["rows"] for m in months)
        expected = (pd.Timestamp(last) - pd.Timestamp(first)) / pd.Timedelta(minutes=5) + 1
        have = pd.PeriodIndex([m["month"] for m in months], freq="M")
        missing = pd.period_range(have.min(), have.max(), freq="M").difference(have)
        rows.append({
            "file": name, "months": len(months), "rows": n_rows,
            "rows_pct_of_5min": round(100 * n_rows / expected, 1),
            "duplicates": sum(m["duplicates"] for m in months),
            "payload_mb": round(sum(m["end"] - m["start"] for m in months) / 1e6, 1),
            "traffic_mb": round((entry["plan_bytes"] + entry["traffic_bytes"]) / 1e6, 1),
            "first_ts": first, "last_ts": last,
            "missing_months": ",".join(str(p) for p in missing)})
    return pd.DataFrame(rows)


def download_period(config, files=None, dry_run=False):
    """Download the analysis period of the configured files, resuming after the last saved month.

    The exact size is known before the first byte of data is requested (file size minus the
    start offset). The run stops if it exceeds period.approved_mb.
    """
    period = config["period"]
    s3 = get_s3_client()
    plans, manifest = plan_period(s3, config, files)

    total = sum(p["size"] - p["entry"]["start_offset"] for p in plans)
    todo = sum(p["size"] - p["resume"] for p in plans)
    log.info(f"{'file':14s} {'period MB':>10s} {'saved MB':>9s} {'to download MB':>15s}  months saved")
    for p in plans:
        full = p["size"] - p["entry"]["start_offset"]
        left = p["size"] - p["resume"]
        log.info(f"{p['name']:14s} {full / 1e6:10.1f} {(full - left) / 1e6:9.1f} {left / 1e6:15.1f}  "
              f"{len(p['entry']['months'])}")
    log.info(f"To download: {todo / 1e6:.1f} MB (approved: {period['approved_mb']} MB; "
          f"headers and search so far: {bytes_transferred / 1e6:.2f} MB)")
    if todo / 1e6 > period["approved_mb"]:
        raise SystemExit("Exceeds period.approved_mb. Ask for approval before raising it.")
    if dry_run or todo == 0:
        return

    progress = {"total": total, "done": total - todo, "t0": time.time(),
                "traffic0": bytes_transferred}
    try:
        for i, plan in enumerate(plans, 1):
            if plan["resume"] < plan["size"]:
                _download_file(s3, plan, f"[{i}/{len(plans)}]", config, manifest, progress)
    except BaseException:
        log.warning("Interrupted. Run the same command again: it continues after the last saved "
              "month, saved months are not downloaded again.", flush=True)
        raise

    report = period_report(config)
    log.info("\nDownload log (cumulative over all runs):")
    log.info(report.to_string(index=False))
    report.to_csv(Path(period["out_dir"]) / "download_log.csv", index=False)
    log.info(f"Transferred in this run: {bytes_transferred / 1e6:.1f} MB")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Download configured PVDAQ data.")
    parser.add_argument("--config", required=True, help="e.g. configs/system_9069.yaml")
    parser.add_argument("--period", action="store_true",
                        help="download the whole analysis period, resumable, month by month "
                             "(default: the small sample of data.sample_months)")
    parser.add_argument("--files", nargs="+", help="with --period: only these files")
    parser.add_argument("--dry-run", action="store_true",
                        help="only print the size (reads headers and binary-search probes)")
    parser.add_argument("--yes", action="store_true",
                        help="sample mode: do not ask for confirmation")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.period:
        download_period(config, files=args.files, dry_run=args.dry_run)
    else:
        download_sample(config, dry_run=args.dry_run, assume_yes=args.yes)


if __name__ == "__main__":
    main()
