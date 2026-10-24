"""Access to NREL PVDAQ data in the public OEDI S3 bucket (no AWS credentials).

Bucket layout (s3://oedi-data-lake/pvdaq/), as found on inspection:
- csv/systems_YYYYMMDD.csv           one-row-per-system summary, several dated versions
- parquet/{site,system,mount,...}/   normalised metadata tables, one parquet file each
- parquet/metrics/                   one file per system: list of measured channels
- 2023-solar-data-prize/<id>_OEDI/   six detailed systems, each with metadata/*.json
"""

from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

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
