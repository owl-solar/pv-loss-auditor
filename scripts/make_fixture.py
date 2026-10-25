"""Cut a small slice of the REAL clean dataset of 9069 for tests and CI (no network needed).

The slice keeps only the columns that `src.model.load_inputs` reads, for 14 UTC days. The window
2020-12-08 ... 2020-12-21 contains the real plant-wide stop of 2020-12-12 ... 12-18 (26 inverters
down), so availability, downtime and the waterfall are exercised on real outage data.
Values are copied as measured: nothing is edited or filled in. Run from the project root:

    python scripts/make_fixture.py --config configs/system_9069.yaml
"""

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from src.config import load_config
from src.model import inverter_columns

START, END = "2020-12-08", "2020-12-22"      # END exclusive: 14 days
OUT = Path("tests/fixtures/9069_clean")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    m = config["model"]
    meter = config["kpi"]["meter_channel"]

    src_file = Path(config["clean"]["out_dir"]) / "9069_clean_2020-12.parquet"
    names = pq.read_schema(src_file).names
    sensors = m["poa_channels"] + m["t_mod_channels"] + [m["t_amb_channel"], meter]
    wanted = sensors + inverter_columns(names) + ["flag_daylight", "sun_elevation_deg", "poa_clearsky_wm2"]
    flags = [c for c in names
             if c.startswith(("flag_stale__", "flag_range__")) and c.split("__", 1)[1] in wanted]
    data = pd.read_parquet(src_file, columns=wanted + flags)
    tz = data.index.tz
    piece = data[(data.index >= pd.Timestamp(START, tz=tz)) & (data.index < pd.Timestamp(END, tz=tz))]

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "9069_clean_2020-12.parquet"
    piece.to_parquet(path, compression="zstd")
    print(f"{len(piece)} rows, {piece.index.min()} .. {piece.index.max()}, {path.stat().st_size / 1024:.0f} kB")


if __name__ == "__main__":
    main()
