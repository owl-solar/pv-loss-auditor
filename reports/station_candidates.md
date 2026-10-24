# PVDAQ station candidates

Shortlist for Phase 1, screened against the selection criteria in `CLAUDE.md`.
Source: PVDAQ metadata only (no time series downloaded), OEDI bucket `s3://oedi-data-lake/pvdaq/`:
`csv/systems_20250729.csv`, `parquet/{metrics,inverters,mount}/`,
`2023-solar-data-prize/<id>_OEDI/metadata/*_system_metadata.json`.
"not in metadata" means the field does not exist or is empty; nothing is inferred.

## Candidates

| ID | Name | State | Connection type | DC, kW | AC, kW | Tilt / azimuth | Period (first – last timestamp) | Resolution | POA | Module temp | AC power | Inverter-level |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9069 | Simon Solar Farm, Social Circle | GA | not in metadata | 33 000 (CSV) / 38 687 (JSON) | `AC capacity(kW)` = 825 for 5 of 40 inverters; empty for the other 35 | 20° / 180°, fixed, ground | 2016-02-17 – 2023-11-29 (7.8 y) | 5 min | 13 | 15 | 45 | yes, 40 inverters |
| 9068 | SR_CO, Kersey | CO | not in metadata | 4 738 | 2 × 1 910 = 3 820 (`AC capacity(kW)`) | single-axis tracker (axis tilt 0° / az 180°) | 2017-08-23 – 2025-04-30 (CSV) / 2023-11-16 (JSON) | 5 min | 4 | 4 | 14 | yes, 2 inverters (10 inverter-tagged AC power channels) |
| 1283 | NREL Research Support Facility II, Golden | CO | not in metadata | 408.2 | not in metadata (inverters: 2 × SMA Sunny Central 250U) | 10° / 165°, fixed, roof | 1997-02-19 – 2024-05-07 (27.2 y) ¹ | not in metadata | 2 (pyranometer + ref. cell) | 1 | 6 (inv1, inv2, total, 3 meter) | yes, 2 inverters |
| 1200 | Distributed Sun – BWI Hilton, Linthicum Heights | MD | not in metadata | 51.8 | not in metadata (inverters: 6 × SMA SB8000W) | 10° / 205°, fixed, roof | 2010-10-03 – 2020-07-26 (9.8 y) | not in metadata | 1 | 1 (`module_temp_F`, °F) | 8 (inv1–inv6, total, meter) | yes, 6 inverters |
| 4902 | NIST Ground 1, Gaithersburg | MD | not in metadata | 270.7 | not in metadata (inverter: 1 × PV Powered PVP 260kW) | 20° / 180°, fixed, ground | 2014-07-29 – 2018-03-14 (3.6 y) | not in metadata | 3 | 1 + 10 backsheet RTDs | 6 (1 inverter + 5 meter channels ²) | no, single central inverter |

Notes:
- **Connection type** (ISO grid-connected / behind-the-meter) is not a metadata field for any
  candidate. Mounting type (roof / ground) is available but does not determine it. To be
  established from external sources before pricing (Phase 3).
- **Period** is the span between first and last timestamp, not data coverage. Most systems carry
  NREL's automated QA flag "Filtered time series less than 1.0 years data"; real coverage is
  checked in Phase 1.
- **AC capacity**: for 1283, 1200 and 4902 only inverter model and quantity are given; a nominal
  rating appears in some model names but is not a metadata value.
- ¹ 1283: first timestamp 1997 is older than the building; likely a metadata error, to verify.
- ² 4902: some channels labelled "AC power" are meter energy registers (`PwrMtrEdel_kWh_Max`,
  `..._kVARh_Max`), not power.
- CSV and Data Prize JSON disagree for 9069 (DC capacity) and 9068 (last timestamp).
- NREL QA flags include "Wrong mounting config identified" and "correlation coefficient < 0.5"
  for 9069 and 4902, and "Percent clipping exceeded threshold of 10%" for 4902 and 1283.
- Approximate dataset size: 9069 ≈ 23 GB, 1283 ≈ 8.3 GB, 4902 ≈ 4.2 GB, 9068 ≈ 1.5 GB, 1200 ≈ 0.15 GB.

Excluded but notable:
- **7333** Shine On Solar Facility, CA: 205–258 MW, trackers, 112 inverters; 10-second data
  (≈ 273 GB; 5-min copy ≈ 34 GB); AC capacity not in metadata.
- **2107** Farm Solar Array, CA: 893 kW, 25° / 180°, 24 × ABB TRIO-27.6; no module temperature channel.

## Screening check

1860 unique systems in metadata (1862 CSV rows; 2 systems have one row per mount). Passed each
criterion independently → cumulatively: years ≥ 3: 1690 → 1690; channel list available: 161 → 121;
POA: 64 → 56; module temp (module or backsheet): 63 → 51; AC power: 106 → 39;
tilt & azimuth known: 1570 → 33; DC capacity known: 1795 → 33; AC capacity field: 2 → 2;
inverter-level AC power (≥ 2 channels): 19 → 2 (9068, 9069).
