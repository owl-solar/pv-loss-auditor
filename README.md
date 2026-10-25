# PV Loss Auditor

Diagnoses underperformance of a PV plant from measured data: compares actual generation with a
weather-corrected expected generation, computes plant KPIs and breaks the energy gap down by cause
into a loss waterfall in kWh and USD.

**Status: MVP (v0.1).** Phases 0-3 are done: data QA, expected-yield model, KPIs and the loss
waterfall for one plant. Degradation / soiling, anomaly detection and the dashboard are next
(see [Project status](#project-status)).

## Result in one table

Plant: PVDAQ 9069, Simon Solar Farm (GA), 38.7 MWp DC / 33 MW AC, fixed tilt, 40 inverters.
Period 2020-05 ... 2023-11. Energy at the plant meter against the nameplate energy at the measured
irradiance:

| Cause | GWh | USD thousand (base) | USD thousand (band ±30 %) |
|---|---|---|---|
| Design (temperature, assumed DC losses, inverter efficiency, clipping) | 35.9 | 1085 | 760 ... 1410 |
| **Downtime**, net of data-channel loss | 23.4 | 706 | 494 ... 917 |
| **Running shortfall** (inverters that run, against the model) | 19.9 | 600 | 420 ... 780 |
| After inverters (transformer, cables) and meter reconciliation | 3.4 | 103 | 72 ... 134 |

Downtime plus running shortfall, the part the audit points at: **43.3 GWh, 1.31 M USD
(0.91 ... 1.70 M)**. The "design" block is the plant as built (and partly an assumption), not a
recoverable loss.

![Loss waterfall by year](reports/figures/9069_loss_waterfall.png)

What the waterfall shows:
- **Downtime is the story of 2022-2023:** 3 % of the nominal energy in 2020-2021, 14 % in 2022,
  26 % in 2023. In 2023 mostly plant-wide events (>= 10 of 40 inverters down at once).
- **Data-channel loss in 2023: 4.8 GWh.** The meter reports energy that the inverter channels do
  not (channels write 0 while the inverter runs). A daily reconciliation against the meter credits
  it back, so these zeros are not counted as downtime.
- **Running shortfall 7.6-8.7 % of nominal**, about 0.91 of the model above 400 W/m2 in 2020-2022
  and 3 percentage points lower in 2023 (a real step, not a sensor effect). It mixes soiling,
  degradation, AOI / low-light behaviour and the error of the assumed design losses; splitting it is
  Phase 4.
- **Clipping is negligible** (<= 0.08 % of nominal). **No curtailment plateau** was found.

Read every USD figure with these limits: the price is a **projection** (Georgia Power avoided energy
cost for 2026, 30.19 USD/MWh) applied flat to 2020-2023 - realised wholesale prices in 2021-2022 were
higher - and it is the **utility's avoided cost, not this plant's PPA price** (unknown). The ranking
of the causes in kWh does not depend on the price. Full discussion, checks and sensitivities:
[`notebooks/03_loss_waterfall.ipynb`](notebooks/03_loss_waterfall.ipynb).

### Limits of the MVP
- The expected-yield model uses nameplate data and typical design losses (6.34 %, no PVsyst report
  for this plant); it is **not calibrated** on the data. At 4 % / 8 % design losses the running
  shortfall is 24.2 / 16.8 GWh instead of 19.9; downtime changes by less than 0.7 GWh.
- The waterfall covers the 5-min steps with daylight, irradiance, cell temperature and meter known
  (95.7-99.2 % of daylight steps per year), so annual totals are understated by up to 4 %.
- The data-channel credit assumes a constant inverter-to-meter factor of 1.022; 1 percentage point of
  it moves the 2023 credit by about 0.35 GWh. It cannot say *which* inverter writes false zeros.
- Inverter-based availability and PR including downtime are understated in 2023 (channel loss).

## Motivation
The output is designed like a monthly O&M performance report: energy vs expected, Performance Ratio,
availability, loss breakdown and recommendations. Here the input is public PVDAQ files; in a real
company the same pipeline would connect to SCADA or a monitoring platform.

## Data
- **NREL PVDAQ** (Photovoltaic Data Acquisition), public OEDI S3 bucket
  `s3://oedi-data-lake/pvdaq/`, licence CC BY 4.0.
  DOI: [10.25984/1846021](https://doi.org/10.25984/1846021).
  Measured plant data: AC power, POA irradiance, module temperature.
  - **Station: 9069 (Simon Solar Farm, Social Circle, GA)**: fixed-tilt ground mount
    20° / 180° (verified on clear days), 38.7 MWp DC, 40 × 825 kW inverters (33 MW AC),
    5-min data. Chosen from a shortlist of five stations (`reports/station_candidates.md`).
  - **Analysis period: 2020-05 ... 2023-11** (43 months): inverter AC power, the plant meter,
    three POA reference cells and three module-temperature sensors report on at least 90 % of
    daylight steps in most months. Data coverage and sensor availability for the full 2016-2023
    record: [`notebooks/01_data_qa.ipynb`](notebooks/01_data_qa.ipynb), section "Data coverage and
    sensor availability" (figure: `reports/figures/9069_data_coverage.png`).
  - **Data QA findings** (notebook 01, report [`reports/9069_qa_report.md`](reports/9069_qa_report.md)):
    - the files use different clocks: irradiance and environment switch daylight saving time on
      EU dates, inverter and meter data on US dates - without a per-file conversion to UTC, power
      is shifted by 1 h against irradiance for about 3 weeks a year
      (`reports/figures/9069_clock_offset.png`);
    - the NREL flag "Wrong mounting config" is not supported: fixed 20°/180° fits measured POA
      with nRMSE 4.9 %, a single-axis tracker with 36 % (`reports/figures/9069_geometry_check.png`);
    - the string metadata (272 vs 68 strings per inverter) contradicts equal clear-day energy of
      all inverters; one thermocouple reports implausible values for the whole period; several
      inverters have frozen readings - all flagged, nothing deleted.
- **Open-Meteo**: historical weather, to be used as a satellite / reanalysis alternative to on-site
  sensors (Phase 4).
- **Electricity price** (`src/prices.py`): 9069 is utility-scale and front-of-meter, in Georgia, which
  has no ISO / RTO and so no nodal prices (`gridstatus`) and no EIA wholesale hub. The price is the
  Georgia Power avoided energy cost 2026 for "Solar Ground-Mounted Fixed Tilt" from the 2025
  projections filed with the Georgia PSC (filing #223157): flat 30.19 USD/MWh, reported with a
  ±30 % band. Limits as stated above.

## Methods
- **pvlib**: expected-yield model with the PVWatts DC and inverter functions and the cell-temperature
  function called directly (not ModelChain), so every step is visible
  ([`src/model.py`](src/model.py)). POA = mean of two reference cells; cell temperature = mean of
  three back-of-module thermocouples + 3 °C × G/1000.
- **KPIs** per IEC 61724-1 ([`src/kpi.py`](src/kpi.py)): Performance Ratio at inverter-step level
  (ratio of sums) in three temperature treatments (raw, T_ref 25 °C, T_ref = POA-weighted mean cell
  temperature of the period), specific yield, time-based availability. PR of running inverters (T_ref
  of the period): 0.762 / 0.762 / 0.761 / 0.735 for 2020-2023.
- **Loss waterfall** ([`src/losses.py`](src/losses.py)): a sequential cascade in physical order, each
  step the difference of two neighbouring cumulative quantities, so the steps add up to
  (nominal - metered) by construction; a closure check stops the run otherwise. Downtime is separated
  from data-channel loss by a daily reconciliation of the inverter sum against the plant meter.
- **To come:** pvanalytics (QA, clipping detection), RdTools (year-on-year degradation, soiling),
  Isolation Forest on residuals (measured minus modelled power), not on raw power.

## Project status
- [x] Phase 0: repo skeleton, environment
- [x] Phase 1: data selection and QA
- [x] Phase 2: expected yield model and KPIs
- [x] Phase 3: loss waterfall MVP (kWh and USD)
- [ ] Phase 4: degradation and soiling (RdTools); on-site sensor vs Open-Meteo
- [ ] Phase 5: anomaly detection on residuals; synthetic string faults
- [ ] Phase 6: Streamlit dashboard; case-study write-up

## How to reproduce
Windows PowerShell, Python 3.13:
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Download the analysis period of PVDAQ system 9069 (public OEDI S3 bucket, no credentials;
about 1.2 GB, resumes month by month after an interruption), then build the clean dataset:
```powershell
python -m src.pvdaq --config configs/system_9069.yaml --period --dry-run   # size estimate only
python -m src.pvdaq --config configs/system_9069.yaml --period
python -m src.clean --config configs/system_9069.yaml                       # -> data/processed/9069/
```
The price file (Georgia PSC filing #223157, 249 kB) goes to `data/raw/prices/` (see `price:` in the
config). Then run the notebooks in order: `notebooks/01_data_qa.ipynb` (QA, clock, geometry; report
`reports/9069_qa_report.md`), `02_expected_yield.ipynb` (model and KPIs),
`03_loss_waterfall.ipynb` (waterfall, checks, USD).

Tests (synthetic data only, no download needed):
```powershell
python -m pytest
```

## Repository structure
```
data/raw/          # downloaded data (not committed)
data/processed/    # cleaned parquet files (not committed)
notebooks/         # analysis notebooks, one per phase
src/               # reusable functions (loading, QA, model, KPIs, losses, prices)
tests/             # unit tests on synthetic inputs
app/               # Streamlit dashboard (Phase 6)
reports/figures/   # figures used in this README
```

## How this project was built
Developed with AI assistance (Claude Code). I defined the methodology, station selection criteria and decision thresholds, reviewed all changes, and investigated data issues; the AI accelerated implementation.
