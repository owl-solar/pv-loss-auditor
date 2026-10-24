# PV Loss Auditor

Diagnoses underperformance of a PV plant using measured data: compares actual
generation with weather-corrected expected generation, computes plant KPIs and
breaks the energy gap down by cause into a loss waterfall in kWh and USD
(expected → temperature → clipping → downtime → residual).

## Motivation
The output is designed like a monthly O&M performance report: energy vs expected,
Performance Ratio, availability, loss breakdown and recommendations.
Here the input is public PVDAQ files; in a real company the same pipeline would
connect to SCADA or a monitoring platform.

## Data
- **NREL PVDAQ** (Photovoltaic Data Acquisition), public OEDI S3 bucket
  `s3://oedi-data-lake/pvdaq/`, licence CC BY 4.0.
  DOI: [10.25984/1846021](https://doi.org/10.25984/1846021).
  Measured plant data: AC power, POA irradiance, module temperature.
  - **Station: 9069 (Simon Solar Farm, Social Circle, GA)**: fixed-tilt ground mount
    (20° / 180° per metadata, to be verified in QA), 38.7 MWp DC, 40 inverters, 5-min data.
    Chosen from a shortlist of five stations (`reports/station_candidates.md`).
  - **Analysis period: 2020-05 … 2023-11** (43 months): inverter AC power, the plant meter,
    three POA reference cells and four module-temperature sensors all report on at least 90 %
    of daylight snapshots in most months. Data coverage and sensor availability for the full
    2016–2023 record:
    [`notebooks/01_data_qa.ipynb`](notebooks/01_data_qa.ipynb), section "Data coverage and
    sensor availability" (figure: `reports/figures/9069_data_coverage.png`).
- **Open-Meteo**: historical weather, used as a satellite/reanalysis alternative
  to on-site sensors.
- **US electricity prices**, chosen to match the plant's location and connection type:
  - grid-connected plant in an ISO: ISO day-ahead LMP (via `gridstatus`);
  - behind-the-meter system: utility/state commercial tariff (EIA);
  - plant outside ISO regions: regional wholesale estimate (EIA).

  Losses are reported in USD (optional EUR conversion). The price assumption is
  stated explicitly in the results.

## Methods
- **pvlib**: expected yield model (ModelChain, temperature and inverter models).
- **pvanalytics**: data QA, clipping detection.
- **RdTools**: year-on-year degradation, soiling (SRR).
- **Isolation Forest** (scikit-learn) on residuals (measured minus modelled power),
  not on raw power.
- **KPIs** per IEC 61724-1: temperature-corrected Performance Ratio, specific yield,
  availability.

## Project status
- [x] Phase 0: repo skeleton, environment
- [ ] Phase 1: data selection and QA
- [ ] Phase 2: expected yield model and KPIs
- [ ] Phase 3: loss waterfall MVP (kWh and USD)
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
Download data (script to be added in Phase 1):
```powershell
# python -m src.download_data   (placeholder)
```

## Repository structure
```
data/raw/          # downloaded data (not committed)
data/processed/    # cleaned parquet files (not committed)
notebooks/         # analysis notebooks, one per phase
src/               # reusable functions (loading, QA, model, losses, anomalies)
app/               # Streamlit dashboard
reports/figures/   # figures used in this README
```

## How this project was built
Developed with AI assistance (Claude Code). I defined the methodology, station selection criteria and decision thresholds, reviewed all changes, and investigated data issues; the AI accelerated implementation.
