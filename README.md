# PV Loss Auditor

> Work in progress. This README will be rewritten as a case study
> (problem → method → result in EUR) in the final phase.

Diagnoses underperformance of a PV plant: compares actual generation with
weather-corrected expected generation, computes plant KPIs
(temperature-corrected Performance Ratio, specific yield, availability)
and breaks losses down by cause into a loss waterfall in kWh and EUR.

## Data sources
- **NREL PVDAQ** (public OEDI S3 bucket `oedi-data-lake/pvdaq`): measured plant data
- **Open-Meteo**: historical weather (satellite/reanalysis alternative to on-site sensors)
- **ENTSO-E / Nord Pool**: day-ahead electricity prices, Lithuania (LT bidding zone)

## Stack
Python, pandas, pvlib, pvanalytics, RdTools, scikit-learn, plotly, Streamlit.

## Setup
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Repo layout
```
data/raw/          # downloaded data (not committed)
data/processed/    # cleaned parquet files (not committed)
notebooks/         # analysis notebooks, one per phase
src/               # reusable functions (loading, qa, model, losses, anomalies)
app/               # Streamlit dashboard
reports/figures/   # figures used in this README
```

## Status
- [x] Phase 0: repo skeleton, environment
- [ ] Phase 1: data selection and QA
- [ ] Phase 2: expected yield model and KPIs
- [ ] Phase 3: loss waterfall MVP (kWh and EUR)
- [ ] Phase 4: degradation and soiling (RdTools); on-site sensor vs Open-Meteo
- [ ] Phase 5: anomaly detection on residuals
- [ ] Phase 6: Streamlit dashboard and case-study write-up
