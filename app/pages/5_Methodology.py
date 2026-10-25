"""Methodology: data, model, definitions, assumptions that are mine, and what the numbers do not say."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # `common` lives in app/
import common  # noqa: E402
from src.config import load_config  # noqa: E402

st.set_page_config(page_title="Methodology · PV Loss Auditor", page_icon="☀️", layout="wide")
config = load_config(common.ROOT / "configs" / "system_9069.yaml")
info = common.meta()
price = info["price_usd_per_mwh"]

st.title("Methodology")
st.markdown("What the audit does, which numbers are measured, which are my assumptions, and what the figures in this "
            "app do **not** say.")

# --- pipeline --------------------------------------------------------------------------------
st.header("From raw files to the numbers shown here")
st.markdown(
    "1. **Data QA** of the public PVDAQ files: time zones, gaps, placeholders, frozen readings, implausible sensors. "
    "Nothing is deleted: values stay as measured with `flag_*` columns.\n"
    "2. **Expected generation** from measured irradiance and module temperature (pvlib PVWatts functions called "
    "one by one, so every step is visible).\n"
    "3. **KPIs:** temperature-corrected Performance Ratio, specific yield, availability.\n"
    "4. **Loss waterfall:** nominal energy → energy at the plant meter, in kWh and USD.\n"
    "5. **Degradation and soiling** (RdTools) and a comparison with reanalysis weather (Open-Meteo).\n"
    "6. **Anomaly detection** on residuals against the other inverters, tested on SYNTHETIC faults.\n"
    "7. **Export:** `python -m src.export_dashboard` saves the small tables in `app/data/`; this app only reads them.")

# --- plant and data --------------------------------------------------------------------------
st.header("Plant and data")
plant = pd.DataFrame([
    ("Plant", info["plant"]),
    ("DC capacity", f"{info['dc_kw']:,.0f} kW (25 992 × 295 W + 103 398 × 300 W modules)"),
    ("AC capacity", f"{info['ac_kw']:,.0f} kW = 40 inverters × 825 kW (every inverter clips at 825-830 kW)"),
    ("Mounting", "fixed tilt, 20° / 180°: POA nRMSE 4.9 % on clear days against 36 % for a tracker; "
                 "the NREL flag 'Wrong mounting config' is not supported by the data"),
    ("Strings", "metadata (272 vs 68 per inverter) contradicts equal clear-day energy of all inverters; "
                "170 strings per inverter is derived from the DC power"),
    ("Period", "2020-05 ... 2023-11 (43 months, 5-minute data), chosen from the coverage map of 2016-2023; "
               "2018-2020 was rejected (thermocouples work on 0-57 % of the time, a meter gap of 357 days)"),
    ("Source", "NREL PVDAQ, system 9069 (OEDI S3 bucket), CC BY 4.0, DOI 10.25984/1846021"),
], columns=["item", "value"])
st.dataframe(plant, hide_index=True, width="stretch")
st.markdown("**QA findings that change the numbers:**\n"
            "- The files use different clocks: irradiance and environment switch daylight saving time on EU dates, "
            "inverter and meter files on US dates. Without a conversion per file, power is shifted by 1 h against "
            "irradiance for about three weeks a year.\n"
            "- Thermocouple 03 reads implausible values for the whole period (45-80 °C at night) and is excluded.\n"
            "- Inverters 02, 19-22, 25 and 26 have frozen readings on 28-82 days (flagged, not deleted).\n"
            "- Sum of inverters / plant meter is 1.022 when all channels report, 0.76-0.95 from 2023-02: part of the "
            "zeros in 2023 are lost data channels, not stops.")

# --- expected yield --------------------------------------------------------------------------
st.header("Expected generation")
gamma = info["gamma_pct_per_c"]
losses = pd.Series(config["model"]["dc_losses_pct"]).rename("%")
left, right = st.columns([3, 2])
left.markdown(
    "- **Irradiance (POA):** mean of reference cells 04 and 10 (cell 01 reads 3.5-5.4 % low and is not used).\n"
    "- **Cell temperature:** mean of three back-of-module thermocouples (01, 06, 10) + 3 °C × G/1000. No modelled "
    "fallback (measured temperature exists on 97.7 % of daylight steps).\n"
    f"- **DC:** pvlib `pvwatts_dc`, 967.2 kW per inverter, γ = {gamma:.3f} %/°C (DC-weighted nameplate; "
    "measured −0.34 ± 0.04 %/°C per inverter).\n"
    "- **Design DC losses** (right), combined multiplicatively. Soiling, shading, snow, availability and age are set "
    "to zero on purpose: they are what the audit measures.\n"
    f"- **Inverter:** pvlib `inverter.pvwatts`, η_nom {config['model']['inverter']['eta_inv_nom']}, clipped at 825 kW.\n"
    "- **Not calibrated on the data.**")
total_loss = (1 - (1 - losses / 100).prod()) * 100          # multiplicative, as pvlib.pvwatts_losses
right.dataframe(pd.concat([losses, pd.Series({"total (combined)": total_loss})]).to_frame("%").style.format("{:.2f}"),
                width="stretch")
st.warning("**The 6.34 % design losses are an assumption**, not a PVsyst result (there is no report for this plant). "
           "At 4 % / 8 % the running shortfall is 24.2 / 16.8 GWh instead of 19.9; downtime stays within 0.7 GWh.",
           icon="⚠️")

# --- KPIs ------------------------------------------------------------------------------------
st.header("Definitions")
st.markdown(
    "- **Performance Ratio (IEC 61724-1 approach)**, ratio of sums at the inverter-step level. Variant **A** raw, "
    f"**B** temperature-corrected to 25 °C (for STC and contract comparisons), **C** to {info['t_ref_c']:.1f} °C, the "
    "POA-weighted mean cell temperature of the period (for trends and the waterfall). B and C remove the seasonal "
    "swing of A. PR with downtime is taken at the plant meter; the inverter-based PR is understated by about 7 pp "
    "in 2023 because of lost channels.\n"
    "- **Availability:** an inverter is down at a step if its AC power is below 1 % of its rating at POA above "
    "50 W/m². The raw time-based number (lower bound) counts every zero as a stop. The **energy-based** number nets "
    "out the energy the meter confirms was produced (central); the upper bound drops the apparent stops of "
    "channel-loss days. Quote 2023 as **0.79**.\n"
    "- **Channel-loss day:** a local day with sum(inverters) / (meter × 1.022) below 0.97. The factor 1.022 is the "
    "ratio on days when everything reports; one percentage point of it moves the 2023 credit by about 0.35 GWh. "
    "The data cannot say *which* inverter lies, so the correction is plant-level.")

# --- waterfall -------------------------------------------------------------------------------
st.header("Loss waterfall")
st.markdown(
    "A sequential cascade in physical order: **nominal** (P0 × G/1000) → temperature → design DC losses → inverter "
    "efficiency → clipping (= expected) → downtime → unknown AC → channel credit → running shortfall → meter "
    "reconciliation → after inverters (= **metered**, the revenue basis). Each step is the difference of two "
    "neighbouring cumulative quantities, so the steps add up by construction, and the export stops if any month "
    "does not close. Swapping the order of temperature and DC losses moves 0.4-0.5 % of nominal energy. Clipping is "
    "at most 0.08 % (upper bound) and no curtailment plateau was found.")

# --- price -----------------------------------------------------------------------------------
st.header("Price")
st.markdown(f"**{price['base']:.2f} USD/MWh**, flat, band ±30 % ({price['low']:.2f} … {price['high']:.2f}): Georgia Power "
            "avoided energy cost for 2026, *Solar Ground-Mounted Fixed Tilt*, from the 2025 projections filed with the "
            "Georgia PSC (filing #223157, dockets 4822 / 16573). The plant is utility-scale and front-of-meter, in "
            "Georgia, which has no ISO / RTO, so there are no nodal prices and no EIA wholesale hub.")
common.price_caveat()

# --- assumptions -----------------------------------------------------------------------------
st.header("What is my assumption")
st.markdown(
    "Everything below is a choice, not a measurement. Each is varied or shown as a range where it matters.\n\n"
    "| Assumption | Value | Where it matters |\n|---|---|---|\n"
    "| Design DC losses | 6.34 % | split of design vs running shortfall |\n"
    "| Inverter η_nom | 0.98 | same |\n"
    "| Inverter-to-meter factor | 1.022 | channel credit in 2023 |\n"
    "| Channel-loss tolerance | 3 % | which days are channel-loss days |\n"
    "| Day filters for RdTools | ≥ 90 % usable steps, ≥ 50 % running, ≥ 2 kWh/m² | degradation before 2023 (−1.3 … −0.5 %/yr) |\n"
    "| Rain threshold as cleaning | 2 mm | soiling (1.1-1.6 % for 1-5 mm) |\n"
    "| AOI model | ASHRAE b0 = 0.05 | AOI bound (1.8-4.1 % for 0.03-0.07) |\n"
    "| Own-norm window | 121 days | faults up to about 60 days |\n"
    "| False-alarm budget | 1 % of clean inverter-days | detection rates |\n"
    "| Meter-verdict thresholds | 0.97 / 0.95 | wording of the episode table |\n"
    "| Price | 30.19 USD/MWh ± 30 % | every USD figure |\n")

# --- limits ----------------------------------------------------------------------------------
st.header("What these numbers do not say")
st.warning(
    "- Every USD figure rests on a **projected, utility-level price**, not on this plant's PPA price.\n"
    "- The expected-yield model is **not calibrated**; the running shortfall mixes soiling, degradation, AOI, low-light "
    "behaviour, sensor bias and the error of the assumed design losses.\n"
    "- The anomaly results come from **SYNTHETIC** faults (abrupt, constant, AC side, 62 test events in 14 cells).\n"
    "- The causes of group failures (shared transformer skid or feeder) are hypotheses; the files do not contain them.\n"
    "- Two things are **unexplained**: the 2023 step of about 3 pp in measured / expected, and the PR dip in "
    "November-December.\n"
    "- The Open-Meteo data are **reanalysis (ERA5, IFS), not satellite data**.\n"
    "- One plant. The second system (1200, Linthicum Heights, MD) for a scalability check is not part of this version.",
    icon="⚠️")
