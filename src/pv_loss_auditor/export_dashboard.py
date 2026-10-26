"""Export small, committed tables for the Streamlit dashboard (app/data/).

The dashboard does not read the 1.2 GB raw data or the 49 MB clean parquet files: this script runs
the same functions as the notebooks once and saves their results. Everything the app shows is
read from these files, so a public deployment needs no download.

Prices are NOT applied here: tables are in kWh and the app multiplies by the price chosen in its
sidebar, so the price band can be changed without a new export. Run from the project root:

    python -m pv_loss_auditor.export_dashboard --config configs/system_9069.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from pv_loss_auditor import anomalies as an
from pv_loss_auditor import degradation as dg
from pv_loss_auditor import kpi, losses, outages, prices
from pv_loss_auditor import openmeteo as om
from pv_loss_auditor.config import load_config
from pv_loss_auditor.logs import setup_logging
from pv_loss_auditor.model import cell_temperature, load_inputs, weighted_gamma
from pv_loss_auditor.synthetic_faults import inject_faults, plan_faults

log = logging.getLogger(__name__)

OUT_DIR = Path("app/data")


def _period_labels(table):
    """Period index -> text ("2021", "2021-06"), because parquet keeps plain columns best."""
    out = table.copy()
    out.index = out.index.astype(str)
    out.index.name = "period"
    return out


def waterfall_tables(inputs, ac, t_cell, config, gamma):
    """Waterfall in kWh per month (the app sums months into years) and the DC-loss sensitivity.

    The closure of every month is checked here, so a broken export cannot be published.
    """
    monthly = losses.loss_waterfall(inputs, ac, t_cell, config, gamma, freq="M")
    losses.check_closure(monthly)
    sens = losses.dc_loss_sensitivity(inputs, ac, t_cell, config, gamma)
    return {"waterfall_monthly": _period_labels(monthly), "dc_loss_sensitivity": sens}


def kpi_tables(inputs, ac, t_cell, config, gamma):
    """KPIs and the revised availability, per month and per year, plus T_ref of variant C."""
    base = inputs["daylight"] & inputs["poa_wm2"].notna() & t_cell.notna()
    t_ref = kpi.weighted_t_ref(t_cell, inputs["poa_wm2"], base)
    tables = {}
    for freq, name in (("M", "monthly"), ("Y", "yearly")):
        k = kpi.kpi_table(inputs, ac, t_cell, config, gamma, t_ref, freq=freq)
        a = losses.availability_revised(inputs, ac, t_cell, config, gamma, freq=freq)
        table = _period_labels(pd.concat([k, a], axis=1))
        if freq == "M":
            # The first UTC hours of 2020-05-01 are the evening of 2020-04-30 local time: drop
            # that stub month. Not for years: the label "2020" sorts before "2020-05".
            table = table.loc["2020-05":]
        tables[f"kpi_{name}"] = table
    return tables, t_ref


def outage_tables(inputs, ac, config):
    """Down-days per inverter (local date x inverter), inverter pairs and the episode table."""
    is_down = outages.down_days(ac, inputs["poa_wm2"], config)
    ratio = outages.inverter_meter_ratio(inputs, ac, config)
    episodes = outages.episode_table(is_down, ratio, config["anomalies"]["episodes"])
    pairs = outages.pair_overlap(is_down)
    down = is_down.copy()
    down.index.name = "date"
    return {
        "down_days": down,
        "inverter_meter_ratio": ratio.rename("ratio").rename_axis("date").to_frame(),
        "down_pairs": pairs,
        "outage_episodes": episodes,
    }


def degradation_tables(inputs, ac, t_cell, config):
    """Phase 4: daily series for RdTools, YoY rates, soiling, external weather, shortfall split.

    Same calls as notebooks/04_degradation_soiling.ipynb (computations live in src/pv_loss_auditor/degradation.py);
    the Open-Meteo files must be cached in data/raw/openmeteo/ (no download if they are).
    """
    daily = dg.daily_series(inputs, ac, t_cell, config)
    era5, ifs = om.fetch("era5", config), om.fetch("ifs", config)
    rain = dg.daily_precipitation(era5)
    out = {"daily_series": daily.join(rain.rename("rain_mm")).rename_axis("date")}
    out["degradation_yoy"] = dg.yoy_table(daily)
    # The rate before 2023 depends on the day filters (my assumptions): keep the whole grid.
    out["degradation_filters"] = dg.filter_sensitivity(inputs, ac, t_cell, config, [1, 2, 3, 4], [0.25, 0.5, 0.8])
    soil = dg.soiling_table(daily, rain)
    out["soiling"] = soil

    ext = {"era5": om.poa_from_weather(era5, config), "ifs": om.poa_from_weather(ifs, config)}
    stats, ratio = dg.external_poa_comparison(dg.hourly_plant_poa(inputs["poa_wm2"]), ext)
    out["external_poa_stats"] = stats
    ratio = ratio["2020-05-01":]                         # the first month starts on 30 April local
    ratio.index = ratio.index.strftime("%Y-%m")
    out["external_poa_ratio"] = ratio.rename_axis("month")
    out["pr_by_poa_source"] = dg.pr_by_poa_source(inputs, ac, config, ext).rename_axis("year")

    aoi = {b0: dg.weighted_loss_pct(ext["era5"], om.aoi_loss(era5, config, b0=b0)) for b0 in (0.03, 0.05, 0.07)}
    out["aoi_by_b0"] = pd.DataFrame({"b0": list(aoi), "aoi_loss_pct": list(aoi.values())})
    is_main = (soil.cleaning_info == "rain >= 2 mm + jumps") & (soil.period == "full")
    out["shortfall_split"] = dg.shortfall_split(daily, float(soil.loc[is_main, "loss_pct"].iloc[0]),
                                                aoi[0.05]).rename_axis("year")
    return out


def anomaly_tables(inputs, ac, t_cell, config):
    """Phase 5: detector comparison on SYNTHETIC string faults and the real events.

    Same steps as notebooks/05_anomaly_detection.ipynb. Everything that depends on the injected
    faults carries synthetic=True; the real events are flags of the same detectors on real days.
    """
    A = config["anomalies"]
    feat = an.daily_features(an.relative_residual(inputs, ac, t_cell, config), config)
    x_real = an.own_norm_features(feat, config)
    sigma = an.robust_sigma(x_real["dev_rel_median"])         # sigma of the REAL data, no faults

    events = plan_faults(feat["rel_median"].unstack().notna(), config)
    ac_synth = inject_faults(ac, events)                      # a copy: the real ac is untouched
    feat_s = an.daily_features(an.relative_residual(inputs, ac_synth, t_cell, config), config)
    x = an.own_norm_features(feat_s, config)
    labels = an.day_labels(x.index, events, config)
    results, split = an.run_detectors(x, labels, events, config, sigma)

    out = {}
    out["anomaly_summary"] = an.summary_table(results).assign(synthetic=True)
    out["anomaly_by_cell"] = an.detection_by_cell(results).assign(synthetic=True)
    tuning = [t.assign(criterion=c, detector=name) for c, by_det in results.items()
              for name, (t, *_rest) in by_det.items()]
    out["anomaly_tuning"] = pd.concat(tuning, ignore_index=True).assign(synthetic=True)

    where = pd.Series("dropped (straddles the split)", index=events.index)
    where[split["ev_tune"].index] = "tuning"
    where[split["ev_test"].index] = "test"
    out["anomaly_events"] = events.assign(duration_days=(events.end - events.start).dt.days + 1,
                                          set=where, synthetic=True)

    # Real events found in the daily residuals (config: anomalies.synthetic.exclude_inverter).
    flags = {name: v[4] for name, v in results["far"].items()}
    real = [e for e in A["synthetic"]["exclude_inverter"] if e["inverter"] in ("p_ac_inv30_kw", "p_ac_inv13_kw")]
    windows = [(e["inverter"], e["start"], e["end"]) for e in real]
    out["anomaly_real_events"] = an.flags_in_windows(flags, windows)
    excluded = (split["lab"] == "excluded").to_numpy()
    out["anomaly_excluded_flag_share"] = pd.DataFrame(
        {"detector": list(flags), "flagged_share": [float(f[excluded].mean()) for f in flags.values()],
         "rows": int(excluded.sum())})

    # Inverter 30, the real event of 2020-09-22 ... 11-04: deviation from own norm and the flags.
    idx = x.index.get_level_values
    m = (idx("inverter") == "p_ac_inv30_kw") & (idx("date") >= "2020-07-01") & (idx("date") <= "2021-01-31")
    inv30 = x.loc[m, ["dev_rel_median"]].droplevel("inverter")
    for name in ("rule (baseline)", "forest, 4 features"):
        inv30["flag " + name] = flags[name][m].droplevel("inverter")
    out["anomaly_inv30"] = inv30

    # Noise of the real data in the units a reader can compare with a fault.
    n_str = A["synthetic"]["strings_per_inverter"]
    out["anomaly_noise"] = pd.DataFrame([{
        "robust_sigma": sigma, "rows": len(x_real), "std": float(x_real["dev_rel_median"].std()),
        "strings_per_inverter": n_str, "one_string_pct": 100 / n_str,
        "three_sigma_strings": 3 * sigma * n_str,
        "events": len(events), "events_test": len(split["ev_test"]),
        "clean_days_test": int(results["far"]["rule (baseline)"][2]["clean_days"]),
        "far_max": A["forest"]["far_max"], "tune_until": A["forest"]["tune_until"]}])
    return out


def meta(config, t_ref):
    """Scalars the app shows next to the numbers: price assumption, period, T_ref."""
    base, low, high = prices.price_usd_per_mwh(config)
    return {
        "plant": f"{config['system']['name']} ({config['system']['location']}), PVDAQ {config['system']['id']}",
        "dc_kw": config["capacity"]["dc_kw"], "ac_kw": config["capacity"]["ac_kw"],
        "price_usd_per_mwh": {"base": base, "low": low, "high": high},
        "price_source": config["price"]["source"].strip(),
        "t_ref_c": float(t_ref),
        "gamma_pct_per_c": float(weighted_gamma(config) * 100),   # DC-weighted nameplate, shown on Methodology
        "design_dc_losses_pct": "6.34 (assumption, no PVsyst report)",
    }


def write_tables(tables, out_dir):
    """Save every table as parquet (index kept) and return the sizes in kB."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = {}
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path)
        sizes[name] = round(path.stat().st_size / 1024, 1)
    return sizes


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()

    config = load_config(args.config)
    inputs, ac = load_inputs(config)
    t_cell = cell_temperature(inputs, config)
    gamma = weighted_gamma(config)

    tables = waterfall_tables(inputs, ac, t_cell, config, gamma)
    kpis, t_ref = kpi_tables(inputs, ac, t_cell, config, gamma)
    tables.update(kpis)
    tables.update(outage_tables(inputs, ac, config))
    tables.update(degradation_tables(inputs, ac, t_cell, config))
    tables.update(anomaly_tables(inputs, ac, t_cell, config))

    out = Path(args.out)
    sizes = write_tables(tables, out)
    (out / "meta.json").write_text(json.dumps(meta(config, t_ref), indent=2), encoding="utf8")
    log.info("written to %s (kB): %s", out, sizes)


if __name__ == "__main__":
    main()
