"""Export small, committed tables for the Streamlit dashboard (app/data/).

The dashboard does not read the 1.2 GB raw data or the 49 MB clean parquet files: this script runs
the same functions as the notebooks once and saves their results. Everything the app shows is
read from these files, so a public deployment needs no download.

Prices are NOT applied here: tables are in kWh and the app multiplies by the price chosen in its
sidebar, so the price band can be changed without a new export. Run from the project root:

    python -m src.export_dashboard --config configs/system_9069.yaml
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src import degradation as dg
from src import kpi, losses, openmeteo as om, outages, prices
from src.config import load_config
from src.model import cell_temperature, load_inputs, weighted_gamma

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

    Same calls as notebooks/04_degradation_soiling.ipynb (computations live in src/degradation.py);
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


def meta(config, t_ref):
    """Scalars the app shows next to the numbers: price assumption, period, T_ref."""
    base, low, high = prices.price_usd_per_mwh(config)
    return {
        "plant": f"{config['system']['name']} ({config['system']['location']}), PVDAQ {config['system']['id']}",
        "dc_kw": config["capacity"]["dc_kw"], "ac_kw": config["capacity"]["ac_kw"],
        "price_usd_per_mwh": {"base": base, "low": low, "high": high},
        "price_source": config["price"]["source"].strip(),
        "t_ref_c": float(t_ref),
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

    out = Path(args.out)
    sizes = write_tables(tables, out)
    (out / "meta.json").write_text(json.dumps(meta(config, t_ref), indent=2), encoding="utf8")
    print("written to", out, "(kB):", sizes)


if __name__ == "__main__":
    main()
