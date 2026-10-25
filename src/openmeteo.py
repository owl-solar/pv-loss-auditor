"""Open-Meteo historical weather for 9069 from two model sources (archive API, explicit models).

- era5: ECMWF ERA5 reanalysis (about 25 km grid), hourly irradiance, temperature, wind, rain;
- ifs: ECMWF IFS (about 9 km), same variables.
The satellite API has NO coverage at 9069 ("No data is available for this location"), so neither
source is satellite-derived. ALWAYS pass `models`: without it the archive returns `best_match`
(= IFS here) and the label would lie.
One parquet per source in data/raw/openmeteo/ (never downloaded twice, no API key).
Hourly values are means over the PRECEDING hour (Open-Meteo convention), index in UTC, so they
are labelled at the end of the hour; align with the 5-min plant data accordingly.
"""

from pathlib import Path

import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = ["shortwave_radiation", "direct_normal_irradiance", "diffuse_radiation",
          "temperature_2m", "wind_speed_10m", "precipitation"]
MODELS = {"era5": "era5", "ifs": "ecmwf_ifs"}


def fetch(source, config, out_dir="data/raw/openmeteo"):
    """Hourly DataFrame (UTC index) for the analysis period; read from cache if present."""
    path = Path(out_dir) / f"9069_{source}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    sysc = config["system"]
    params = {
        "latitude": sysc["latitude"], "longitude": sysc["longitude"],
        "start_date": "2020-05-01", "end_date": "2023-11-30",
        "hourly": ",".join(HOURLY), "models": MODELS[source], "timezone": "UTC",
    }
    response = requests.get(URL, params=params, timeout=120)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    data = pd.DataFrame(hourly).set_index("time")
    data.index = pd.to_datetime(data.index, utc=True).rename("time_utc")
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path)
    return data


def poa_from_weather(weather, config, albedo=0.2):
    """Plane-of-array irradiance (W/m2) of the fixed array from hourly GHI / DNI / DHI.

    Open-Meteo labels an hourly mean at the END of the hour, so the sun position is taken at the
    middle of that hour (label - 30 min). Perez transposition (pvlib) with the model's own DNI and
    DHI: no decomposition model is needed. Albedo 0.2 is the same default as geometry.py.
    """
    import pvlib

    sysc, mount = config["system"], config["mount"]
    mid_hour = weather.index - pd.Timedelta(minutes=30)
    sun = pvlib.solarposition.get_solarposition(mid_hour, sysc["latitude"], sysc["longitude"],
                                                sysc["elevation_m"])
    dni_extra = pvlib.irradiance.get_extra_radiation(mid_hour)
    airmass = pvlib.atmosphere.get_relative_airmass(sun["apparent_zenith"])
    total = pvlib.irradiance.get_total_irradiance(
        mount["tilt_deg"], mount["azimuth_deg"], sun["apparent_zenith"].to_numpy(),
        sun["azimuth"].to_numpy(), weather["direct_normal_irradiance"].to_numpy(),
        weather["shortwave_radiation"].to_numpy(), weather["diffuse_radiation"].to_numpy(),
        dni_extra=dni_extra.to_numpy(), airmass=airmass.to_numpy(), albedo=albedo, model="perez")
    return pd.Series(total["poa_global"], index=weather.index, name="poa_wm2").fillna(0)
