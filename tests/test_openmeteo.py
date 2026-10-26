"""Tests of src/pv_loss_auditor/openmeteo.py without internet (requests.get is replaced) on SYNTHETIC weather."""

import pandas as pd
import pytest

from pv_loss_auditor import openmeteo as om
from pv_loss_auditor.config import load_config

CONFIG = load_config("configs/system_9069.yaml")


class FakeResponse:
    def raise_for_status(self):
        pass

    def json(self):
        times = pd.date_range("2020-05-01", periods=3, freq="1h").strftime("%Y-%m-%dT%H:%M")
        return {
            "hourly": {
                "time": list(times),
                "shortwave_radiation": [0.0, 100.0, 200.0],
                "precipitation": [0.0, 0.0, 1.2],
            }
        }


def test_fetch_always_sends_an_explicit_model_and_caches(monkeypatch, tmp_path):
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        return FakeResponse()

    monkeypatch.setattr(om.requests, "get", fake_get)
    for source, model in [("era5", "era5"), ("ifs", "ecmwf_ifs")]:
        data = om.fetch(source, CONFIG, out_dir=tmp_path)
        assert calls[-1]["models"] == model  # never the silent best_match of the archive
        assert calls[-1]["timezone"] == "UTC"
        assert data.index.tz is not None and str(data.index.tz) == "UTC"
        assert len(data) == 3
    n_calls = len(calls)
    again = om.fetch("era5", CONFIG, out_dir=tmp_path)  # second call: from the parquet cache
    assert len(calls) == n_calls
    assert again["shortwave_radiation"].tolist() == [0.0, 100.0, 200.0]


def test_fetch_uses_the_plant_coordinates(monkeypatch, tmp_path):
    seen = {}

    def fake_get(url, params, timeout):
        seen.update(params)
        return FakeResponse()

    monkeypatch.setattr(om.requests, "get", fake_get)
    om.fetch("era5", CONFIG, out_dir=tmp_path)
    assert seen["latitude"] == pytest.approx(33.6762) and seen["longitude"] == pytest.approx(-83.676)


def weather(rows):
    """Hourly frame; rows = list of (UTC hour-end label, GHI, DNI, DHI)."""
    idx = pd.DatetimeIndex([r[0] for r in rows], tz="UTC")
    return pd.DataFrame(
        {
            "shortwave_radiation": [r[1] for r in rows],
            "direct_normal_irradiance": [r[2] for r in rows],
            "diffuse_radiation": [r[3] for r in rows],
        },
        index=idx,
    )


def test_poa_is_zero_at_night_and_plausible_in_diffuse_and_clear_hours():
    w = weather(
        [
            ("2021-06-21 05:00", 0, 0, 0),  # 01:00 EDT, night
            ("2021-06-21 17:00", 300, 0, 300),  # midday, overcast: diffuse only
            ("2021-06-21 17:00", 300, 0, 300),
        ]
    ).iloc[[0, 1]]
    poa = om.poa_from_weather(w, CONFIG)
    assert poa.iloc[0] == 0.0
    # Diffuse sky on a 20 deg tilt: about (1 + cos 20 deg) / 2 = 0.97 of the horizontal value.
    assert 270 < poa.iloc[1] < 320
    clear = om.poa_from_weather(weather([("2021-06-21 17:00", 900, 800, 100)]), CONFIG)
    assert 800 < clear.iloc[0] < 1100  # summer noon, 20 deg tilt facing south


def test_aoi_loss_is_zero_at_night_small_at_noon_and_larger_at_low_sun():
    w = weather(
        [
            ("2021-06-21 05:00", 0, 0, 0),  # night
            ("2021-06-21 17:00", 900, 800, 100),  # noon: beam near the normal of the modules
            ("2021-06-21 23:00", 300, 600, 60),  # ~19:00 EDT: low sun in the west
        ]
    )
    loss = om.aoi_loss(w, CONFIG)
    assert loss.iloc[0] == 0.0
    assert 0 <= loss.iloc[1] < 0.06
    assert loss.iloc[2] > loss.iloc[1]
    assert (loss >= 0).all() and (loss <= 1).all()


def test_aoi_loss_grows_with_the_ashrae_b0():
    w = weather([("2021-06-21 23:00", 300, 600, 60)])
    low, high = om.aoi_loss(w, CONFIG, b0=0.03), om.aoi_loss(w, CONFIG, b0=0.07)
    assert high.iloc[0] > low.iloc[0] > 0
