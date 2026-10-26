"""Tests of the price band and the plateau (curtailment) check."""

from pathlib import Path

import pandas as pd
import pytest

from pv_loss_auditor import prices
from pv_loss_auditor import sensor_checks as sc
from pv_loss_auditor.config import load_config

CONFIG = load_config("configs/system_9069.yaml")


def test_price_band_is_symmetric_around_the_base(monkeypatch):
    table = pd.DataFrame({CONFIG["price"]["technology"]: [30.0]}, index=pd.Index([CONFIG["price"]["year"]]))
    monkeypatch.setattr(prices, "load_avoided_energy_cost", lambda config: table)
    base, low, high = prices.price_usd_per_mwh(CONFIG)
    assert (base, low, high) == pytest.approx((30.0, 21.0, 39.0))  # +-30 %


@pytest.mark.skipif(not Path(CONFIG["price"]["file"]).exists(), reason="price file is not committed")
def test_price_file_gives_the_documented_value():
    base, low, high = prices.price_usd_per_mwh(CONFIG)
    assert base == pytest.approx(30.19, abs=0.005)
    assert (low, high) == pytest.approx((21.13, 39.25), abs=0.01)


def column(values):
    return pd.DataFrame({"p": values}, dtype=float)


def test_plateau_flags_a_held_level_but_not_a_short_run():
    rating = 825.0
    # 10 steps held at 300 kW (9 equal pairs), then a ramp, then a 4-step hold: too short.
    values = [300.0] * 10 + [320.0, 340.0, 360.0] + [400.0] * 4 + [450.0, 500.0]
    flagged = sc.plateau_steps(column(values), rating)["p"]
    assert flagged.iloc[:10].sum() == 9  # first step of the run has no equal predecessor
    assert flagged.iloc[10:].sum() == 0


def test_plateau_ignores_zeros_and_the_clipping_level():
    rating = 825.0
    zeros = [0.0] * 12  # downtime, not curtailment
    clipped = [820.0] * 12  # 99.4 % of the rating = clipping plateau
    assert sc.plateau_steps(column(zeros), rating)["p"].sum() == 0
    assert sc.plateau_steps(column(clipped), rating)["p"].sum() == 0


def test_plateau_tolerates_small_noise():
    rating = 825.0
    noisy = [300.0, 300.2, 299.9, 300.1, 300.0, 299.8, 300.1, 300.0]  # within 0.5 kW
    assert sc.plateau_steps(column(noisy), rating)["p"].sum() >= 6
