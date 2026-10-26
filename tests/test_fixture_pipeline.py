"""Tests on the 14-day REAL fixture (tests/fixtures/9069_clean): pipeline vs golden values, and an
injected SYNTHETIC string fault that the detectors must find. No network, no external data."""

import json
from pathlib import Path

import pandas as pd
import pytest

from pv_loss_auditor import anomalies as an
from pv_loss_auditor.model import cell_temperature, load_inputs
from pv_loss_auditor.synthetic_faults import inject_faults
from scripts.capture_golden import pipeline_section
from scripts.compare_golden import compare

GOLDEN = Path("tests/fixtures/golden_values.json")
# Price keys need the Georgia Power file, which is not in the repository (and not in CI).
PRICE_KEYS = ("price_usd_per_mwh", "audit_loss_kwh", "audit_loss_usd")


def test_full_pipeline_on_the_fixture_reproduces_the_golden_values(fixture_config):
    # Same code path as the dashboard export (model -> KPIs -> waterfall -> down-days), compared
    # with the numbers captured BEFORE the refactor. Only rounding errors are allowed (1e-9).
    now, _ = pipeline_section(fixture_config)
    golden = json.loads(GOLDEN.read_text(encoding="utf8"))["fixture"]
    for key in PRICE_KEYS:
        golden.pop(key, None)
        now.pop(key, None)
    differences, n_compared = compare(golden, now, rel_tol=1e-9)
    assert n_compared > 100
    assert differences == [], differences[:5]


def test_fixture_waterfall_closes_and_the_stop_of_december_12_to_18_is_found(fixture_config):
    now, _ = pipeline_section(fixture_config)
    assert now["waterfall_max_closure_kwh"] < 1e-6  # nominal = sum of steps + metered
    # 26 inverters were down 12-18 December: hundreds of inverter-days in a 14-day window
    assert now["down_inverter_days"] > 150
    assert now["kpi_yearly"]["columns"]["availability"][0] < 0.8


# --- an injected SYNTHETIC fault on real data -------------------------------------------------

FAULT = pd.DataFrame(
    [
        {
            "inverter": "p_ac_inv05_kw",
            "start": pd.Timestamp("2020-12-09"),
            "end": pd.Timestamp("2020-12-11"),
            "strings": 40,
            "loss_fraction": 40 / 170,
            "is_synthetic": True,
        }
    ]
)  # SYNTHETIC: 40 of 170 strings = -23.5 %


def daily_rel_median(config, ac_override=None):
    inputs, ac = load_inputs(config)
    t_cell = cell_temperature(inputs, config)
    ac = inject_faults(ac, FAULT) if ac_override else ac
    rel = an.relative_residual(inputs, ac, t_cell, config)
    return an.daily_features(rel, config)["rel_median"]


def test_injected_string_fault_lowers_the_residual_by_the_lost_share(fixture_config):
    # A fault of 40/170 strings multiplies AC by 0.765. Against its peers the inverter must drop by
    # that factor, and only inside the window; the peers and the days outside stay at 1.
    real = daily_rel_median(fixture_config).unstack()["p_ac_inv05_kw"]
    faulty = daily_rel_median(fixture_config, ac_override=True).unstack()["p_ac_inv05_kw"]
    window = (faulty.index >= "2020-12-09") & (faulty.index <= "2020-12-11")
    ratio = (faulty / real)[window]
    assert ratio.to_numpy() == pytest.approx(1 - 40 / 170, abs=0.01)
    assert (faulty / real)[~window].dropna().to_numpy() == pytest.approx(1.0, abs=1e-9)


def test_detectors_find_the_injected_fault_on_the_fixture(fixture_config):
    # Daily deviation of each inverter from the fleet; the fault days must be the 3 most unusual
    # rows for the Isolation Forest and exceed the 5-sigma rule, with no false alarm among clean rows.
    real = daily_rel_median(fixture_config).dropna()
    faulty = daily_rel_median(fixture_config, ac_override=True).dropna()
    x_real = pd.DataFrame({"dev_rel_median": real - 1.0})
    x = pd.DataFrame({"dev_rel_median": faulty - 1.0})
    sigma = an.robust_sigma(x_real["dev_rel_median"])  # sigma of the clean data (as in the project)
    score = an.rule_score(x, sigma)
    is_fault = (
        (x.index.get_level_values("inverter") == "p_ac_inv05_kw")
        & (x.index.get_level_values("date") >= "2020-12-09")
        & (x.index.get_level_values("date") <= "2020-12-11")
    )
    assert is_fault.sum() == 3
    assert (score[is_fault] > 5).all()
    assert (score[~is_fault] > 5).sum() == 0
    forest = an.forest_scores(x, fixture_config)
    assert set(forest.nlargest(3).index) == set(x.index[is_fault])
