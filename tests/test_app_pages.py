"""Smoke tests of the Streamlit pages: every page runs without an exception, tables are centred and
horizontal legends sit above the plot. Uses Streamlit's headless AppTest and the committed app/data,
so it needs neither a browser nor the raw data (skipped where Streamlit is not installed)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent  # AppTest resolves relative paths against this file
PAGES = [str(ROOT / "app" / "Home.py"), *sorted(str(p) for p in (ROOT / "app" / "pages").glob("*.py"))]


@pytest.fixture(scope="module", params=PAGES)
def page(request):
    return AppTest.from_file(request.param, default_timeout=90).run()


def test_page_runs_without_an_exception(page):
    assert [e.value for e in page.exception] == []


def test_every_table_column_is_centred(page):
    for table in page.dataframe:
        columns = json.loads(table.proto.columns)
        shown = {name: cfg for name, cfg in columns.items() if not cfg.get("hidden")}
        assert shown, "table without visible columns"
        assert all(cfg.get("alignment") == "center" for cfg in shown.values()), shown


def test_horizontal_legends_sit_above_the_plot_and_grow_upwards(page):
    # yanchor "top" at y > 1 would hang the legend into the plot and cover the curves
    for chart in page.get("plotly_chart"):
        legend = json.loads(chart.proto.spec)["layout"].get("legend", {})
        if legend.get("orientation") == "h":
            assert legend.get("yanchor") == "bottom" and legend.get("y", 0) >= 1.0, legend
