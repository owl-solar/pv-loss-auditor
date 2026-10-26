"""Electricity price for valuing lost energy (Phase 3).

9069 is a utility-scale plant behind a step-up transformer (inverters / meter = 1.02) next to a
high-voltage substation, in Georgia, which has no ISO / RTO and so no nodal prices. The price is
the Georgia Power avoided energy cost from the 2025 projections filed with the Georgia PSC
(filing #223157, dockets 4822 and 16573). It is a utility avoided cost projection for
2026-2045, not a realised price and not the plant's PPA price; see `price:` in the config.
"""

import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _read_sheet(xlsx, sheet_number):
    """All cells of one sheet as {'B8': value}. An xlsx file is a zip of XML files; reading it
    directly avoids a new dependency (openpyxl) for one small table."""
    strings = []
    for si in ET.fromstring(xlsx.read("xl/sharedStrings.xml")).findall("m:si", NS):
        strings.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
    root = ET.fromstring(xlsx.read(f"xl/worksheets/sheet{sheet_number}.xml"))
    cells = {}
    for c in root.iter("{%s}c" % NS["m"]):
        v = c.find("m:v", NS)
        if v is not None:
            cells[c.get("r")] = strings[int(v.text)] if c.get("t") == "s" else float(v.text)
    return cells


def load_avoided_energy_cost(config):
    """Avoided energy cost in USD/MWh by projection year and technology (intermittent sheet)."""
    path = Path(config["price"]["file"])
    with zipfile.ZipFile(path) as outer:
        name = [n for n in outer.namelist() if n.endswith(".xlsx")][0]
        with zipfile.ZipFile(outer.open(name)) as xlsx:
            cells = _read_sheet(xlsx, config["price"]["sheet_number"])
    # Row 6 holds the technology names (C..F), rows from 8 hold one projection year each.
    technologies = {col: cells[f"{col}6"] for col in "CDEF"}
    rows = {}
    for r in range(8, 60):
        year = cells.get(f"B{r}")
        if year is None:
            break
        rows[int(year)] = {technologies[col]: cells[f"{col}{r}"] for col in "CDEF"}
    return pd.DataFrame(rows).T.rename_axis("year")


def price_usd_per_mwh(config):
    """(base, low, high) flat price in USD/MWh for the audit: one projection year and one
    technology, with a symmetric sensitivity band."""
    p = config["price"]
    table = load_avoided_energy_cost(config)
    base = float(table.loc[p["year"], p["technology"]])
    band = p["sensitivity_fraction"]
    return base, base * (1 - band), base * (1 + band)
