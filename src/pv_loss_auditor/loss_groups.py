"""Names of the waterfall steps and their grouping into causes (no heavy imports).

Kept apart from src/pv_loss_auditor/losses.py so that the dashboard can use them without importing the model, the
cleaning code and the data loaders. src.losses re-exports these names.
"""

import pandas as pd

STEPS = ["temperature", "dc_losses", "inverter", "clipping", "downtime", "unknown_ac",
         "channel_credit", "running_shortfall", "reconciliation", "after_inverters"]


# Presentation groups: columns of the waterfall table -> cause shown to a reader.
CAUSES = {
    "design (temperature, DC, inverter, clipping)": ["temperature", "dc_losses", "inverter", "clipping"],
    "downtime (net of channel loss)": ["downtime", "unknown_ac", "channel_credit"],
    "running shortfall": ["running_shortfall"],
    "after inverters, meter reconciliation": ["after_inverters", "reconciliation"],
}


def by_cause(table):
    """Waterfall steps summed into the groups of CAUSES (kWh)."""
    return pd.DataFrame({name: table[cols].sum(axis=1) for name, cols in CAUSES.items()})
