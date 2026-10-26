"""Which inverters are down together, and are the zeros real? (Phase 5, diagnostics)

An inverter-day is "down" if the inverter is below 1 % of its rating on more than half of its
known daylight steps. Two inverters that fail together on (almost) the same days share something
upstream (a transformer, a breaker, a feeder) or a data logger. Whether the zeros are real is
checked against the plant meter: stopped inverters lower the meter as well, so
sum(inverters) / (meter x inv_to_meter) stays at ~1; false zeros (lost data channels) lower only
the sum.
"""

import numpy as np
import pandas as pd

from pv_loss_auditor.kpi import inverter_states, step_hours

LOCAL_TZ = "America/New_York"


def _local_day(index):
    return index.tz_convert(LOCAL_TZ).normalize().tz_localize(None)


def down_days(ac, poa, config, min_known_steps=24):
    """Frame (local dates x inverters): 1.0 = down on more than half of the known steps, 0.0 = up.

    Days with fewer than min_known_steps known daylight steps are NaN (neither up nor down).
    """
    known, down = inverter_states(ac, poa, config)
    day = _local_day(ac.index)
    n_known, n_down = known.groupby(day).sum(), down.groupby(day).sum()
    share = n_down / n_known.where(n_known >= min_known_steps)
    return (share > 0.5).astype(float).where(share.notna())


def pair_overlap(is_down, min_union=10):
    """Jaccard index of the down-days of every inverter pair, best pairs first.

    Jaccard = days both are down / days at least one is down (1.0 = always together).
    Only days where both inverters are known count; pairs with fewer than min_union such days
    where at least one is down are skipped.
    """
    cols = list(is_down.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            known = is_down[a].notna() & is_down[b].notna()
            da, db = (is_down[a] == 1) & known, (is_down[b] == 1) & known
            union = int((da | db).sum())
            if union >= min_union:
                both = int((da & db).sum())
                rows.append({"a": a, "b": b, "both": both, "union": union, "jaccard": both / union})
    return pd.DataFrame(rows).sort_values("jaccard", ascending=False).reset_index(drop=True)


def inverter_meter_ratio(inputs, ac, config):
    """Daily sum(inverter AC) / (meter x inv_to_meter) over daylight steps with POA and meter."""
    dt = step_hours(inputs.index)
    lit = inputs["daylight"] & inputs["poa_wm2"].notna() & inputs["p_ac_meter_kw"].notna()
    day = _local_day(inputs.index)
    keep = np.repeat(lit.to_numpy()[:, None], ac.shape[1], axis=1)
    e_inv = (ac.where(keep).sum(axis=1) * dt).groupby(day).sum()
    e_meter = (inputs["p_ac_meter_kw"].where(lit) * dt).groupby(day).sum()
    return e_inv / (e_meter * config["losses"]["inv_to_meter"])


def episode_table(is_down, ratio, episodes):
    """One row per episode: which inverters are down, and what the meter says.

    stopped: inverters down on more than 80 % of the episode's days; partly: 20-80 %.
    ratio_median: median daily inverters/meter ratio (~1 = zeros are real, below = false zeros).
    """
    rows = []
    for e in episodes:
        sl = slice(pd.Timestamp(e["start"]), pd.Timestamp(e["end"]))
        d = is_down.loc[sl]
        share = d.mean()
        r = ratio.loc[sl].dropna()
        rows.append(
            {
                "start": e["start"],
                "end": e["end"],
                "days": len(d),
                "stopped": ",".join(c[8:10] for c in share[share > 0.8].index),
                "n_stopped": int((share > 0.8).sum()),
                "partly": ",".join(c[8:10] for c in share[(share > 0.2) & (share <= 0.8)].index),
                "ratio_median": r.median() if len(r) else np.nan,
                "ratio_p5": r.quantile(0.05) if len(r) else np.nan,
            }
        )
    return pd.DataFrame(rows)
