"""Clock of each PVDAQ file: detect it from the physics of the data and convert labels to UTC.

PVDAQ 9069 timestamps are naive. Phase 1 QA found that the files do not share one clock:
irradiance and environment follow EU daylight-saving dates on a UTC-5 base, electrical_ac and
meter follow US dates (America/New_York). The detection never uses the timestamp axis itself
(its night-time jumps do not match the data); it compares each day's profile with the shape of
the sun, which cannot be wrong about the time.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from pv_loss_auditor.coverage import SENTINEL_MAX

STEP = pd.Timedelta("5min")
STEPS_PER_DAY = 288
STEPS_PER_HOUR = 12

# Clock rules (config: clock.rules). "eu_dst_utc-5" = UTC-5, plus 1 h between the last Sunday
# of March and the last Sunday of October, switching at 01:00 UTC (the EU rule).
EU_DST_UTC5 = "eu_dst_utc-5"


def load_period(config, file, columns=None):
    """All monthly parquet files of one source file -> one DataFrame on the naive label index.

    Logger placeholders (<= SENTINEL_MAX, e.g. -297) become NaN; nothing else is changed.
    """
    folder = Path(config["period"]["out_dir"]) / file
    parts = [pd.read_parquet(p, columns=columns) for p in sorted(folder.glob(f"{file}_*.parquet"))]
    df = pd.concat(parts)
    df = df[~df.index.duplicated()]  # month files do not overlap; this is only a safety net
    return df.mask(df <= SENTINEL_MAX)


def to_utc(index, rule):
    """Naive label index -> tz-aware UTC index, using the clock rule of the file.

    The repeated autumn hour (US rule) and the skipped spring hour become NaT: both are at night.
    """
    if rule == EU_DST_UTC5:
        # Europe/London is UTC+1 exactly while EU summer time is in force, so its offset at a
        # UTC instant tells whether the logger clock was 1 h ahead at that instant.
        def summer(utc):
            return utc.tz_convert("Europe/London").map(lambda t: t.utcoffset()) != pd.Timedelta(0)

        as_winter = (index + pd.Timedelta(hours=5)).tz_localize("UTC")  # label read as UTC-5
        as_summer = (index + pd.Timedelta(hours=4)).tz_localize("UTC")  # label read as UTC-4
        s_w, s_s = summer(as_winter), summer(as_summer)
        # Both readings agree except in the hour next to a switch: there the label is
        # ambiguous (autumn) or should not exist (spring) -> NaT, as for the US rule.
        utc = pd.DatetimeIndex(np.where(s_w & s_s, as_summer, as_winter))
        return utc.where(s_w == s_s, pd.NaT)
    return index.tz_localize(rule, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")


def expected_offset_h(days, rule):
    """Offset label - UTC (hours) at local noon of each day under a clock rule."""
    noon = pd.DatetimeIndex(days).normalize() + pd.Timedelta(hours=12)
    return (noon - to_utc(noon, rule).tz_localize(None)) / pd.Timedelta(hours=1)


def daily_clock_offset(series, latitude, longitude, search_h=(-6.0, -3.0), min_r=0.0):
    """Best clock offset (label - UTC, hours) of every day, from the shape of the daily profile.

    For each candidate offset (5-min steps) the day's values are correlated with clear-sky GHI
    (Haurwitz: needs only the sun position) at UTC = label - offset; the offset with the highest
    correlation r wins. Correlation ignores the amplitude, so the method works for irradiance,
    AC power or any channel that follows the sun. Clouds lower r: keep days with r > 0.97 for
    conclusions.

    Returns a DataFrame indexed by day: offset_h, r. Days with > 10 % missing values are skipped.
    """
    start = series.index.min().normalize()
    grid = pd.date_range(start, series.index.max().normalize() + pd.Timedelta(days=1),
                         freq=STEP, inclusive="left")
    values = series.reindex(grid).to_numpy(dtype=float)

    # Clear-sky once, on a UTC grid wide enough for all candidate offsets. Label step i with
    # offset k steps corresponds to UTC step i - k, i.e. position pad + i - k in this array.
    pad = int(abs(min(search_h)) * STEPS_PER_HOUR) + STEPS_PER_HOUR
    utc = pd.date_range(grid[0] - pad * STEP, periods=len(grid) + 2 * pad, freq=STEP, tz="UTC")
    clear = pvlib.location.Location(latitude, longitude).get_clearsky(utc, model="haurwitz")
    clear = clear["ghi"].to_numpy()
    offsets = np.arange(int(search_h[0] * STEPS_PER_HOUR), int(search_h[1] * STEPS_PER_HOUR) + 1)

    rows = []
    for d0 in range(0, len(grid), STEPS_PER_DAY):
        day = values[d0:d0 + STEPS_PER_DAY]
        if np.isnan(day).mean() > 0.1:
            continue
        best_k, best_r = np.nan, -np.inf
        for k in offsets:
            cs = clear[pad + d0 - k: pad + d0 - k + STEPS_PER_DAY]
            ok = ~np.isnan(day) & (cs > 20)  # daytime by the model; night adds no information
            if ok.sum() < 50 or np.std(day[ok]) == 0:
                continue
            r = np.corrcoef(day[ok], cs[ok])[0, 1]
            if r > best_r:
                best_k, best_r = k, r
        rows.append((grid[d0], best_k / STEPS_PER_HOUR, best_r))
    out = pd.DataFrame(rows, columns=["day", "offset_h", "r"]).set_index("day")
    return out[out["r"] > min_r]


def rule_mismatches(offsets, rules, min_r=0.97, tolerance_h=0.5):
    """How many clear days contradict each clock rule (|detected - expected| > tolerance)."""
    clear = offsets[offsets["r"] > min_r]
    rows = []
    for rule in rules:
        miss = (clear["offset_h"] - expected_offset_h(clear.index, rule)).abs() > tolerance_h
        rows.append({"rule": rule, "clear_days": len(clear), "mismatches": int(miss.sum())})
    return pd.DataFrame(rows).set_index("rule")


def lag_h(reference, other, start, end, max_lag_h=2.0, min_ref=20):
    """Lag (hours) that best aligns `other` with `reference` in [start, end], and its r.

    Positive lag = `other` is labelled later than `reference`. Both series must share one
    index type (both naive labels, or both UTC); only steps with reference > min_ref count.
    """
    grid = pd.date_range(start, end, freq=STEP, tz=reference.index.tz)
    ref = reference.reindex(grid)
    oth = other.reindex(grid)
    best = (np.nan, np.nan)
    for k in range(-int(max_lag_h * STEPS_PER_HOUR), int(max_lag_h * STEPS_PER_HOUR) + 1):
        shifted = oth.shift(-k)
        ok = ref.notna() & shifted.notna() & (ref > min_ref)
        # A constant series (e.g. an inverter that is off) has no correlation: skip it.
        if ok.sum() < 100 or shifted[ok].std() == 0:
            continue
        r = np.corrcoef(ref[ok], shifted[ok])[0, 1]
        if np.isnan(best[1]) or r > best[1]:
            best = (k / STEPS_PER_HOUR, r)
    return best


def rule_windows(years, rule_a="America/New_York", rule_b=EU_DST_UTC5):
    """Date ranges where two clock rules disagree by one hour (only there can they be told apart).

    Found by comparing the expected offsets day by day. Returns a list of (first day, last day).
    """
    days = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    differ = expected_offset_h(days, rule_a) != expected_offset_h(days, rule_b)
    runs = (pd.Series(differ, index=days) != pd.Series(differ, index=days).shift()).cumsum()
    return [(g.index[0], g.index[-1]) for _, g in pd.Series(differ, index=days).groupby(runs)
            if g.iloc[0]]
