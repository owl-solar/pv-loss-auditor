"""Checks that set the model parameters (notebook 02, sections 2-4).

- time offset between AC power and irradiance (lag of the best correlation on clear days);
- temperature coefficient gamma measured on the plant, per inverter;
- drift of the reference cells against the class-A pyranometer 02B (year-over-year).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pvlib

from pv_loss_auditor.data import load_clean
from pv_loss_auditor.geometry import LOCAL_TZ, local_day
from pv_loss_auditor.model import valid


def sun_and_aoi(index: pd.DatetimeIndex, config: Mapping[str, Any]) -> pd.DataFrame:
    """Sun azimuth (deg) and angle of incidence on the fixed 20/180 plane (deg), UTC index."""
    s, mount = config["system"], config["mount"]
    sun = pvlib.solarposition.get_solarposition(
        index, s["latitude"], s["longitude"], altitude=s["elevation_m"]
    )
    aoi = pvlib.irradiance.aoi(
        mount["tilt_deg"], mount["azimuth_deg"], sun["apparent_zenith"], sun["azimuth"]
    )
    return pd.DataFrame({"sun_azimuth_deg": sun["azimuth"], "aoi_deg": aoi})


def clear_days_from_poa(
    poa: pd.Series, poa_clearsky: pd.Series, daylight: pd.Series, min_r: float = 0.998, min_steps: int = 130
) -> pd.DatetimeIndex:
    """Local days whose measured POA follows the clear-sky POA curve (Pearson r >= min_r).

    Needs no GHI (GHI exists only until 2022-03), so it works for the whole period.
    min_steps: daylight 5-min steps with data, about 11 h; drops short winter days and gaps.
    """
    df = pd.DataFrame({"poa": poa, "cs": poa_clearsky})[daylight]
    days = local_day(df.index)
    r = df.groupby(days).apply(lambda x: x["poa"].corr(x["cs"]))
    n = df["poa"].groupby(days).count()
    return set(r[(r >= min_r) & (n >= min_steps)].index)


def lag_curve(
    a: pd.Series, b: pd.Series, days: pd.DatetimeIndex, lags: Sequence[int], mask: pd.Series | None = None
) -> pd.Series:
    """Mean over days of corr(a, b shifted by lag steps), for each lag (in steps).

    A positive lag shifts b later. mask: optional boolean Series, extra condition on the steps.
    """
    df = pd.DataFrame({"a": a, "b": b, "use": True if mask is None else mask})
    df["day"] = local_day(df.index)
    out = {}
    for lag in lags:
        df["b_shifted"] = df["b"].shift(lag)  # shift on the full grid: 5-min steps are regular
        sub = df[df["day"].isin(days) & df["use"] & df["a"].notna() & df["b_shifted"].notna()]
        sub = sub[sub.groupby("day")["a"].transform("size") > 40]  # enough points per day
        out[lag] = sub.groupby("day").apply(lambda x: x["a"].corr(x["b_shifted"])).mean()
    return pd.Series(out)


def peak_lag_minutes(curve: pd.Series, step_min: int = 5) -> float:
    """Lag of the correlation maximum, refined between grid steps by a parabola through the
    best point and its two neighbours (sub-step estimate; precision about +-2.5 min)."""
    k = curve.idxmax()
    y0, y1, y2 = curve[k - 1], curve[k], curve[k + 1]
    denom = y0 - 2 * y1 + y2
    offset = 0.5 * (y0 - y2) / denom if denom else 0.0
    return (k + offset) * step_min


def gamma_regression(y: pd.Series, g: pd.Series, t: pd.Series, groups: pd.Series) -> dict[str, float]:
    """Temperature coefficient (%/C) from y = a + b (G - 800) + c (T - 25).

    y: normalised power P / (P0 x G/1000); g: POA; t: cell temperature; groups: e.g. year-month.
    pooled:       one fit over all points (season and temperature are mixed);
    within_month: all three variables are centred on their monthly mean first, so only the
                  temperature variation inside a month is used (removes seasonal spectrum,
                  soiling and degradation, which change slowly).
    gamma = c / a (slope relative to the level at 25 C) in %/C.
    """
    df = pd.DataFrame({"y": y, "g": g - 800, "t": t - 25, "grp": groups}).dropna()
    x = np.c_[np.ones(len(df)), df["g"], df["t"]]
    a, _, c = np.linalg.lstsq(x, df["y"], rcond=None)[0]
    pooled = 100 * c / a
    centred = df[["y", "g", "t"]] - df.groupby("grp")[["y", "g", "t"]].transform("mean")
    _, c_w = np.linalg.lstsq(centred[["g", "t"]].to_numpy(), centred["y"], rcond=None)[0]
    # Level a of the pooled fit is reused: centring removes the intercept.
    within = 100 * c_w / a
    return {"pooled": pooled, "within_month": within, "n": len(df), "y_mean": df["y"].mean()}


def gamma_per_inverter(
    inputs: pd.DataFrame,
    ac: pd.DataFrame,
    t_cell: pd.Series,
    aoi: pd.Series,
    config: Mapping[str, Any],
    p0_kw: float,
) -> pd.DataFrame:
    """gamma for every inverter on steps where the model assumptions hold.

    Selection: POA 700-900 W/m2 (narrow band, so G and T are less correlated), AOI < 40 deg
    (angle losses small), AC below 95 % of the rating (no clipping) and normalised power above
    0.5 (the inverter is running: excludes outages and partial trips).
    """
    rating = config["inverters"]["ac_kw"][0]["ac_kw"]
    poa = inputs["poa_wm2"]
    base = poa.between(700, 900) & (aoi < 40) & t_cell.notna()
    month = inputs.index.tz_convert(LOCAL_TZ).strftime("%Y-%m")
    rows = {}
    for col in ac.columns:
        y = ac[col] / (p0_kw * poa / 1000)
        m = base & y.notna() & (ac[col] < 0.95 * rating) & (y > 0.5)
        rows[col] = gamma_regression(y[m], poa[m], t_cell[m], month[m.to_numpy()])
    return pd.DataFrame(rows).T


def load_irradiance_sensors(config: Mapping[str, Any], channels: Sequence[str]) -> pd.DataFrame:
    """Irradiance channels from the clean dataset, flagged values set to NaN."""
    flags = [f"{t}__{c}" for c in channels for t in ("flag_stale", "flag_range")]
    data = load_clean(config, columns=[*channels, *flags])
    return pd.DataFrame({c: valid(data, c) for c in channels})


def daily_ratio(num: pd.Series, den: pd.Series, mask: pd.Series, min_steps: int = 40) -> pd.Series:
    """Median ratio num/den per local day over masked steps; NaN for days with < min_steps.

    Ratios outside 0.7-1.3 are dropped (one sensor shaded or iced, not calibration).
    """
    r = (num / den)[mask]
    r = r[(r > 0.7) & (r < 1.3)]
    days = local_day(r.index)
    out = r.groupby(days).median().where(r.groupby(days).count() >= min_steps)
    out.index = pd.to_datetime(out.index)
    return out


def yoy_drift(daily: pd.Series) -> pd.DataFrame:
    """Year-over-year drift in %/year: monthly medians, then month m vs the same month a year
    earlier. Seasonal effects (spectrum, angle of incidence) cancel inside each pair.
    Returns (median, min, max, number of pairs)."""
    monthly = daily.groupby(daily.index.to_period("M")).median().dropna()
    pairs = [100 * (monthly[m] / monthly[m - 12] - 1) for m in monthly.index if m - 12 in monthly.index]
    if not pairs:
        return np.nan, np.nan, np.nan, 0
    return float(np.median(pairs)), float(np.min(pairs)), float(np.max(pairs)), len(pairs)


def plateau_steps(
    ac_raw: pd.DataFrame,
    rating_kw: float,
    min_steps: int = 6,
    tol_kw: float = 0.5,
    lo: float = 0.05,
    hi: float = 0.98,
) -> pd.DataFrame:
    """Boolean frame: steps inside a run of >= min_steps nearly constant AC values.

    A curtailment (power limit) holds the output on a level below the rating while the sun
    changes; clipping sits at ~the rating (hi) and zeros are downtime (lo). A step belongs to a
    run if it differs from the previous one by less than tol_kw. Unlike the stuck-value test of
    the cleaning (identical values), a small tolerance also catches plateaus with noise.
    """
    same = ac_raw.diff().abs() < tol_kw  # step equals the previous one
    in_band = (ac_raw > lo * rating_kw) & (ac_raw < hi * rating_kw)
    out = pd.DataFrame(False, index=ac_raw.index, columns=ac_raw.columns)
    for col in ac_raw.columns:
        s = same[col] & in_band[col]
        # run id: increases whenever the condition is False; the run length is a group size
        run_id = (~s).cumsum()
        length = s.groupby(run_id).transform("sum")  # steps with True in this run
        out[col] = s & (length >= min_steps - 1)  # n constant steps = n-1 equal pairs
    return out
