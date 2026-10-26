"""Plant geometry and sensor checks on clear days: which mounting do the POA sensors 'see'?

Measured POA (reference cells) is compared with POA modelled for the metadata geometry (fixed
20 deg / 180 deg) and for a single-axis tracker, the alternative behind the NREL flag "Wrong
mounting config identified". Two irradiance inputs are used for the models:
- measured GHI -> Erbs decomposition -> Perez transposition: real sky, decomposition error;
- clear-sky Ineichen GHI/DNI/DHI -> Perez: no decomposition error, valid on clear points only.
All inputs must be on a tz-aware UTC index (see src/pv_loss_auditor/timealign.py).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pvlib

LOCAL_TZ = "America/New_York"  # only for grouping by local day / hour, never for storage


class Site:
    """Sun position, clear-sky and modelled POA for one location on one UTC time index."""

    def __init__(self, latitude: float, longitude: float, altitude: float, index: pd.DatetimeIndex) -> None:
        self.location = pvlib.location.Location(latitude, longitude, altitude=altitude)
        self.index = index
        self.solpos = self.location.get_solarposition(index)
        # Ineichen with monthly Linke turbidity from the climatology bundled with pvlib.
        self.clearsky = self.location.get_clearsky(index, model="ineichen", solar_position=self.solpos)
        self.dni_extra = pvlib.irradiance.get_extra_radiation(index)
        self.airmass = pvlib.atmosphere.get_relative_airmass(self.solpos["apparent_zenith"])

    def components(self, ghi: pd.Series | None = None) -> pd.DataFrame:
        """(GHI, DNI, DHI): measured GHI split with Erbs, or clear-sky if ghi is None."""
        if ghi is None:
            cs = self.clearsky
            return cs["ghi"], cs["dni"], cs["dhi"]
        erbs = pvlib.irradiance.erbs(ghi, self.solpos["zenith"], self.index)
        return ghi, erbs["dni"], erbs["dhi"]

    def poa(
        self, surface_tilt: float, surface_azimuth: float, ghi: pd.Series | None = None, albedo: float = 0.2
    ) -> pd.Series:
        """Modelled POA global (W/m2), Perez transposition. Tilt/azimuth may be Series (tracker)."""
        g, dni, dhi = self.components(ghi)
        return pvlib.irradiance.get_total_irradiance(
            surface_tilt,
            surface_azimuth,
            self.solpos["apparent_zenith"],
            self.solpos["azimuth"],
            dni,
            g,
            dhi,
            dni_extra=self.dni_extra,
            airmass=self.airmass,
            model="perez",
            albedo=albedo,
        )["poa_global"]

    def tracker_orientation(
        self, axis_tilt_deg: float, axis_azimuth_deg: float, max_angle_deg: float, backtrack: bool, gcr: float
    ) -> pd.DataFrame:
        """Surface tilt and azimuth of a single-axis tracker (pvlib); NaN at night -> flat."""
        st = pvlib.tracking.singleaxis(
            self.solpos["apparent_zenith"],
            self.solpos["azimuth"],
            axis_tilt=axis_tilt_deg,
            axis_azimuth=axis_azimuth_deg,
            max_angle=max_angle_deg,
            backtrack=backtrack,
            gcr=gcr,
        )
        return st["surface_tilt"].fillna(0), st["surface_azimuth"].fillna(axis_azimuth_deg + 90)

    def clear_points(self, ghi: pd.Series, window_length: int = 30) -> pd.Series:
        """Clear-sky flag per time step (Reno & Hansen, pvlib.clearsky.detect_clearsky).

        The method compares mean, max, line length, variability and slope of measured GHI with
        the clear-sky curve in a moving window. It was designed for 1-min data with a 10-min
        window; for 5-min data the window is widened to 30 min (6 points) and infer_limits=True
        scales the thresholds to the data interval.
        """
        return pvlib.clearsky.detect_clearsky(
            ghi.fillna(0), self.clearsky["ghi"], window_length=window_length, infer_limits=True
        )

    def daylight(self, min_elevation_deg: float) -> pd.Series:
        return self.solpos["apparent_elevation"] > min_elevation_deg


def local_day(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Local calendar day of each UTC timestamp (tz-naive dates, for grouping)."""
    return index.tz_convert(LOCAL_TZ).normalize().tz_localize(None)


def clear_days(clear: pd.Series, daylight: pd.Series, min_fraction: float) -> pd.DatetimeIndex:
    """Local days on which at least `min_fraction` of the daylight steps are flagged clear."""
    day = local_day(clear.index)
    frac = (clear & daylight).groupby(day).sum() / daylight.groupby(day).sum()
    return frac[frac >= min_fraction].index


def fit_metrics(model: pd.Series, measured: pd.Series, mask: pd.Series) -> dict[str, float]:
    """nRMSE and MBE in % of the mean measured value, R2, and number of points used."""
    ok = mask & model.notna() & measured.notna()
    err = model[ok] - measured[ok]
    mean = measured[ok].mean()
    return {
        "n": int(ok.sum()),
        "nRMSE_pct": 100 * np.sqrt((err**2).mean()) / mean,
        "MBE_pct": 100 * err.mean() / mean,
        "R2": 1 - (err**2).sum() / ((measured[ok] - mean) ** 2).sum(),
    }


def orientation_grid(
    site: Site,
    measured: pd.Series,
    mask: pd.Series,
    tilts: Sequence[float],
    azimuths: Sequence[float],
    ghi: pd.Series | None = None,
) -> pd.DataFrame:
    """nRMSE (%) of fixed-tilt POA for every (tilt, azimuth): a table, tilts x azimuths."""
    table = pd.DataFrame(
        index=pd.Index(tilts, name="tilt_deg"), columns=pd.Index(azimuths, name="azimuth_deg"), dtype=float
    )
    for tilt in tilts:
        for az in azimuths:
            table.loc[tilt, az] = fit_metrics(site.poa(tilt, az, ghi), measured, mask)["nRMSE_pct"]
    return table


def residual_by_hour(model: pd.Series, measured: pd.Series, mask: pd.Series) -> pd.Series:
    """Mean (model - measured) per local hour, in % of the overall mean measured value."""
    ok = mask & model.notna() & measured.notna()
    err = (model - measured)[ok]
    return 100 * err.groupby(err.index.tz_convert(LOCAL_TZ).hour).mean() / measured[ok].mean()


def sensor_agreement(df: pd.DataFrame, reference: str, mask: pd.Series) -> pd.DataFrame:
    """Each column vs the reference column on masked steps: r and ratio of sums."""
    rows = []
    for col in df.columns.drop(reference):
        ok = mask & df[col].notna() & df[reference].notna()
        x, y = df.loc[ok, reference], df.loc[ok, col]
        rows.append(
            {
                "sensor": col,
                "n": int(ok.sum()),
                "r": np.corrcoef(x, y)[0, 1],
                "ratio_to_reference": y.sum() / x.sum(),
            }
        )
    return pd.DataFrame(rows).set_index("sensor")


def clear_day_inverter_summary(ac: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Per inverter on the given local days: median and max daily peak (kW), median daily energy
    relative to the median inverter. Equal energies mean equal DC per inverter."""
    sel = ac[local_day(ac.index).isin(days)]
    day = local_day(sel.index)
    peak = sel.groupby(day).max()
    energy = sel.clip(lower=0).groupby(day).sum() / 12  # 5-min steps -> kWh
    rel = energy.median() / energy.median().median()
    return pd.DataFrame({"peak_median_kw": peak.median(), "peak_max_kw": peak.max(), "energy_rel": rel})
