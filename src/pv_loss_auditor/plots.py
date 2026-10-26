"""Static figures (matplotlib) for notebooks and README. Static PNG renders on GitHub."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

# Sequential single-hue ramp (light = low, dark = high), steps 100..700 of a validated blue.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
NO_DATA_GRAY = "#d9d8d4"  # neutral, not part of the ramp: "cannot be assessed", not "0 %"
INK, MUTED = "#0b0b0b", "#898781"


def availability_heatmap(table, groups, gaps=None, title="", min_gap_days=7):
    """Channel x month heatmap of availability in %.

    table  : DataFrame, index = channel labels, columns = monthly Periods, values 0..100 or NaN
    groups : list of (group name, [labels in display order]) - draws separators between groups
    gaps   : DataFrame with last_before / first_after / days; gaps > min_gap_days are marked
             as bars above the heatmap.
    """
    order = [lab for _, labs in groups for lab in labs]
    data = table.reindex(order)
    months = data.columns
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)

    # Two stacked axes sharing x: a thin strip for gaps on top, the heatmap below.
    fig, (ax_gap, ax) = plt.subplots(
        2,
        1,
        figsize=(15, 0.26 * len(order) + 2.6),
        sharex=True,
        gridspec_kw={"height_ratios": [0.9, len(order) * 0.26]},
    )
    ax.set_facecolor(NO_DATA_GRAY)
    # NaN cells are not drawn by pcolormesh, so the gray, hatched background shows through.
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            len(months),
            len(order),
            facecolor=NO_DATA_GRAY,
            hatch="///",
            edgecolor="#bfbdb7",
            linewidth=0,
            zorder=0,
        )
    )
    mesh = ax.pcolormesh(
        np.ma.masked_invalid(data.to_numpy(dtype=float)),
        cmap=cmap,
        vmin=0,
        vmax=100,
        edgecolors="white",
        linewidth=0.5,
        zorder=1,
    )

    ax.set_yticks(np.arange(len(order)) + 0.5, order, fontsize=8, color=INK)
    ax.invert_yaxis()
    # Year ticks at January; minor ticks every month.
    jan = [i for i, m in enumerate(months) if m.month == 1]
    ax.set_xticks([i + 0.5 for i in jan], [str(months[i].year) for i in jan], fontsize=9)
    ax.set_xticks(np.arange(len(months)) + 0.5, minor=True)
    ax.tick_params(axis="x", which="minor", length=2, color=MUTED)
    ax.tick_params(colors=INK)
    for x in jan:
        ax.axvline(x, color="white", linewidth=1.5, zorder=2)

    # Group separators and group names on the right.
    y = 0
    for name, labs in groups:
        if y:
            ax.axhline(y, color=INK, linewidth=1.2, zorder=3)
        ax.text(len(months) + 0.8, y + len(labs) / 2, name, va="center", fontsize=9, color=INK, rotation=270)
        y += len(labs)

    # Gap strip: periods with no rows in the files, one bar per distinct gap.
    ax_gap.set_ylim(0, 1)
    ax_gap.axis("off")
    ax_gap.text(0, 0.95, f"gaps > {min_gap_days} d (no rows)", fontsize=8, color=MUTED, va="top")
    if gaps is not None and len(gaps):
        big = gaps[gaps["days"] > min_gap_days].copy()
        # The same gap appears in several files with starts minutes to days apart: keep the
        # longest gap per start month.
        big["start_month"] = pd.to_datetime(big["last_before"]).dt.to_period("M")
        big = big.sort_values("days", ascending=False).drop_duplicates("start_month")

        def month_x(ts):
            """Timestamp -> x position (month index + fraction of the month)."""
            ts = pd.Timestamp(ts)
            return (pd.Period(ts, "M") - months[0]).n + (ts.day - 1) / ts.days_in_month

        for _, g in big.iterrows():
            x0, x1 = month_x(g["last_before"]), month_x(g["first_after"])
            ax_gap.plot([x0, x1], [0.25, 0.25], color=INK, linewidth=4, solid_capstyle="butt")
            ax_gap.text((x0 + x1) / 2, 0.45, f"{g['days']:.0f} d", ha="center", fontsize=7, color=INK)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.035, fraction=0.02)
    cbar.set_label("% of daylight snapshots with a valid value", fontsize=9, color=INK)
    ax.legend(
        handles=[
            Patch(
                facecolor=NO_DATA_GRAY,
                hatch="///",
                edgecolor="#bfbdb7",
                label="no daylight snapshots in month (gap or too few samples)",
            )
        ],
        loc="upper left",
        bbox_to_anchor=(0, -0.06),
        frameon=False,
        fontsize=8,
    )
    ax.set_xlim(0, len(months))
    ax.set_ylim(len(order), 0)
    ax_gap.set_title(title, loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def geometry_check_plot(day_poa, grid, ac_profiles, ac_rating_kw, title=""):
    """Three panels: POA on one clear day, nRMSE over tilt x azimuth, clear-day AC profiles.

    day_poa     : DataFrame on local time of one day; columns 'measured', 'fixed', 'tracker'
    grid        : DataFrame tilts x azimuths of nRMSE (%), from geometry.orientation_grid
    ac_profiles : DataFrame, index = local hour (float), columns = days; AC / rating
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.2, 1, 1.2]})
    hours = day_poa.index.hour + day_poa.index.minute / 60
    ax1.plot(hours, day_poa["measured"], color=INK, linewidth=2.2, label="measured (mean of cells)")
    ax1.plot(hours, day_poa["fixed"], color="#256abf", linewidth=1.4, label="model: fixed 20°/180°")
    ax1.plot(hours, day_poa["tracker"], color="#d9622b", linewidth=1.4, label="model: 1-axis tracker")
    ax1.set_xlabel("local time, h", fontsize=9)
    ax1.set_ylabel("POA, W/m²", fontsize=9)
    ax1.set_title(f"Clear day {day_poa.index[0]:%Y-%m-%d}", loc="left", fontsize=10, color=INK)
    ax1.legend(fontsize=8, frameon=False)

    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP[::-1])  # dark = better fit
    mesh = ax2.pcolormesh(
        np.arange(len(grid.columns) + 1),
        np.arange(len(grid.index) + 1),
        grid.to_numpy(dtype=float),
        cmap=cmap,
        edgecolors="white",
        linewidth=0.5,
    )
    for i, tilt in enumerate(grid.index):
        for j, az in enumerate(grid.columns):
            v = grid.loc[tilt, az]
            ax2.text(
                j + 0.5,
                i + 0.5,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if v < grid.to_numpy().mean() else INK,
            )
    ax2.set_xticks(np.arange(len(grid.columns)) + 0.5, grid.columns, fontsize=8)
    ax2.set_yticks(np.arange(len(grid.index)) + 0.5, grid.index, fontsize=8)
    ax2.set_xlabel("azimuth, °", fontsize=9)
    ax2.set_ylabel("tilt, °", fontsize=9)
    ax2.set_title("POA nRMSE (%) of fixed-tilt models", loc="left", fontsize=10, color=INK)
    fig.colorbar(mesh, ax=ax2, fraction=0.05, pad=0.02)

    for col in ac_profiles.columns:
        ax3.plot(ac_profiles.index, ac_profiles[col], color="#6da7ec", linewidth=0.7, alpha=0.6)
    ax3.plot(
        ac_profiles.index, ac_profiles.median(axis=1), color=INK, linewidth=2, label="median of clear days"
    )
    ax3.axhline(1, color="#d9622b", linewidth=1, linestyle="--", label=f"{ac_rating_kw:.0f} kW rating")
    ax3.set_xlabel("local time, h", fontsize=9)
    ax3.set_ylabel("AC / inverter rating", fontsize=9)
    ax3.set_title("Inverter 01 AC on clear days", loc="left", fontsize=10, color=INK)
    ax3.legend(fontsize=8, frameon=False, loc="center left")
    for ax in (ax1, ax3):
        ax.grid(color="#e6e5e1")
        ax.tick_params(colors=INK, labelsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def clock_offset_plot(offsets, base, other, windows, title=""):
    """Detected daily clock offset per channel, relative to a base rule, one panel per channel.

    The two candidate rules differ only in a few weeks a year, which are invisible on a plot of
    the offset itself. Plotting detected - base rule makes them stand out: a channel on the base
    rule stays at 0 everywhere; a channel on the other rule drops to -1 h in the shaded weeks.

    offsets : {channel label: DataFrame from timealign.daily_clock_offset (clear days only)}
    base    : (rule label, Series of expected offset per day)
    other   : (rule label, Series of expected offset per day)
    windows : [(first day, last day)] where the rules differ - shaded
    """
    base_label, base_line = base
    other_label, other_line = other
    fig, axes = plt.subplots(len(offsets), 1, figsize=(14, 2.0 * len(offsets)), sharex=True, squeeze=False)
    for ax, (label, off) in zip(axes[:, 0], offsets.items()):
        for a, b in windows:
            ax.axvspan(a, b + pd.Timedelta(days=1), color="#f3d3c0", linewidth=0, zorder=0)
        ax.axhline(0, color="#256abf", linewidth=1.2, label=base_label, zorder=1)
        ax.step(
            other_line.index,
            other_line - base_line,
            where="post",
            color="#d9622b",
            linewidth=1.2,
            label=other_label,
            zorder=1,
        )
        resid = off["offset_h"] - base_line.reindex(off.index)
        ax.scatter(off.index, resid, s=9, color=INK, zorder=3, label="detected, clear days")
        ax.set_ylim(-1.6, 0.6)
        ax.set_yticks([-1, 0], ["−1 h", "0"])
        ax.set_ylabel(label, fontsize=9, color=INK, rotation=0, ha="right", va="center")
        ax.grid(axis="y", color="#e6e5e1")
        ax.tick_params(colors=INK, labelsize=9)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False, ncol=3)
    axes[0, 0].set_title(title, loc="left", fontsize=11, color=INK, pad=22)
    fig.tight_layout()
    return fig


# Categorical slots 1-3 of the reference palette, always in this order (never cycled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
GRID = "#e6e5e1"


def _style(ax, title):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def gamma_plot(table, nameplate_pct, title=""):
    """gamma per inverter (within-month fit) with the DC-weighted nameplate value and -0.28."""
    fig, ax = plt.subplots(figsize=(12, 3.8))
    x = np.arange(len(table))
    ax.plot(x, table["within_month"], "o", color=SERIES[0], markersize=6, label="measured, within-month fit")
    ax.axhline(nameplate_pct, color=INK, linewidth=1.2, label=f"nameplate, DC-weighted ({nameplate_pct:.2f})")
    ax.axhline(-0.28, color=MUTED, linewidth=1, linestyle="--", label="295 W module datasheet (-0.28)")
    ax.axhline(-0.45, color=MUTED, linewidth=1, linestyle=":", label="300 W module datasheet (-0.45)")
    ax.set_xticks(x, [c[8:10] for c in table.index], fontsize=7)
    ax.set_xlabel("inverter", fontsize=9)
    ax.set_ylabel("gamma, %/°C", fontsize=9)
    # Legend below the axis label: inside the axes it would cover the reference lines.
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.2))
    _style(ax, title)
    fig.tight_layout()
    return fig


def drift_plot(monthly, title=""):
    """Monthly median ratio of each reference cell to pyranometer 02B (one line per cell)."""
    fig, ax = plt.subplots(figsize=(12, 3.8))
    x = monthly.index.to_timestamp()
    for color, col in zip(SERIES, monthly.columns):
        ax.plot(x, monthly[col], color=color, linewidth=2, marker="o", markersize=4, label=col)
        ax.annotate(
            col,
            (x[-1], monthly[col].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK,
            va="center",
        )
    ax.set_ylabel("cell / pyranometer 02B", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=3)
    _style(ax, title)
    fig.tight_layout()
    return fig


def pr_variants_plot(table, title=""):
    """Monthly PR: raw (A), STC-corrected (B), weather-corrected (C); excluding downtime."""
    fig, ax = plt.subplots(figsize=(12, 4))
    x = table.index.to_timestamp()
    labels = {
        "pr_raw_excl_down": "A  raw",
        "pr_stc_excl_down": "B  T_ref = 25 °C",
        "pr_wc_excl_down": "C  T_ref = period mean",
    }
    for color, (col, label) in zip(SERIES, labels.items()):
        ax.plot(x, table[col], color=color, linewidth=2, marker="o", markersize=4, label=label)
    ax.set_ylabel("PR (running inverters)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="lower left", ncol=3)
    _style(ax, title)
    fig.tight_layout()
    return fig


def downtime_heatmap(down_share, title=""):
    """Share of lit inverter-steps with the inverter down: inverters x local months.

    Sequential single hue, light = little downtime, dark = a lot; gray = no usable data.
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    cmap.set_bad(NO_DATA_GRAY)
    values = np.ma.masked_invalid(down_share.to_numpy(dtype=float))
    mesh = ax.pcolormesh(
        np.arange(down_share.shape[1] + 1),
        np.arange(down_share.shape[0] + 1),
        values,
        cmap=cmap,
        vmin=0,
        vmax=1,
        edgecolors="white",
        linewidth=0.4,
    )
    months = [str(p) for p in down_share.columns]
    step = 3
    ax.set_xticks(np.arange(0, len(months), step) + 0.5, months[::step], fontsize=7, rotation=90)
    ax.set_yticks(np.arange(len(down_share.index)) + 0.5, [c[8:10] for c in down_share.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_ylabel("inverter", fontsize=9)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.03, pad=0.01)
    cbar.set_label("share of lit time down", fontsize=8)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    return fig


WATERFALL_LABELS = {
    "temperature": "temperature",
    "dc_losses": "design DC",
    "inverter": "inverter eff.",
    "clipping": "clipping",
    "downtime": "downtime",
    "unknown_ac": "AC unknown",
    "channel_credit": "channel credit",
    "running_shortfall": "running shortfall",
    "reconciliation": "meter recon.",
    "after_inverters": "after inverters",
}
# Colour = what the step rests on: gray = model / assumption, orange = measured loss,
# green = credit (energy that was produced but not reported by the inverter channels).
_WATERFALL_COLOR = {
    "temperature": MUTED,
    "dc_losses": MUTED,
    "inverter": MUTED,
    "clipping": MUTED,
    "downtime": SERIES[1],
    "unknown_ac": SERIES[1],
    "channel_credit": SERIES[2],
    "running_shortfall": SERIES[1],
    "reconciliation": MUTED,
    "after_inverters": MUTED,
}


def waterfall_plot(table, steps, title=""):
    """One waterfall per period (panel): nominal -> steps -> meter, in GWh. Needs table columns
    nominal_kwh, metered_kwh and the loss steps (kWh, positive = loss)."""
    n = len(table)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(14, 4.6 * ((n + 1) // 2)), squeeze=False)
    for ax, (period, row) in zip(axes.ravel(), table.iterrows()):
        level = row["nominal_kwh"] / 1e6
        ax.bar(0, level, color=SERIES[0])
        for i, step in enumerate(steps, start=1):
            loss = row[step] / 1e6
            ax.bar(i, abs(loss), bottom=level - max(loss, 0), color=_WATERFALL_COLOR[step])
            level -= loss
        ax.bar(len(steps) + 1, row["metered_kwh"] / 1e6, color=SERIES[0])
        ax.set_xticks(
            range(len(steps) + 2),
            ["nominal"] + [WATERFALL_LABELS[s] for s in steps] + ["meter"],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_ylabel("GWh", fontsize=9)
        _style(ax, f"{title} {period}")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def loss_causes_plot(share, title=""):
    """Monthly losses as a share of the nominal energy, stacked by cause (columns of `share`)."""
    colors = [MUTED, SERIES[1], SERIES[0], BLUE_RAMP[1]]
    fig, ax = plt.subplots(figsize=(13, 4.2))
    x = share.index.to_timestamp()
    ax.stackplot(x, share.T.to_numpy() * 100, labels=list(share.columns), colors=colors[: share.shape[1]])
    ax.set_ylabel("% of nominal energy", fontsize=9)
    ax.set_ylim(0, share.sum(axis=1).max() * 100 * 1.2)  # headroom: the legend must not cover data
    ax.legend(fontsize=8, frameon=False, loc="upper left", ncol=4)
    _style(ax, title)
    fig.tight_layout()
    return fig
