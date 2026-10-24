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
        2, 1, figsize=(15, 0.26 * len(order) + 2.6), sharex=True,
        gridspec_kw={"height_ratios": [0.9, len(order) * 0.26]})
    ax.set_facecolor(NO_DATA_GRAY)
    # NaN cells are not drawn by pcolormesh, so the gray, hatched background shows through.
    ax.add_patch(plt.Rectangle((0, 0), len(months), len(order), facecolor=NO_DATA_GRAY,
                               hatch="///", edgecolor="#bfbdb7", linewidth=0, zorder=0))
    mesh = ax.pcolormesh(np.ma.masked_invalid(data.to_numpy(dtype=float)), cmap=cmap,
                         vmin=0, vmax=100, edgecolors="white", linewidth=0.5, zorder=1)

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
        ax.text(len(months) + 0.8, y + len(labs) / 2, name, va="center", fontsize=9,
                color=INK, rotation=270)
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
            ax_gap.text((x0 + x1) / 2, 0.45, f"{g['days']:.0f} d", ha="center",
                        fontsize=7, color=INK)

    cbar = fig.colorbar(mesh, ax=ax, pad=0.035, fraction=0.02)
    cbar.set_label("% of daylight snapshots with a valid value", fontsize=9, color=INK)
    ax.legend(handles=[Patch(facecolor=NO_DATA_GRAY, hatch="///", edgecolor="#bfbdb7",
                             label="no daylight snapshots in month (gap or too few samples)")],
              loc="upper left", bbox_to_anchor=(0, -0.06), frameon=False, fontsize=8)
    ax.set_xlim(0, len(months))
    ax.set_ylim(len(order), 0)
    ax_gap.set_title(title, loc="left", fontsize=11, color=INK)
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
    fig, axes = plt.subplots(len(offsets), 1, figsize=(14, 2.0 * len(offsets)), sharex=True,
                             squeeze=False)
    for ax, (label, off) in zip(axes[:, 0], offsets.items()):
        for a, b in windows:
            ax.axvspan(a, b + pd.Timedelta(days=1), color="#f3d3c0", linewidth=0, zorder=0)
        ax.axhline(0, color="#256abf", linewidth=1.2, label=base_label, zorder=1)
        ax.step(other_line.index, other_line - base_line, where="post", color="#d9622b",
                linewidth=1.2, label=other_label, zorder=1)
        resid = off["offset_h"] - base_line.reindex(off.index)
        ax.scatter(off.index, resid, s=9, color=INK, zorder=3, label="detected, clear days")
        ax.set_ylim(-1.6, 0.6)
        ax.set_yticks([-1, 0], ["−1 h", "0"])
        ax.set_ylabel(label, fontsize=9, color=INK, rotation=0, ha="right", va="center")
        ax.grid(axis="y", color="#e6e5e1")
        ax.tick_params(colors=INK, labelsize=9)
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False,
                      ncol=3)
    axes[0, 0].set_title(title, loc="left", fontsize=11, color=INK, pad=22)
    fig.tight_layout()
    return fig
