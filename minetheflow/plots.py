"""Figures for the demo (matplotlib, static PNGs for slides)."""
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .network import PRESSURE_SENSORS, SENSORS

INK, INK2, MUTED, SURF, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#e34948"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f0efec", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#c2410c", "#eb6834", "#f0efec", "#3987e5", "#184f95"])

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def network_map(wn, coords, loc, leak_node, path):
    fig, ax = plt.subplots(figsize=(9, 6.2))
    for name, p in wn.pipes():
        (x1, y1), (x2, y2) = coords[p.start_node_name], coords[p.end_node_name]
        ax.plot([x1, x2], [y1, y2], color="#c9c8c2", lw=1 + 12 * p.diameter, zorder=1, solid_capstyle="round")
    js = wn.junction_name_list
    corr = np.array([loc["corr"][n] for n in js]) if loc else np.zeros(len(js))
    xy = np.array([coords[n] for n in js])
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=corr, cmap=SEQ, vmin=float(np.percentile(corr, 10)), vmax=float(corr.max()), s=70,
                    edgecolors=SURF, linewidths=1.5, zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Leak-signature correlation (night residuals)", color=INK2)
    cb.outline.set_visible(False)
    rx, ry = coords["R"]
    ax.scatter([rx], [ry], marker="s", s=120, color=INK2, zorder=4)
    ax.annotate("Reservoir", (rx, ry), xytext=(0, 12), textcoords="offset points", ha="center", color=INK2)
    ax.annotate("inlet main + flow meter Q_in", ((rx + coords["J00"][0]) / 2, ry), xytext=(0, 9),
                textcoords="offset points", ha="center", color=INK2, fontsize=9)
    for s, n in PRESSURE_SENSORS.items():
        x, y = coords[n]
        ax.scatter([x], [y], marker="D", s=95, facecolor="none", edgecolors=INK, linewidths=1.6, zorder=5)
        ax.annotate(s, (x, y), xytext=(7, 7), textcoords="offset points", fontsize=9, fontweight="bold")
    lx, ly = coords[leak_node]
    ax.scatter([lx], [ly], marker="*", s=420, color=ORANGE, edgecolors=SURF, linewidths=1.5, zorder=6)
    ax.annotate(f"true leak {leak_node}", (lx, ly), xytext=(10, -16), textcoords="offset points", color=INK, fontsize=9)
    if loc:
        best = loc["ranking"][0][0]
        bx, by = coords[best]
        ax.scatter([bx], [by], s=420, facecolor="none", edgecolors=INK, linewidths=2, linestyle="--", zorder=6)
        ax.annotate(f"estimated {best}", (bx, by), xytext=(10, 10), textcoords="offset points", fontsize=9)
    ax.set_title("Synthetic DMA: 42 junctions, 1 inlet meter + 9 pressure loggers\n"
                 "Cloud-side leak localisation with an EPANET sensitivity matrix")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def edge_signals(idx, raw, clean, flags, truth, labels, path):
    T0 = idx[0]
    panels = [("P2", "Data gap (logger dropout): filled from daily profile + level", 6, 5, 15),
              ("P4", "Spikes: removed by rate + deviation check", 6, 0, 24 * 4),
              ("P5", "Frozen sensor: 6 equal samples -> flagged, reconstructed", 7, 0, 22),
              ("P7", "Slow drift (-0.4 m/day): passes every edge check", 6, 0, 24 * 4)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, (s, title, d, h0, h1) in zip(axes.ravel(), panels):
        w = (idx >= T0 + pd.Timedelta(days=d, hours=h0)) & (idx < T0 + pd.Timedelta(days=d, hours=h1))
        t = idx[w]
        bad = (flags[s] != "good").to_numpy()[w]
        ax.plot(t, truth[s][w], color=INK, lw=1.0, label="true value (simulation)", zorder=2)
        ax.plot(t, raw[s][w], color=MUTED, lw=1.2, alpha=0.8, label="raw sensor reading", zorder=1)
        cl = clean[s][w].where(bad)
        ax.plot(t, cl, color=BLUE, lw=2.2, label="edge reconstruction", zorder=3)
        rb = raw[s][w].where(bad)
        ax.scatter(t, rb, s=16, color=ORANGE, zorder=4, label="flagged by edge")
        ax.set_title(f"{s}: {title}")
        ax.set_ylabel("pressure head (m)")
        lo, hi = np.nanmin(raw[s][w]), np.nanmax(raw[s][w])
        lo, hi = min(lo, np.nanmin(truth[s][w])), max(hi, np.nanmax(truth[s][w]))
        pad = max(1.0, (hi - lo) * 0.15)
        ax.set_ylim(lo - pad, hi + pad)
        ax.tick_params(axis="x", labelrotation=0, labelsize=8)
        import matplotlib.dates as md
        ax.xaxis.set_major_formatter(md.DateFormatter("%a %H:%M"))
        ax.xaxis.set_major_locator(md.HourLocator(interval=3 if (h1 - h0) <= 24 else 24))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Edge layer: per-sensor validation and reconstruction (each logger sees only itself)",
                 x=0.01, ha="left", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def cloud_residuals(hyb, cld, leak_start, test_start, path):
    fig = plt.figure(figsize=(13, 7.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[5, 1, 1], width_ratios=[60, 1], hspace=0.35, wspace=0.02)
    Z = hyb["Z"][hyb["Z"].index >= test_start - pd.Timedelta(days=1)]
    ax = fig.add_subplot(gs[0, 0])
    lim = 8
    im = ax.imshow(Z.T.to_numpy(), aspect="auto", cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), interpolation="nearest",
                   extent=[0, len(Z), len(SENSORS) - 0.5, -0.5])
    ax.set_yticks(range(len(SENSORS)), SENSORS)
    days = [i for i, t in enumerate(Z.index) if t.hour == 0]
    ax.set_xticks(days, [Z.index[i].strftime("%a %d") for i in days])
    ax.grid(False)
    li = list(Z.index).index(leak_start.floor("h"))
    for a in [ax]:
        a.axvline(li, color=INK, lw=1.5, ls="--")
    ax.annotate("leak starts", (li, len(SENSORS) - 0.5), xytext=(4, -14), textcoords="offset points", fontsize=9,
                annotation_clip=False, bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none"))
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 1]))
    cb.set_label("whitened residual  (measured - digital twin) / sigma", color=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Cloud layer (hybrid): residuals vs the EPANET digital twin, hourly\n"
                 "One sensor alone = sensor fault.  All sensors together, in a hydraulically consistent pattern = leak.", pad=10)
    colors = {"none": "#f0efec", "sensor_fault": YELLOW, "leak": ORANGE, "leak+sensor_fault": RED, "no_data": MUTED}
    for k, (name, r) in enumerate([("Hybrid", hyb), ("Cloud only", cld)]):
        a = fig.add_subplot(gs[1 + k, 0], sharex=ax)
        d = r["dec"].loc[Z.index]
        for i, lab in enumerate(d.label):
            a.axvspan(i, i + 1, color=colors[lab], lw=0)
            if lab in ("sensor_fault", "leak+sensor_fault") and (i == 0 or d.fault_sensor.iat[i - 1] != d.fault_sensor.iat[i]):
                a.text(i + 0.2, 0.5, d.fault_sensor.iat[i], va="center", fontsize=7, color=INK)
        a.set_yticks([0.5], [name])
        a.set_ylim(0, 1)
        a.grid(False)
        a.axvline(li, color=INK, lw=1.5, ls="--")
        a.tick_params(axis="x", labelbottom=(k == 1))
    from matplotlib.patches import Patch
    fig.legend([Patch(color=colors[c]) for c in ["sensor_fault", "leak", "leak+sensor_fault", "no_data"]],
               ["sensor fault", "leak", "leak + sensor fault", "no data (outage)"], loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.9, bottom=0.11)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def comparison_table(table, path):
    rows = list(table.index)
    cols = list(table.columns)
    wrap = lambda s, w: "\n".join(textwrap.wrap(str(s), w))
    cell = [[wrap(table.loc[r, c], 34) for c in cols] for r in rows]
    nl = [max(c.count("\n") + 1 for c in row + [wrap(r, 18)]) for row, r in zip(cell, rows)]
    fig, ax = plt.subplots(figsize=(14, 1.0 + 0.2 * (sum(nl) + 2 * len(rows) + 2)))
    ax.axis("off")
    t = ax.table(cellText=cell, rowLabels=[wrap(r, 18) for r in rows], colLabels=cols, loc="upper center",
                 cellLoc="left", rowLoc="left", bbox=[0.1, 0, 0.9, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor(GRID)
        c.set_facecolor(SURF if i % 2 else "#f4f3f0")
        units = [2] + [n + 1.2 for n in nl]
        c.set_height(units[i] / sum(units))
        if i == 0 or j == -1:
            c.get_text().set_fontweight("bold")
    ax.set_title("Edge only vs cloud only vs hybrid on the same 10 days of data", loc="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
