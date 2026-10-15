"""Phase 4.5 figures -- render the diagnostic audit tables as manuscript plots.

Reads the tables emitted by ``phase45_itte_diagnostics.py`` (single source of
truth -- no recompute, no drift) and produces three publication figures:

  1. ``phase45_conservatism_waterfall.png`` (4.5.1) -- where the 0.90 -> ~0.997
     coverage gap comes from: base per-arm CP, alpha->alpha/2 inflation, the
     Bonferroni union (a reduction), and the dominant interval-arithmetic slack.
  2. ``phase45_overlap_diagnostics.png`` (4.5.5) -- inf%, weighting efficiency
     ESS/n_cal, max normalized test weight and finite-band width vs overlap scale
     (small multiples -- never a dual axis). The Phase-6.5 target panel.
  3. ``phase45_modulation_width.png`` (4.5.4) -- ITTE band width by modulation and
     grid resolution, with/without the whole-interval grid slack.

Design: dataviz default palette (categorical blue/aqua/yellow in fixed order;
blue<->red for the diverging waterfall), recessive grid/axes, direct labels.

    .venv/bin/python experiments/phase45_figures.py \
        --in-dir experiments/out/phase45 --out-dir experiments/figures/phase45
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- dataviz default palette (light surface) -------------------------------
BLUE = "#2a78d6"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#dddddd"
SURFACE = "#fcfcfb"
_MODCOLOR = {"constant": YELLOW, "pointwise-sd": BLUE, "lipschitz": AQUA}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


_NUM = re.compile(r"^-?\d+(?:\.\d+)?$")


def _numeric_rows(path: Path, ncols: int):
    """All whitespace-split lines with exactly ``ncols`` numeric tokens -> float rows."""
    rows = []
    for line in path.read_text().splitlines():
        toks = line.split()
        if len(toks) == ncols and all(_NUM.match(t) for t in toks):
            rows.append([float(t) for t in toks])
    return np.array(rows, dtype=float) if rows else np.empty((0, ncols))


# =========================================================== Fig 1: waterfall
def fig_waterfall(in_dir: Path, out_dir: Path, target=0.90):
    """Reconstruct the coverage waterfall from the decompose ladder table."""
    rows = _numeric_rows(in_dir / "decompose.txt", 6)  # dim arm@a arm@a/2 both ITTE ITTE+grid
    if rows.size == 0:
        print("! decompose.txt has no ladder rows; skipping waterfall")
        return
    arm_a, arm_h, both, itte = rows[:, 1].mean(), rows[:, 2].mean(), rows[:, 3].mean(), rows[:, 4].mean()
    steps = [
        ("target\n(1-alpha)", target, None),
        ("base per-arm\nCP", (arm_a - target) * 100, "add"),
        ("alpha/2\nper arm", (arm_h - arm_a) * 100, "add"),
        ("Bonferroni\nunion", (both - arm_h) * 100, "sub"),
        ("interval\narithmetic", (itte - both) * 100, "add"),
        ("empirical\nITTE", itte, None),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    _style(ax)
    base = target * 100
    x = np.arange(len(steps))
    for i, (label, val, kind) in enumerate(steps):
        if kind is None:  # anchor bar from 0
            h = val * 100
            ax.bar(i, h, width=0.62, color=INK2, zorder=3)
            ax.text(i, h + 0.4, f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, color=INK, fontweight="bold")
            base = h
        else:
            color = BLUE if kind == "add" else RED
            bottom = base if kind == "add" else base + val
            ax.bar(i, abs(val), width=0.62, bottom=bottom, color=color, zorder=3)
            sign = "+" if val >= 0 else "−"
            ax.text(i, base + max(val, 0) + 0.4, f"{sign}{abs(val):.2f}pp",
                    ha="center", va="bottom", fontsize=9, color=color, fontweight="bold")
            base = base + val
        if i < len(steps) - 1:
            ax.plot([i + 0.31, i + 1 - 0.31], [base, base], color=INK2,
                    linewidth=0.8, linestyle=(0, (3, 3)), zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in steps], fontsize=8.5, color=INK2)
    ax.set_ylabel("simultaneous coverage (%)", fontsize=10, color=INK2)
    ax.set_ylim(target * 100 - 1.5, 101)
    ax.set_title("Where the ITTE conformal band's conservatism enters\n"
                 "interval arithmetic is the dominant driver, not scalarization or the alpha-split",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    out = out_dir / "phase45_conservatism_waterfall.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


# =========================================================== Fig 2: overlap
def fig_overlap(in_dir: Path, out_dir: Path, alpha=0.1):
    """Small-multiples of the overlap diagnostics (never a dual axis)."""
    # scale dim cov_uncond cov_finite inf% ESS ESS/ncal max_wt w_finite
    rows = _numeric_rows(in_dir / "overlap.txt", 9)
    if rows.size == 0:
        print("! overlap.txt has no rows; skipping overlap figure")
        return
    scales = np.unique(rows[:, 0])
    agg = {s: rows[rows[:, 0] == s].mean(axis=0) for s in scales}  # avg over dims
    inf = np.array([agg[s][4] for s in scales])
    ess_eff = np.array([agg[s][6] for s in scales])
    maxw = np.array([agg[s][7] for s in scales])
    width = np.array([agg[s][8] for s in scales])

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4))
    fig.patch.set_facecolor(SURFACE)
    panels = [
        (axes[0, 0], inf, "unbounded bands  inf%  (%)", "% of test bands = +inf", RED, True),
        (axes[0, 1], ess_eff, "weighting efficiency  ESS / n_cal", "1.0 = randomized / uniform", BLUE, False),
        (axes[1, 0], maxw, "max normalized test weight", f"+inf atom fires when > alpha/2 = {alpha/2:.2f}", VIOLET, False),
        (axes[1, 1], width, "finite-band width (mean)", "mean band width | bounded", GREEN, False),
    ]
    for ax, y, title, sub, color, is_pct in panels:
        _style(ax)
        ax.plot(scales, y, color=color, linewidth=2.0, marker="o", markersize=6,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
        for xs, ys in zip(scales, y):
            ax.annotate(f"{ys:.2f}" if not is_pct else f"{ys:.0f}",
                        (xs, ys), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel(f"overlap scale (higher = weaker overlap)\n{sub}", fontsize=8.5, color=INK2)
        ax.set_xticks(scales)
    if maxw.max() > alpha / 2:
        axes[1, 0].axhline(alpha / 2, color=RED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    fig.suptitle("Weak overlap drives the +inf atom -- the Phase-6.5 stabilization target",
                 fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = out_dir / "phase45_overlap_diagnostics.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


# =========================================================== Fig 3: modulation
def fig_modulation(in_dir: Path, out_dir: Path):
    """Dot plot: ITTE width by modulation x resolution, grid-slack off/on.

    Differences here are ~1-7%, so a 0-baseline bar would crush them to
    invisibility; a dot plot with an honest (non-zero) width range is the right
    form for comparing small differences.
    """
    path = in_dir / "modulation.txt"
    mods = ("constant", "pointwise-sd", "lipschitz")
    tmp = {}  # (res, mod, gs_on) -> [widths]
    for line in path.read_text().splitlines():
        toks = line.split()
        if len(toks) == 7 and toks[1] in mods:
            res = int(toks[0]); mod = toks[1]; gs_on = toks[2].startswith("on")
            tmp.setdefault((res, mod, gs_on), []).append(float(toks[6]))
    if not tmp:
        print("! modulation.txt has no rows; skipping modulation figure")
        return
    data = {k: float(np.mean(v)) for k, v in tmp.items()}
    resolutions = sorted({k[0] for k in data})
    all_w = list(data.values())
    lo, hi = min(all_w), max(all_w)
    pad = 0.08 * (hi - lo)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, gs_on, tag in ((axes[0], False, "grid slack off (grid-only guarantee)"),
                           (axes[1], True, "grid slack on, L=2 (whole-interval guarantee)")):
        _style(ax)
        x = np.arange(len(resolutions))
        for mod in mods:
            vals = [data.get((r, mod, gs_on), np.nan) for r in resolutions]
            # lipschitz overplots pointwise-sd exactly -> dash it so both read
            ls = (0, (1, 1)) if mod == "lipschitz" else "-"
            ax.plot(x, vals, color=_MODCOLOR[mod], linewidth=2.0, linestyle=ls,
                    marker="o", markersize=8, markerfacecolor=_MODCOLOR[mod],
                    markeredgecolor=SURFACE, markeredgewidth=1.4, label=mod, zorder=3)
            for xi, vi in zip(x, vals):
                if mod != "lipschitz" and not np.isnan(vi):
                    ax.annotate(f"{vi:.2f}", (xi, vi), textcoords="offset points",
                                xytext=(0, 9), ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"res={r}" for r in resolutions], fontsize=9.5, color=INK2)
        ax.set_xlim(-0.4, len(resolutions) - 0.6)
        ax.set_title(tag, fontsize=9.5, color=INK, loc="left")
    axes[0].set_ylabel("mean ITTE band width", fontsize=10, color=INK2)
    axes[0].set_ylim(lo - pad, hi + 2.4 * pad)
    handles, labels = axes[0].get_legend_handles_labels()
    labels = [l + " (= pointwise-sd)" if l == "lipschitz" else l for l in labels]
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Modulation & grid-slack audit -- pointwise-sd = lipschitz (floor never binds); "
                 "constant ~1% wider; grid slack ~ 1/resolution",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    out = out_dir / "phase45_modulation_width.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("experiments/out/phase45"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/figures/phase45"))
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_waterfall(args.in_dir, args.out_dir, target=1 - args.alpha)
    fig_overlap(args.in_dir, args.out_dir, alpha=args.alpha)
    fig_modulation(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
