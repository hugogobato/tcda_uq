"""Phase 6 figures -- render the topology + stabilized-weighting tables as manuscript plots.

Reads the tables emitted by the Phase-6 experiments (single source of truth -- no recompute)
and produces:

  1. ``phase6_stabilized_overlap.png`` (6.5/6.6, HEADLINE) -- inf% and finite width vs overlap
     scale, one line per weighting method: naive blows up (inf% -> ~34%) while the
     overlap/matching stabilizers stay bounded at ~flat width -- the positivity-stabilization
     win.
  2. ``phase6_degree_multiplicity.png`` (6.4) -- total simultaneous-over-degree band width by
     scheme (joint / Bonferroni / Sidak) at two overlap scales; the joint sup-over-degree
     scheme is tightest and resists the weak-overlap unboundedness.
  3. ``phase6_r_study.png`` (6.3) -- ITTE band width vs the power exponent r, per homology dim,
     with the coverage line -- the width/coverage tuning knob unique to the topological setting.
  4. ``phase6_bridge.png`` (6.1) -- the Theorem-5.3 bridge constant K vs r (log scale): the
     bound always holds, but K explodes for r>1 unless diagrams are persistence-thresholded.

Design: the repo's dataviz default palette (categorical order, recessive grid, direct labels),
matching the Phase 4.5 / 5 figures.

    .venv/bin/python experiments/phase6_figures.py \
        --in-dir experiments/out/phase6 --out-dir experiments/figures/phase6
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

# method -> (color, label) in a fixed categorical order
_METHOD_STYLE = [
    ("naive", RED, "naive 1/pi"),
    ("clip", YELLOW, "clip"),
    ("shrink", VIOLET, "shrink"),
    ("tilted", AQUA, "tilted(g)"),
    ("matching", GREEN, "matching"),
    ("overlap", BLUE, "overlap"),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def _method_key(name: str) -> str:
    """Map a table method label (e.g. 'clip(eps=0.1)') to its base key ('clip')."""
    return re.split(r"[(\s]", name, 1)[0]


# =========================================================== Fig 1: stabilized overlap
def fig_stabilized(in_dir: Path, out_dir: Path):
    path = in_dir / "stabilized_overlap.txt"
    if not path.exists():
        print("! stabilized_overlap.txt missing; skipping headline figure")
        return
    # rows: scale method dim cov cov_fin inf width ess maxw  (method non-numeric)
    data = {}   # (mkey, dim) -> {scale: (inf, width)}
    for line in path.read_text().splitlines():
        t = line.split()
        if len(t) != 9:
            continue
        try:
            scale = float(t[0]); dim = int(t[2])
            inf = float(t[5]); width = float(t[6])
        except ValueError:
            continue
        mkey = _method_key(t[1])
        data.setdefault((mkey, dim), {})[scale] = (inf, width)
    if not data:
        print("! could not parse stabilized_overlap.txt; skipping")
        return

    dim = 1  # H1 (loops) as the representative degree
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    for ax in (axL, axR):
        _style(ax)
    for mkey, color, label in _METHOD_STYLE:
        d = data.get((mkey, dim))
        if not d:
            continue
        scales = sorted(d)
        infs = [d[s][0] for s in scales]
        widths = [d[s][1] for s in scales]
        axL.plot(scales, infs, "-o", color=color, lw=2, ms=4, label=label, zorder=3)
        axR.plot(scales, widths, "-o", color=color, lw=2, ms=4, label=label, zorder=3)
    axL.set_title("Unbounded bands vs overlap (H1)", color=INK, fontsize=11, loc="left")
    axL.set_xlabel("overlap scale (larger = thinner overlap)", color=INK2)
    axL.set_ylabel("inf %  (share of +inf bands)", color=INK2)
    axR.set_title("Finite band width vs overlap (H1)", color=INK, fontsize=11, loc="left")
    axR.set_xlabel("overlap scale", color=INK2)
    axR.set_ylabel("mean finite width", color=INK2)
    axR.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Phase 6.5/6.6 -- positivity-stabilized weighting eliminates the weak-overlap "
                 "+inf atom at flat width", color=INK, fontsize=12.5, x=0.5, y=1.02)
    fig.tight_layout()
    out = out_dir / "phase6_stabilized_overlap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# =========================================================== Fig 2: degree multiplicity
def fig_degree(in_dir: Path, out_dir: Path):
    path = in_dir / "degree_multiplicity.txt"
    if not path.exists():
        print("! degree_multiplicity.txt missing; skipping degree figure")
        return
    # scan blocks '-- overlap = X --' then scheme rows
    overlap, blocks = None, {}
    for line in path.read_text().splitlines():
        m = re.search(r"overlap = ([\d.]+)", line)
        if m:
            overlap = float(m.group(1))
            continue
        t = line.split()
        if overlap is not None and t and t[0] in ("joint", "bonferroni", "sidak"):
            try:
                jc, ji, tw = float(t[1]), float(t[2]), float(t[3])
            except (ValueError, IndexError):
                continue
            blocks.setdefault(overlap, {})[t[0]] = (jc, ji, tw)
    if not blocks:
        print("! could not parse degree_multiplicity.txt; skipping")
        return

    overlaps = sorted(blocks)
    schemes = ["joint", "bonferroni", "sidak"]
    colors = {"joint": BLUE, "bonferroni": YELLOW, "sidak": AQUA}
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    _style(ax)
    w = 0.24
    x = np.arange(len(overlaps))
    for k, sch in enumerate(schemes):
        vals = [blocks[o].get(sch, (np.nan, np.nan, np.nan))[2] for o in overlaps]
        vals = [v if np.isfinite(v) else 0.0 for v in vals]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=colors[sch], label=sch, zorder=3)
        for xi, o in zip(x, overlaps):
            ji = blocks[o].get(sch, (0, np.nan, 0))[1]
            v = blocks[o].get(sch, (0, 0, 0))[2]
            if np.isfinite(ji) and ji > 0.5:
                ax.text(xi + (k - 1) * w, (v if np.isfinite(v) else 0) + 0.1,
                        f"{ji:.0f}% inf", ha="center", va="bottom", fontsize=7.5, color=RED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"overlap {o}" for o in overlaps], color=INK2)
    ax.set_ylabel("total simultaneous-over-degree width (sum over H0..H2)", color=INK2)
    ax.set_title("Phase 6.4 -- degree-multiplicity schemes: joint sup-over-degree is tightest "
                 "and resists weak-overlap unboundedness", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    out = out_dir / "phase6_degree_multiplicity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# =========================================================== Fig 3: r study
def fig_r_study(in_dir: Path, out_dir: Path, target=0.90):
    path = in_dir / "r_study.txt"
    if not path.exists():
        print("! r_study.txt missing; skipping r-study figure")
        return
    rows = []
    for line in path.read_text().splitlines():
        t = line.split()
        if len(t) == 8 and re.match(r"^-?\d", t[0]):
            try:
                rows.append([float(x) for x in t])
            except ValueError:
                pass
    rows = np.array(rows)
    if rows.size == 0:
        print("! could not parse r_study.txt; skipping")
        return
    dims = sorted(set(rows[:, 1].astype(int)))
    fig, axes = plt.subplots(1, len(dims), figsize=(5.4 * len(dims), 4.4), squeeze=False)
    for j, d in enumerate(dims):
        ax = axes[0, j]
        _style(ax)
        sub = rows[rows[:, 1] == d]
        sub = sub[np.argsort(sub[:, 0])]
        r, cov, width = sub[:, 0], sub[:, 2], sub[:, 4]
        ax.plot(r, width, "-o", color=BLUE, lw=2, ms=5, label="width", zorder=3)
        ax.set_xlabel("power exponent r", color=INK2)
        ax.set_ylabel("mean ITTE band width", color=BLUE)
        ax.tick_params(axis="y", colors=BLUE)
        ax2 = ax.twinx()
        ax2.plot(r, cov, "-s", color=GREEN, lw=1.6, ms=4, label="coverage", zorder=3)
        ax2.axhline(target, color=INK2, lw=1, ls="--", zorder=2)
        ax2.set_ylabel("coverage", color=GREEN)
        ax2.set_ylim(min(target - 0.03, cov.min() - 0.01), 1.005)
        ax2.tick_params(axis="y", colors=GREEN)
        for s in ("top",):
            ax2.spines[s].set_visible(False)
        ax.set_title(f"H{d}", color=INK, fontsize=11, loc="left")
    fig.suptitle("Phase 6.3 -- power exponent r as a width/coverage knob (topological DGP): "
                 "H0 width explodes with r, H1 stays stable", color=INK, fontsize=12, y=1.02)
    fig.tight_layout()
    out = out_dir / "phase6_r_study.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# =========================================================== Fig 4: bridge constant
def fig_bridge(in_dir: Path, out_dir: Path):
    path = in_dir / "bridge.txt"
    if not path.exists():
        print("! bridge.txt missing; skipping bridge figure")
        return
    # parse the 'by r' table:  r  satisfied%  median_K  median_ratio
    rs, Ks, sat = [], [], []
    in_table = False
    for line in path.read_text().splitlines():
        if "satisfied%" in line and "median K" in line:
            in_table = True
            continue
        if in_table:
            t = line.split()
            m = re.match(r"^\d", t[0]) if t else None
            if not m:
                if t and not t[0][0].isdigit():
                    in_table = False
                continue
            try:
                rs.append(float(t[0]))
                sat.append(float(t[1].rstrip("%")))
                Ks.append(float(t[2]))
            except (ValueError, IndexError):
                in_table = False
    if not rs:
        print("! could not parse bridge.txt 'by r' table; skipping bridge figure")
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    _style(ax)
    ax.plot(rs, Ks, "-o", color=BLUE, lw=2, ms=6, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("power exponent r", color=INK2)
    ax.set_ylabel("bridge constant K = 1 + 2 L r c^(r-1)  (log)", color=INK2)
    for r, K, s in zip(rs, Ks, sat):
        ax.annotate(f"{s:.0f}% hold", (r, K), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color=INK2)
    ax.set_title("Phase 6.1 -- Theorem-5.3 bridge: the bound always holds, but K explodes for "
                 "r>1 (near-diagonal noise)", color=INK, fontsize=10.5, loc="left")
    fig.tight_layout()
    out = out_dir / "phase6_bridge.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("experiments/out/phase6"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/figures/phase6"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_stabilized(args.in_dir, args.out_dir)
    fig_degree(args.in_dir, args.out_dir)
    fig_r_study(args.in_dir, args.out_dir)
    fig_bridge(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
