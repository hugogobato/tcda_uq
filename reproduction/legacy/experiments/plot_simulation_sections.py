"""Generate publication figures for the revised simulation studies section.

Produces:
  1. manuscript/figures/sim_bounded_benchmark.pdf:
     - Panel (a): Mean confidence coverage (TATE and CTATE profiles) vs sample size n
     - Panel (b): TATE band width vs n (log-log scale showing 1/sqrt(n) contraction)
     - Panel (c): Conformal prediction unbounded share and coverage across targets vs n

  2. manuscript/figures/sim_stress_diagnostics.pdf:
     - Panel (a): Functional design: Marginal TATE vs Fixed-Profile CTATE coverage under Base, Step, and Three components
     - Panel (b): Point-cloud design: TATE and CTATE profile coverage under good overlap (scale 1) vs weak overlap (scale 4)
     - Panel (c): Conformal prediction under overlap stress (scale 4): arm intersection coverage and unbounded shares
     - Panel (d): Topological power exponent r in {1,2,3,4,5}: width scaling vs invariant coverage
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.titlesize": 9.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "revision_20261010" / "audit" / "final_empirical_checks_20261011.json"
OUT_DIR = ROOT / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_PATH) as f:
    RAW = json.load(f)

RUNS = RAW["runs"]


def get_run(kind, **kwargs):
    def match_val(a, b):
        if a == b:
            return True
        try:
            return float(a) == float(b)
        except (ValueError, TypeError):
            return False

    matches = [
        r for r in RUNS
        if r["configuration"].get("kind") == kind and
        all(match_val(r["configuration"].get(k), v) for k, v in kwargs.items() if k != "cohort")
    ]
    if "cohort" in kwargs:
        cohort_matches = [r for r in matches if r.get("cohort") == kwargs["cohort"]]
        if cohort_matches:
            return cohort_matches[0]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Run not found: {kind}, {kwargs}")
    return matches[0]


def get_stat(run, metric, target=None, scheme=None, degree=None, profile=None):
    for s in run["statistics"]:
        if (s.get("metric") == metric and
            (target is None or s.get("target") == target) and
            (scheme is None or s.get("scheme") == scheme) and
            (degree is None or s.get("degree") == degree) and
            (profile is None or s.get("profile") == profile)):
            return s
    raise ValueError(f"Stat not found: {metric}, {target}, {scheme}, {degree}, {profile}")


# =========================================================================
# Figure 1: Bounded diagram-valued benchmark
# =========================================================================
def plot_bounded_benchmark():
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.3), constrained_layout=True)

    ns = [200, 500, 1000]
    colors = {
        "tate": "#0072B2",      # Blue
        "p0": "#D55E00",        # Vermilion
        "p1": "#009E73",        # Green
        "p2": "#CC79A7",        # Purple
        "full": "#0072B2",
        "overlap": "#D55E00",
        "matching": "#009E73"
    }

    # (a) Mean Coverage vs n
    ax = axes[0]
    tate_cov = []
    tate_se = []
    p0_cov, p0_se = [], []
    p1_cov, p1_se = [], []
    p2_cov, p2_se = [], []
    tate_width = []

    for n in ns:
        r = get_run("bounded", n=n)
        st_t = get_stat(r, "TATE", target="full", degree=0)
        tate_cov.append(st_t["coverage"])
        tate_se.append(st_t["coverage_mcse"])
        tate_width.append(st_t["width"])

        st_p0 = get_stat(r, "CTATE", target="fixed_profile", degree=0, profile=0)
        p0_cov.append(st_p0["coverage"])
        p0_se.append(st_p0["coverage_mcse"])

        st_p1 = get_stat(r, "CTATE", target="fixed_profile", degree=0, profile=1)
        p1_cov.append(st_p1["coverage"])
        p1_se.append(st_p1["coverage_mcse"])

        st_p2 = get_stat(r, "CTATE", target="fixed_profile", degree=0, profile=2)
        p2_cov.append(st_p2["coverage"])
        p2_se.append(st_p2["coverage_mcse"])

    x = np.arange(len(ns))
    w = 0.15
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.9, label="Nominal (0.90)")
    ax.errorbar(x - 1.5 * w, tate_cov, yerr=2 * np.array(tate_se), fmt="o", color=colors["tate"],
                capsize=2, markersize=3.5, label="TATE (marginal)")
    ax.errorbar(x - 0.5 * w, p0_cov, yerr=2 * np.array(p0_se), fmt="s", color=colors["p0"],
                capsize=2, markersize=3, label=r"CTATE $x^{(0)}$")
    ax.errorbar(x + 0.5 * w, p1_cov, yerr=2 * np.array(p1_se), fmt="^", color=colors["p1"],
                capsize=2, markersize=3, label=r"CTATE $x^{(1)}$")
    ax.errorbar(x + 1.5 * w, p2_cov, yerr=2 * np.array(p2_se), fmt="d", color=colors["p2"],
                capsize=2, markersize=3, label=r"CTATE $x^{(2)}$")

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Simultaneous coverage")
    ax.set_ylim(0.865, 0.925)
    ax.set_title("(a) Mean confidence coverage")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=6.2, ncol=2)

    # (b) Width vs n (log-log scale with 1/sqrt(n) rate)
    ax = axes[1]
    ax.plot(ns, tate_width, "o-", color=colors["tate"], linewidth=1.2, markersize=4, label="TATE width")
    # Reference line 1.6 / sqrt(n)
    n_dense = np.linspace(170, 1150, 50)
    ref = 1.60 / np.sqrt(n_dense)
    ax.plot(n_dense, ref, "--", color="black", linewidth=0.9, label=r"$1.60 / \sqrt{n}$ reference")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(160, 1250)
    ax.set_xticks(ns)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_yticks([0.05, 0.07, 0.10, 0.12])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_xlabel("Sample size $n$ (log scale)")
    ax.set_ylabel("Grid-average width (log scale)")
    ax.set_title(r"(b) $1/\sqrt{n}$ width contraction")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=6.8)

    # (c) Prediction unbounded share & coverage vs n
    ax = axes[2]
    unb_full, unb_ov, unb_mat = [], [], []
    for n in ns:
        r = get_run("bounded", n=n)
        st_f = get_stat(r, "ITTE", target="full", scheme="joint")
        st_o = get_stat(r, "ITTE", target="overlap", scheme="joint")
        st_m = get_stat(r, "ITTE", target="matching", scheme="joint")
        unb_full.append(st_f["unbounded_fraction"] * 100)
        unb_ov.append(st_o["unbounded_fraction"] * 100)
        unb_mat.append(st_m["unbounded_fraction"] * 100)

    bar_w = 0.22
    ax.bar(x - bar_w, unb_full, width=bar_w, color=colors["full"], label="Full target", alpha=0.85)
    ax.bar(x, unb_ov, width=bar_w, color=colors["overlap"], label="Overlap target", alpha=0.85)
    ax.bar(x + bar_w, unb_mat, width=bar_w, color=colors["matching"], label="Matching target", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("Sample size $n$")
    ax.set_ylabel("Unbounded prediction (%)")
    ax.set_ylim(-0.5, 27)
    ax.set_title("(c) Conformal unbounded share")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=6.8)
    # Annotation for coverage
    ax.text(0.5, 0.45, "All targets maintain\nITTE cov $\\geq 99.8\\%$ (nom. $95\\%$)",
            transform=ax.transAxes, ha="center", fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))

    out_file = OUT_DIR / "sim_bounded_benchmark.pdf"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved: {out_file}")


# =========================================================================
# Figure 2: Stress diagnostics across Functional and Point-Cloud designs
# =========================================================================
def plot_stress_diagnostics():
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.8), constrained_layout=True)

    # --- Panel (a): Functional design Base vs Step vs Three components ---
    ax = axes[0, 0]
    scenarios = ["Base (d=0,1)", "Step (d=0,1)", "Three-comp. (d=0,1,2)"]
    x = np.arange(len(scenarios))
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.9)

    # Base
    r_base = get_run("functional", scenario="base", degrees=2, scale=1.0, weights="known")
    t0_b = get_stat(r_base, "TATE", target="full", degree=0)
    t1_b = get_stat(r_base, "TATE", target="full", degree=1)
    cp0_b = [get_stat(r_base, "CTATE", target="fixed_profile", degree=0, profile=p)["coverage"] for p in range(3)]
    cp1_b = [get_stat(r_base, "CTATE", target="fixed_profile", degree=1, profile=p)["coverage"] for p in range(3)]

    # Step
    r_step = get_run("functional", scenario="step", degrees=2, scale=1.0, weights="known")
    t0_s = get_stat(r_step, "TATE", target="full", degree=0)
    t1_s = get_stat(r_step, "TATE", target="full", degree=1)
    cp0_s = [get_stat(r_step, "CTATE", target="fixed_profile", degree=0, profile=p)["coverage"] for p in range(3)]
    cp1_s = [get_stat(r_step, "CTATE", target="fixed_profile", degree=1, profile=p)["coverage"] for p in range(3)]

    # Three comp
    r_three = get_run("functional", scenario="base", degrees=3, scale=1.0, weights="known")
    t0_3 = get_stat(r_three, "TATE", target="full", degree=0)
    t1_3 = get_stat(r_three, "TATE", target="full", degree=1)
    t2_3 = get_stat(r_three, "TATE", target="full", degree=2)
    cp0_3 = [get_stat(r_three, "CTATE", target="fixed_profile", degree=0, profile=p)["coverage"] for p in range(3)]
    cp1_3 = [get_stat(r_three, "CTATE", target="fixed_profile", degree=1, profile=p)["coverage"] for p in range(3)]
    cp2_3 = [get_stat(r_three, "CTATE", target="fixed_profile", degree=2, profile=p)["coverage"] for p in range(3)]

    # Plot TATE with error bars
    ax.errorbar([0 - 0.15, 0 - 0.05], [t0_b["coverage"], t1_b["coverage"]],
                yerr=[2 * t0_b["coverage_mcse"], 2 * t1_b["coverage_mcse"]], fmt="o", color="#0072B2",
                capsize=2, markersize=3.5, label="TATE $H_0, H_1$")
    ax.errorbar([1 - 0.15, 1 - 0.05], [t0_s["coverage"], t1_s["coverage"]],
                yerr=[2 * t0_s["coverage_mcse"], 2 * t1_s["coverage_mcse"]], fmt="o", color="#0072B2",
                capsize=2, markersize=3.5)
    ax.errorbar([2 - 0.18, 2 - 0.08, 2 + 0.02], [t0_3["coverage"], t1_3["coverage"], t2_3["coverage"]],
                yerr=[2 * t0_3["coverage_mcse"], 2 * t1_3["coverage_mcse"], 2 * t2_3["coverage_mcse"]], fmt="o", color="#0072B2",
                capsize=2, markersize=3.5)

    # Plot CTATE profile ranges
    ax.plot([0 + 0.15, 0 + 0.15], [min(cp0_b + cp1_b), max(cp0_b + cp1_b)], "-", color="#D55E00", lw=2, label="CTATE profile range")
    ax.scatter([0 + 0.15] * len(cp0_b + cp1_b), cp0_b + cp1_b, marker="_", color="#D55E00", s=30)

    # Step CTATE collapses to 0!
    ax.scatter([1 + 0.15] * len(cp0_s + cp1_s), cp0_s + cp1_s, marker="x", color="#D55E00", s=40, lw=1.5)
    ax.text(1 + 0.18, 0.05, "CTATE = 0.0\n(basis collapse)", color="#D55E00", fontsize=6.8, ha="left", va="bottom",
            fontweight="bold")

    ax.plot([2 + 0.15, 2 + 0.15], [min(cp0_3 + cp1_3 + cp2_3), max(cp0_3 + cp1_3 + cp2_3)], "-", color="#D55E00", lw=2)
    ax.scatter([2 + 0.15] * len(cp0_3 + cp1_3 + cp2_3), cp0_3 + cp1_3 + cp2_3, marker="_", color="#D55E00", s=30)

    ax.set_xticks(x)
    ax.set_xticklabels(["Base", "Step (pert.)", "3-Component"], fontsize=7.5)
    ax.set_ylabel("Coverage frequency")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(a) Functional design: TATE vs CTATE collapse")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="center left", framealpha=0.9, fontsize=6.5)

    # --- Panel (b): Point-cloud design: Overlap scale 1 vs scale 4 ---
    ax = axes[0, 1]
    scales = ["Good overlap (scale 1)", "Weak overlap (scale 4)"]
    x = np.arange(len(scales))
    ax.axhline(0.90, color="gray", linestyle="--", linewidth=0.9, label="Nominal (0.90)")

    r_p1 = get_run("topological", scale=1.0, r=3.0, cohort="primary")
    t0_p1 = get_stat(r_p1, "TATE", target="full", degree=0)
    t1_p1 = get_stat(r_p1, "TATE", target="full", degree=1)
    cp0_p1 = [get_stat(r_p1, "CTATE", target="fixed_profile", degree=0, profile=p)["coverage"] for p in range(3)]
    cp1_p1 = [get_stat(r_p1, "CTATE", target="fixed_profile", degree=1, profile=p)["coverage"] for p in range(3)]

    r_p4 = get_run("topological", scale=4.0, r=3.0, cohort="primary")
    t0_p4 = get_stat(r_p4, "TATE", target="full", degree=0)
    t1_p4 = get_stat(r_p4, "TATE", target="full", degree=1)
    cp0_p4 = [get_stat(r_p4, "CTATE", target="fixed_profile", degree=0, profile=p)["coverage"] for p in range(3)]
    cp1_p4 = [get_stat(r_p4, "CTATE", target="fixed_profile", degree=1, profile=p)["coverage"] for p in range(3)]

    # TATE H0 and H1
    ax.errorbar([0 - 0.15], [t0_p1["coverage"]], yerr=2 * t0_p1["coverage_mcse"], fmt="o", color="#0072B2",
                capsize=2, markersize=3.5, label=r"TATE $H_0$")
    ax.errorbar([0 - 0.05], [t1_p1["coverage"]], yerr=2 * t1_p1["coverage_mcse"], fmt="s", color="#56B4E9",
                capsize=2, markersize=3.5, label=r"TATE $H_1$")

    ax.errorbar([1 - 0.15], [t0_p4["coverage"]], yerr=2 * t0_p4["coverage_mcse"], fmt="o", color="#0072B2",
                capsize=2, markersize=3.5)
    ax.errorbar([1 - 0.05], [t1_p4["coverage"]], yerr=2 * t1_p4["coverage_mcse"], fmt="s", color="#56B4E9",
                capsize=2, markersize=3.5)

    # CTATE ranges
    ax.plot([0 + 0.12, 0 + 0.12], [min(cp0_p1), max(cp0_p1)], "-", color="#D55E00", lw=1.8, label=r"CTATE $H_0$ range")
    ax.plot([0 + 0.22, 0 + 0.22], [min(cp1_p1), max(cp1_p1)], "-", color="#CC79A7", lw=1.8, label=r"CTATE $H_1$ range")
    ax.scatter([0 + 0.12] * len(cp0_p1), cp0_p1, marker="_", color="#D55E00", s=25)
    ax.scatter([0 + 0.22] * len(cp1_p1), cp1_p1, marker="_", color="#CC79A7", s=25)

    ax.plot([1 + 0.12, 1 + 0.12], [min(cp0_p4), max(cp0_p4)], "-", color="#D55E00", lw=1.8)
    ax.plot([1 + 0.22, 1 + 0.22], [min(cp1_p4), max(cp1_p4)], "-", color="#CC79A7", lw=1.8)
    ax.scatter([1 + 0.12] * len(cp0_p4), cp0_p4, marker="_", color="#D55E00", s=25)
    ax.scatter([1 + 0.22] * len(cp1_p4), cp1_p4, marker="_", color="#CC79A7", s=25)

    ax.set_xticks(x)
    ax.set_xticklabels(["Scale 1 (good)", "Scale 4 (weak)"], fontsize=7.5)
    ax.set_ylabel("Simultaneous coverage")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(b) Point-cloud design: overlap degradation")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=6.2)

    # --- Panel (c): Conformal prediction under weak overlap (scale 4) ---
    ax = axes[1, 0]
    r_pt4 = get_run("topological", scale=4.0, r=3.0, cohort="primary")
    targets = ["full", "overlap", "matching"]
    labels = ["Full", "Overlap", "Matching"]
    unb_pt = [get_stat(r_pt4, "ITTE", target=t, scheme="joint")["unbounded_fraction"] * 100 for t in targets]

    r_fn4 = get_run("functional", scenario="base", degrees=2, scale=4.0, weights="known")
    unb_fn = [get_stat(r_fn4, "ITTE", target=t, scheme="joint")["unbounded_fraction"] * 100 for t in targets]

    x = np.arange(len(targets))
    bar_w = 0.26
    ax.bar(x - bar_w / 2, unb_pt, width=bar_w, color="#D55E00", alpha=0.85, label="Point-cloud (scale 4)")
    ax.bar(x + bar_w / 2, unb_fn, width=bar_w, color="#0072B2", alpha=0.85, label="Functional (scale 4)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Target population tilt")
    ax.set_ylabel("Unbounded prediction (%)")
    ax.set_ylim(0, 80)
    ax.set_title("(c) Overlap-stabilized prediction (scale 4)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=6.8)
    ax.text(0.5, 0.48, "Arm-intersection cov holds\n95.9% - 97.0% across all targets",
            transform=ax.transAxes, ha="center", fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.9))

    # --- Panel (d): Topological power exponent r in {1,2,3,4,5} ---
    ax = axes[1, 1]
    rs = [1.0, 2.0, 3.0, 4.0, 5.0]
    widths_d0, widths_d1 = [], []
    tate_cov_d0 = []
    itte_covs = []

    for r_val in rs:
        r_run = get_run("topological", scale=1.0, r=r_val)
        s_t0 = get_stat(r_run, "TATE", target="full", degree=0)
        s_t1 = get_stat(r_run, "TATE", target="full", degree=1)
        s_it = get_stat(r_run, "ITTE", target="full", scheme="joint")
        widths_d0.append(s_t0["width"])
        widths_d1.append(s_t1["width"])
        tate_cov_d0.append(s_t0["coverage"])
        itte_covs.append(s_it["coverage"])

    ax2 = ax.twinx()
    l1 = ax.plot(rs, widths_d0, "o-", color="#0072B2", lw=1.2, markersize=4, label=r"TATE $H_0$ width")
    l2 = ax.plot(rs, widths_d1, "s-", color="#56B4E9", lw=1.2, markersize=4, label=r"TATE $H_1$ width")

    l3 = ax2.plot(rs, tate_cov_d0, "--", color="#009E73", lw=1.1, label=r"TATE $H_0$ cov. (nom 0.90)")
    l4 = ax2.plot(rs, itte_covs, ":", color="#D55E00", lw=1.2, label=r"ITTE cov. (nom 0.95)")

    ax.set_xlabel(r"Power exponent $r$")
    ax.set_ylabel("Mean band width", color="#0072B2")
    ax2.set_ylabel("Coverage", color="#009E73")
    ax.set_xticks(rs)
    ax2.set_ylim(0.85, 1.02)
    ax.set_title(r"(d) Power exponent $r$: estimand scaling")
    ax.grid(True, linestyle=":", alpha=0.5)

    # combined legend
    lines = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs, loc="center left", framealpha=0.9, fontsize=6.2)

    out_file = OUT_DIR / "sim_stress_diagnostics.pdf"
    fig.savefig(out_file)
    plt.close(fig)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    plot_bounded_benchmark()
    plot_stress_diagnostics()
