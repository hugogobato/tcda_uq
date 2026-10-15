"""Phase 5 figures -- covariate-adaptive conformal (the prediction side of CTATE).

Two manuscript figures:

  1. ``phase5_confidence_vs_prediction.png`` (Phase 5.3) -- the paper's organizing
     contrast, made concrete *at a fixed x*:
       (A) at one covariate ``x``: the CTATE **confidence** band (Phase 3; covers the
           mean ``tau_d(.,x)``, shrinks at 1/sqrt(n)) next to the CTATE-level
           **prediction** band (Phase 5; covers an individual ``delta`` drawn at ``x``,
           keeps a positive aleatoric width), with the true ``tau`` and a scatter of
           true ``delta`` draws at ``x``;
       (B) mean width vs ``n``: confidence width -> 0, prediction width -> a positive
           aleatoric floor. The reason a confidence band must never be read as an
           individual band.

  2. ``phase5_adaptive_width.png`` (Phase 5.1) -- covariate adaptivity:
       (A) band width vs the true local difficulty ``amp(X)``: the marginal Phase-4
           band is ~flat (X-blind); the normalized / meta-learner bands rise with
           difficulty (narrow where easy, wide where hard) -- the Phase-5 exit
           criterion, X-dependent width;
       (B) the aggregate Monte-Carlo evidence from ``hetero.txt`` (adaptivity
           ``w_corr`` and conditional-coverage gap ``cov_gap`` by method), if present.

Design: the repo's dataviz default palette + recessive ``_style`` (shared with
``phase45_figures.py``). Panel A of each figure is a single representative fit
(illustrative); the quantitative backbone stays the MC tables in
``experiments/out/phase5``.

    .venv/bin/python experiments/phase5_figures.py \
        --in-dir experiments/out/phase5 --out-dir experiments/figures/phase5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- dataviz default palette (shared with phase45_figures.py) --------------
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

_METHOD_COLOR = {"marginal": INK2, "normalized": BLUE, "localized": AQUA, "meta": VIOLET}
_METHOD_LABEL = {"marginal": "marginal (Ph 4)", "normalized": "normalized (5.1 A)",
                 "localized": "localized (5.1 A+B)", "meta": "meta-learner (5.2)"}


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)


def _feat(X):
    X = np.atleast_2d(X)
    return np.hstack([X, np.column_stack([X[:, 1] * X[:, 2], X[:, 0] * X[:, 2]])])


# ---------------------------------------------- phase5 table parser (string-aware)
def _phase5_table(path: Path):
    """Parse a phase5 coverage table into ``{(n, method, dim): {metric: value}}``."""
    cols = ["cov", "width", "inf", "cov_gap", "w_corr", "w_cv"]
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        t = line.split()
        if len(t) != 9 or t[0] == "n":
            continue
        try:
            n = int(t[0]); method = t[1]; dim = int(t[2])
            vals = [float(x) for x in t[3:]]
        except ValueError:
            continue
        out[(n, method, dim)] = dict(zip(cols, vals))
    return out


def _delta_draws_at(sim, x, d, k, rng):
    """``k`` true ITTE draws ``delta_d(., x)`` at a *fixed* covariate ``x``.

    Reconstructs the DGP's aleatoric part at ``x``: ``delta = tau_d(.,x) +
    amp(x) * noise_scale * ((z1 - z0) @ Psi)`` with ``z ~ N(0, spectral^2)`` -- the
    same functional noise :meth:`TriOracleSimulation._noise` uses, so the scatter is
    exactly what the prediction band must cover at ``x``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    tau = sim.true_ctate(x)[0, d]                                  # [res]
    amp = 1.0 + sim.hetero_scale * sim.noise_amplitude(x)[0]
    z1 = rng.normal(size=(k, sim.n_basis)) * sim.spectral
    z0 = rng.normal(size=(k, sim.n_basis)) * sim.spectral
    eps = amp * sim.noise_scale * ((z1 - z0) @ sim.Psi)            # [k, res]
    return tau[None, :] + eps


# ================================================= Fig 1: confidence vs prediction
def fig_confidence_vs_prediction(out_dir: Path, *, d=0, alpha=0.1, seed=0):
    from tcda_uq.datasets import TriOracleSimulation
    from tcda_uq.estimators import CTATEDRLearner
    from tcda_uq.uq.asymptotic import ctate_confidence_band
    from tcda_uq.uq.conformal import AdaptiveITTEConformal
    from sklearn.linear_model import LogisticRegression

    sim = TriOracleSimulation(n_hom_dim=2, resolution=60, n_basis=5,
                              noise_scale=0.3, hetero_scale=3.0, seed=seed)
    rng = np.random.default_rng(seed)
    n = 1200
    s = sim.sample(n, rng=rng)

    # a representative x in the higher-difficulty region (wide prediction band)
    st = sim.sample(400, rng=rng)
    amp = sim.noise_amplitude(st.X)
    x0 = st.X[np.argsort(amp)[int(0.8 * len(amp))]]                # ~80th pct difficulty

    learner = CTATEDRLearner(n_basis=5, feature_fn=_feat).fit(
        s.observed, sim.tseq, n_splits=2,
        propensity_estimator=LogisticRegression(max_iter=500),
        propensity_feature_fn=_feat, random_state=1)
    conf = ctate_confidence_band(learner, x0, d, alpha=alpha)

    adap = AdaptiveITTEConformal.fit(s.observed, sim.tseq, propensity_feature_fn=_feat,
                                     local_scale=True, localize=False, random_state=1)
    pred = adap.band(x0, alpha, d)

    tau = sim.true_ctate(np.atleast_2d(x0))[0, d]
    draws = _delta_draws_at(sim, x0, d, 40, np.random.default_rng(seed + 5))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    fig.patch.set_facecolor(SURFACE)

    # ---- Panel A: the two bands at fixed x
    ax = axes[0]; _style(ax)
    t = sim.tseq
    for dr in draws:
        ax.plot(t, dr, color=INK2, lw=0.5, alpha=0.28, zorder=1)
    ax.plot([], [], color=INK2, lw=0.5, alpha=0.5, label="individual $\\delta$ draws at $x$")
    ax.fill_between(t, pred.lower, pred.upper, color=BLUE, alpha=0.20, zorder=2,
                    label="prediction band (covers a draw)")
    ax.fill_between(t, conf.lower, conf.upper, color=RED, alpha=0.45, zorder=4,
                    label="confidence band (covers the mean)")
    ax.plot(t, tau, color=INK, lw=2.2, zorder=5, label=r"true $\tau_d(\cdot,x)$ (the mean)")
    ax.set_xlabel("t", fontsize=10, color=INK2)
    ax.set_ylabel(r"$\delta_d(t)\;/\;\tau_d(t,x)$", fontsize=10, color=INK2)
    ax.set_title("At a fixed x: confidence bands the mean, prediction bands a draw",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # ---- Panel B: width vs n (confidence -> 0, prediction -> floor)
    ax = axes[1]; _style(ax)
    ns = [100, 200, 400, 800, 1600]
    conf_w, pred_w = [], []
    for nn in ns:
        cw, pw = [], []
        for rep in range(3):
            rr = np.random.default_rng(1000 * rep + nn)
            ss = sim.sample(nn, rng=rr)
            lr = CTATEDRLearner(n_basis=5, feature_fn=_feat).fit(
                ss.observed, sim.tseq, n_splits=2,
                propensity_estimator=LogisticRegression(max_iter=500),
                propensity_feature_fn=_feat, random_state=1)
            cb = ctate_confidence_band(lr, x0, d, alpha=alpha)
            ad = AdaptiveITTEConformal.fit(ss.observed, sim.tseq, propensity_feature_fn=_feat,
                                           local_scale=True, localize=False, random_state=1)
            pb = ad.band(x0, alpha, d)
            cw.append(cb.mean_width()); pw.append(pb.mean_width())
        conf_w.append(np.mean(cw)); pred_w.append(np.mean(pw))
    ax.plot(ns, conf_w, color=RED, lw=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4,
            label="confidence width")
    ax.plot(ns, pred_w, color=BLUE, lw=2, marker="s", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4,
            label="prediction width")
    # 1/sqrt(n) reference anchored to the confidence curve
    ref = conf_w[0] * np.sqrt(ns[0]) / np.sqrt(ns)
    ax.plot(ns, ref, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=3,
            label=r"$1/\sqrt{n}$ reference")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("n (train + calibration)", fontsize=10, color=INK2)
    ax.set_ylabel("mean band width", fontsize=10, color=INK2)
    ax.set_title("Confidence width vanishes; prediction width floors",
                 fontsize=10.5, color=INK, loc="left")
    ax.set_xticks(ns); ax.set_xticklabels([str(x) for x in ns])
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)

    fig.suptitle("Confidence versus prediction at the CTATE level",
                 fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "phase5_confidence_vs_prediction.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


# ================================================= Fig 2: adaptive (X-dependent) width
def fig_adaptive_width(out_dir: Path, in_dir: Path, *, d=0, alpha=0.1, seed=0):
    from tcda_uq.datasets import TriOracleSimulation
    from tcda_uq.uq.conformal import (ITTEConformal, AdaptiveITTEConformal,
                                      ConformalMetaLearner)

    sim = TriOracleSimulation(n_hom_dim=2, resolution=40, n_basis=5,
                              noise_scale=0.3, hetero_scale=3.0, seed=seed)
    rng = np.random.default_rng(seed)
    s = sim.sample(1500, rng=rng)
    st = sim.sample(1200, rng=rng)
    amp = sim.noise_amplitude(st.X)

    models = {
        "marginal": ITTEConformal.fit(s.observed, sim.tseq, propensity_feature_fn=_feat,
                                      random_state=1),
        "normalized": AdaptiveITTEConformal.fit(s.observed, sim.tseq, propensity_feature_fn=_feat,
                                                local_scale=True, localize=False, random_state=1),
        "meta": ConformalMetaLearner.fit(s.observed, sim.tseq, propensity_feature_fn=_feat,
                                         feature_fn=_feat, random_state=1),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    fig.patch.set_facecolor(SURFACE)

    # ---- Panel A: width vs true difficulty (decile-binned)
    ax = axes[0]; _style(ax)
    q = np.quantile(amp, np.linspace(0, 1, 11))
    centers = 0.5 * (q[:-1] + q[1:])
    binid = np.clip(np.searchsorted(q, amp, side="right") - 1, 0, 9)
    for name in ("marginal", "normalized", "meta"):
        lo, hi, _ = models[name].band_bounds(st.X, alpha, d=d)
        fin = np.isfinite(hi).all(1) & np.isfinite(lo).all(1)
        w = (hi - lo).mean(1)
        yb = [np.nanmean(w[fin & (binid == b)]) for b in range(10)]
        ax.plot(centers, yb, color=_METHOD_COLOR[name], lw=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3,
                label=_METHOD_LABEL[name])
    ax.set_xlabel("true local difficulty  amp(X)  (low → high aleatoric noise)",
                  fontsize=9.5, color=INK2)
    ax.set_ylabel("mean band width", fontsize=10, color=INK2)
    ax.set_title("Adaptive width tracks difficulty; the marginal band is X-blind (flat)",
                 fontsize=10.5, color=INK, loc="left")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)

    # ---- Panel B: aggregate adaptivity + conditional-coverage gap from the MC table
    ax = axes[1]; _style(ax)
    tbl = _phase5_table(in_dir / "hetero.txt")
    if tbl:
        ns = sorted({k[0] for k in tbl})
        n_big = ns[-1]
        methods = [m for m in ("marginal", "normalized", "localized", "meta")
                   if any((n_big, m, dd) in tbl for dd in (0, 1))]
        # average metrics over homology dims at the largest n
        def _avg(m, key):
            vals = [tbl[(n_big, m, dd)][key] for dd in (0, 1) if (n_big, m, dd) in tbl]
            return float(np.mean(vals))
        wcorr = [_avg(m, "w_corr") for m in methods]
        gap = [_avg(m, "cov_gap") for m in methods]
        xpos = np.arange(len(methods))
        width = 0.38
        b1 = ax.bar(xpos - width / 2, wcorr, width, color=BLUE, zorder=3,
                    label="width–difficulty corr (↑ better)")
        b2 = ax.bar(xpos + width / 2, gap, width, color=RED, zorder=3,
                    label="conditional-coverage gap (↓ better)")
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK2)
        ax.set_xticks(xpos)
        ax.set_xticklabels([_METHOD_LABEL[m].replace(" ", "\n", 1) for m in methods],
                           fontsize=8)
        ax.set_ylabel("metric value", fontsize=10, color=INK2)
        ax.set_ylim(0, 1.14)
        ax.set_title(f"Aggregate Monte-Carlo evidence  (n={n_big})",
                     fontsize=10.5, color=INK, loc="left")
        ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    else:
        ax.text(0.5, 0.5, "run phase5_adaptive_coverage.py\n→ experiments/out/phase5/hetero.txt",
                ha="center", va="center", fontsize=10, color=INK2, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Covariate-adaptive prediction: width follows X-driven difficulty",
                 fontsize=12.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "phase5_adaptive_width.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("experiments/out/phase5"))
    ap.add_argument("--out-dir", type=Path, default=Path("experiments/figures/phase5"))
    ap.add_argument("--alpha", type=float, default=0.1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_confidence_vs_prediction(args.out_dir, alpha=args.alpha)
    fig_adaptive_width(args.out_dir, args.in_dir, alpha=args.alpha)


if __name__ == "__main__":
    main()
