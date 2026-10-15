"""Phase 7B -- real-data demonstrations, excluding GEOM-Drugs 7B.3.

ORBIT is synthetic point-cloud data with generated counterfactuals, so it can
demonstrate TATE, CTATE, and ITTE end-to-end and report ITTE coverage against
known potential outcomes. SARS-CoV-2 CT is treated as a qualitative real-data
demonstration: the TATE construction has a semi-synthetic truth, but honest
per-patient counterfactual ITTE/CTATE coverage is not claimed.

Run:
    .venv/bin/python experiments/phase7_realdata_demo.py --mode orbit

SARS needs the data extra and Kaggle access:
    uv pip install -e ".[data]"
    .venv/bin/python experiments/phase7_realdata_demo.py --mode sars
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from tcda_uq.datasets import gen_covariate, gen_trt_prob, make_orbit_causal
from tcda_uq.estimators import CTATEDRLearner, cross_fit
from tcda_uq.metrics import covers_simultaneous
from tcda_uq.uq.asymptotic import ctate_confidence_band, multiplier_bootstrap_bands
from tcda_uq.uq.conformal import ITTEConformal, make_weight_fn
from tcda_uq.utils import numerical_integration


MU1 = np.array([1, 0.6, -0.7, 2.2, -1.0])
MU2 = np.array([0.4, -0.4, -0.6, 3.3, 3.0])
BETA = np.array([-0.5, -0.1, 0.6, 0.1, 0.1])
SIGMA = np.eye(5) * 0.5


def _interactions(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[1] < 3:
        return X
    extra = np.column_stack([X[:, 1] * X[:, 2], X[:, 0] * X[:, 2]])
    return np.hstack([X, extra])


def _rel_l1(est, truth):
    return float(np.mean(np.abs(est - truth)) / (np.mean(np.abs(truth)) + 1e-9))


def _finite_width(lo, hi):
    finite = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)
    width = np.where(finite, (hi - lo).mean(axis=1), np.nan)
    return finite, float(np.nanmean(width)) if finite.any() else np.inf


def _row(**kw):
    fields = [
        "level", "estimand", "uncertainty", "method", "n", "scenario",
        "dim", "coverage", "mean_width", "inf_pct", "extra",
    ]
    return "ROW " + " ".join(f"{k}={kw.get(k, '')}" for k in fields)


def _plot_tate(tseq, truth, aipw, path: Path, title: str, bands=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_hom = truth.shape[0]
    fig, axes = plt.subplots(1, n_hom, figsize=(6 * n_hom, 4), squeeze=False)
    for d in range(n_hom):
        ax = axes[0, d]
        if bands is not None:
            b = bands[d]
            ax.fill_between(tseq, b.lower, b.upper, alpha=0.25,
                            label=f"{b.level:.0%} confidence band")
        ax.plot(tseq, truth[d], label="truth", lw=3)
        ax.plot(tseq, aipw[d], label="AIPW", lw=2, ls="--")
        ax.set_title(f"{title} TATE H{d}")
        ax.set_xlabel("silhouette scale t")
        ax.set_ylabel(r"$\psi_d(t)$")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_itte(tseq, lo, hi, center, truth, path: Path, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.fill_between(tseq, lo, hi, alpha=0.25, label="ITTE band")
    ax.plot(tseq, center, label="center", lw=2)
    if truth is not None:
        ax.plot(tseq, truth, label="oracle ITTE", lw=2, ls=":")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_itte_grid(tseq, lo, hi, center, truth, idx, path: Path, title: str, ncols: int = 5):
    """Grid of ITTE prediction bands for several test units (small multiples)."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    path.parent.mkdir(parents=True, exist_ok=True)
    k = len(idx)
    ncols = min(ncols, k)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 2.3 * nrows),
                             squeeze=False, sharex=True)

    finite = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)
    ylim = None
    if finite.any():
        ymin = float(np.percentile(lo[finite], 1))
        ymax = float(np.percentile(hi[finite], 99))
        pad = 0.05 * (ymax - ymin + 1e-9)
        ylim = (ymin - pad, ymax + pad)

    for j in range(nrows * ncols):
        ax = axes[j // ncols, j % ncols]
        if j >= k:
            ax.axis("off")
            continue
        if finite[j]:
            ax.fill_between(tseq, lo[j], hi[j], alpha=0.25, color="tab:blue")
            if ylim is not None:
                ax.set_ylim(*ylim)
        else:
            ax.text(0.5, 0.5, "unbounded\nband", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="tab:red")
        ax.plot(tseq, center[j], lw=1.4, color="tab:blue")
        if truth is not None:
            ax.plot(tseq, truth[j], lw=1.4, ls=":", color="tab:orange")
        ax.axhline(0.0, color="0.6", lw=0.6, ls="--")
        ax.set_title(f"unit #{idx[j]}", fontsize=8)
        ax.tick_params(labelsize=7)

    handles = [
        Line2D([], [], color="tab:blue", lw=1.4, label="center"),
        Patch(color="tab:blue", alpha=0.25, label="ITTE band"),
    ]
    if truth is not None:
        handles.append(Line2D([], [], color="tab:orange", ls=":", lw=1.4, label="oracle ITTE"))
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=len(handles))
    fig.suptitle(title)
    fig.supxlabel("silhouette scale t", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_ctate(tseq, bands_by_label, truth, path: Path, title: str, d: int):
    """One panel per covariate eval-point: DR-learner CTATE curve + confidence band."""
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(bands_by_label)
    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), squeeze=False)
    for j, label in enumerate(labels):
        ax = axes[0, j]
        b = bands_by_label[label]
        ax.fill_between(tseq, b.lower, b.upper, alpha=0.25,
                        label=f"{b.level:.0%} confidence band")
        ax.plot(tseq, b.center, lw=2, label=r"DR-learner $\hat\tau_d(\cdot,x)$")
        if truth is not None:
            ax.plot(tseq, truth, lw=2, ls=":", label="TATE truth (ref)")
        ax.set_title(f"{title} CTATE H{d}  x={label}")
        ax.set_xlabel("silhouette scale t")
        ax.set_ylabel(r"$\tau_d(t, x)$")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fit_estimators(po_train, A_train, X_train, tseq, n_basis, seed):
    sample = (po_train[np.arange(len(A_train)), A_train], A_train, X_train)
    prop = LogisticRegression(max_iter=500)
    cf = cross_fit(
        sample,
        tseq,
        n_basis=n_basis,
        n_splits=2,
        propensity_estimator=prop,
        propensity_feature_fn=_interactions,
        random_state=seed,
    )
    ctate = CTATEDRLearner(n_basis=n_basis).fit(
        sample,
        tseq,
        cross_fit_result=cf,
        propensity_estimator=prop,
        propensity_feature_fn=_interactions,
        random_state=seed,
    )
    itte = ITTEConformal.fit(
        sample,
        tseq,
        n_basis=n_basis,
        propensity_estimator=prop,
        propensity_feature_fn=_interactions,
        weight_fn=make_weight_fn("overlap"),
        random_state=seed,
    )
    return sample, cf, ctate, itte


def _common_eval(
    name,
    po,
    tseq,
    X,
    A,
    n_train,
    m_test,
    n_basis,
    alpha,
    figure_dir,
    *,
    honest_itte,
    itte_units: int = 20,
):
    po_train, po_test = po[:n_train], po[n_train:n_train + m_test]
    X_train, X_test = X[:n_train], X[n_train:n_train + m_test]
    A_train = A[:n_train]
    truth_tate = po_train[:, 1].mean(axis=0) - po_train[:, 0].mean(axis=0)
    sample, cf, ctate, itte = _fit_estimators(po_train, A_train, X_train, tseq, n_basis, seed=0)

    tate_bands = multiplier_bootstrap_bands(cf, alpha=alpha, rng=0)

    print(f"\n{name} Phase 7B demo")
    print(f"{'dim':>3} {'estimand':>12} {'method':>12} {'metric':>12} {'value':>10}")
    for d in range(truth_tate.shape[0]):
        l1 = float(numerical_integration(np.abs(cf.aipw[d] - truth_tate[d]), tseq))
        print(f"{d:>3} {'TATE':>12} {'AIPW':>12} {'L1':>12} {l1:>10.4f}")
        print(_row(
            level=f"7B_{name}",
            estimand="TATE",
            uncertainty="point",
            method="AIPW",
            n=str(n_train),
            scenario=name,
            dim=str(d),
            coverage="",
            mean_width="",
            inf_pct="",
            extra=f"L1={l1:.4g}",
        ))
        b = tate_bands[d]
        width = float(numerical_integration(b.upper - b.lower, tseq))
        covered = bool(covers_simultaneous(b.lower[None], b.upper[None], truth_tate[d][None])[0])
        print(f"{d:>3} {'TATE':>12} {'mult_boot':>12} {'width':>12} {width:>10.4f}")
        print(_row(
            level=f"7B_{name}",
            estimand="TATE",
            uncertainty="confidence",
            method="mult_boot",
            n=str(n_train),
            scenario=name,
            dim=str(d),
            coverage=f"{int(covered)}",
            mean_width=f"{width:.4g}",
            inf_pct="",
            extra=f"level={b.level:.2f}_covers_semisynth_truth",
        ))

    _plot_tate(tseq, truth_tate, np.stack(cf.aipw), figure_dir / f"{name.lower()}_tate.png", name,
               bands=tate_bands)

    eval_points = {"mu1": MU1, "mid": 0.5 * (MU1 + MU2), "mu2": MU2}
    ctate_bands = {d: {} for d in range(truth_tate.shape[0])}
    for label, x in eval_points.items():
        tau_hat = ctate.predict(x)
        for d in range(truth_tate.shape[0]):
            rel = _rel_l1(tau_hat[d], truth_tate[d])
            print(f"{d:>3} {'CTATE':>12} {label:>12} {'rel_l1':>12} {rel:>10.4f}")
            print(_row(
                level=f"7B_{name}",
                estimand="CTATE_mean",
                uncertainty="point",
                method="DRLearner",
                n=str(n_train),
                scenario=f"{name}_{label}",
                dim=str(d),
                coverage="",
                mean_width="",
                inf_pct="",
                extra=f"rel_l1={rel:.4g}",
            ))
            band = ctate_confidence_band(ctate, x, d=d, alpha=alpha, rng=0)
            ctate_bands[d][label] = band
            width = float(numerical_integration(band.upper - band.lower, tseq))
            print(_row(
                level=f"7B_{name}",
                estimand="CTATE_mean",
                uncertainty="confidence",
                method="mult_boot",
                n=str(n_train),
                scenario=f"{name}_{label}",
                dim=str(d),
                coverage="",
                mean_width=f"{width:.4g}",
                inf_pct="",
                extra=f"level={band.level:.2f}",
            ))
    for d in range(truth_tate.shape[0]):
        _plot_ctate(
            tseq,
            ctate_bands[d],
            truth_tate[d],
            figure_dir / f"{name.lower()}_ctate_h{d}.png",
            name,
            d,
        )

    for d in range(truth_tate.shape[0]):
        lo, hi, center = itte.band_bounds(X_test, alpha, d=d)
        finite, width = _finite_width(lo, hi)
        inf_pct = 100.0 * float((~finite).mean())
        if honest_itte:
            target = po_test[:, 1, d, :] - po_test[:, 0, d, :]
            covered = covers_simultaneous(lo, hi, target)
            coverage = float(np.where(finite, covered, True).mean())
            cov_s = f"{coverage:.3f}"
            truth_curve = target[0]
        else:
            target = None
            coverage = None
            cov_s = ""
            truth_curve = None
        print(f"{d:>3} {'ITTE':>12} {'overlap_cp':>12} {'width':>12} {width:>10.4f}")
        print(_row(
            level=f"7B_{name}",
            estimand="ITTE",
            uncertainty="prediction" if honest_itte else "qualitative_prediction",
            method="overlap_cp",
            n=str(n_train),
            scenario=name,
            dim=str(d),
            coverage=cov_s,
            mean_width=f"{width:.4g}",
            inf_pct=f"{inf_pct:.3g}",
            extra="honest_counterfactuals" if honest_itte else "no_honest_counterfactuals",
        ))
        _plot_itte(
            tseq,
            lo[0],
            hi[0],
            center[0],
            truth_curve,
            figure_dir / f"{name.lower()}_itte_h{d}.png",
            f"{name} ITTE band H{d}",
        )
        m = lo.shape[0]
        k = min(itte_units, m)
        idx = np.sort(np.random.default_rng(12345).choice(m, size=k, replace=False))
        truth_grid = target[idx] if honest_itte else None
        _plot_itte_grid(
            tseq,
            lo[idx],
            hi[idx],
            center[idx],
            truth_grid,
            idx,
            figure_dir / f"{name.lower()}_itte_h{d}_grid.png",
            f"{name} ITTE bands H{d} — {k} random test units ({int(round((1 - alpha) * 100))}% level)",
        )


def run_orbit(args):
    rng = np.random.default_rng(args.seed)
    n_total = args.n + args.m_test
    causal = make_orbit_causal(
        n=n_total,
        num_pts=args.num_pts,
        interval=(0, 0.2),
        r=3,
        resolution=args.resolution,
        rng=rng,
    )
    _, _, X = gen_covariate(MU1, MU2, SIGMA, n_total, rng=rng)
    _, A = gen_trt_prob(X, BETA, rng=rng)
    _common_eval(
        "ORBIT",
        causal["potential_outcomes"],
        causal["tseq"],
        X,
        A,
        args.n,
        args.m_test,
        args.n_basis,
        args.alpha,
        args.figure_dir,
        honest_itte=True,
        itte_units=args.itte_units,
    )


def run_sars(args):
    from tcda_uq.datasets.sarscov2 import (
        download_ct,
        images_to_silhouettes,
        load_ct_images,
        make_sarscov2_causal,
    )

    rng = np.random.default_rng(args.seed)
    if args.data_root is not None:
        root = str(args.data_root)
        print(f"Using local SARS-CoV-2 CT dataset at: {root}")
    else:
        root = download_ct(cache_dir=args.cache)
    infected, noninfected = load_ct_images(root)
    if args.max_images is not None:
        infected = infected[:args.max_images]
        noninfected = noninfected[:args.max_images]
    print(f"Loaded SARS-CoV-2 CT images: COVID={len(infected)}, non-COVID={len(noninfected)}")
    phi_inf = images_to_silhouettes(
        infected, interval=(0, 1), r=0.1, resolution=args.resolution,
        workers=args.workers,
    )
    phi_non = images_to_silhouettes(
        noninfected, interval=(0, 1), r=0.1, resolution=args.resolution,
        workers=args.workers,
    )
    n_total = args.n + args.m_test
    causal = make_sarscov2_causal(phi_inf, phi_non, n=n_total, p=args.sars_p, rng=rng)
    _, _, X = gen_covariate(MU1, MU2, SIGMA, n_total, rng=rng)
    _, A = gen_trt_prob(X, BETA, rng=rng)
    tseq = np.linspace(0, 1, args.resolution)
    _common_eval(
        "SARS",
        causal["potential_outcomes"],
        tseq,
        X,
        A,
        args.n,
        args.m_test,
        args.n_basis,
        args.alpha,
        args.figure_dir,
        honest_itte=False,
        itte_units=args.itte_units,
    )
    print("SARS CTATE/ITTE rows are qualitative only: per-patient counterfactual coverage is not claimed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["orbit", "sars", "all"], default="orbit")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--m-test", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-basis", type=int, default=3)
    ap.add_argument("--resolution", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--num-pts", type=int, default=300,
                    help="ORBIT points per orbit; use 1000 for the full reproduction")
    ap.add_argument("--cache", default="./experiments/data/sarscov2")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="local SARS dataset root with COVID/ and non-COVID/ subfolders; "
                         "skips the kagglehub download when set")
    ap.add_argument("--sars-p", type=float, default=0.75)
    ap.add_argument("--max-images", type=int, default=None,
                    help="optional SARS image cap for smoke tests")
    ap.add_argument("--workers", type=int, default=1,
                    help="SARS image-to-silhouette worker count")
    ap.add_argument("--itte-units", type=int, default=20,
                    help="number of random test units drawn in the ITTE band grid figure")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--figure-dir", type=Path, default=Path("experiments/figures/phase7"))
    args = ap.parse_args()

    def _run():
        if args.mode in ("orbit", "all"):
            run_orbit(args)
        if args.mode in ("sars", "all"):
            run_sars(args)
        print("\nGEOM-Drugs 7B.3 skipped: val_data.pickle is still unavailable.")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f, contextlib.redirect_stdout(f):
            _run()
        print(args.output.read_text())
    else:
        _run()


if __name__ == "__main__":
    main()
