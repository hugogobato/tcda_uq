"""Persist CTATE/ITTE results for the applied manuscript (WP-P3.3 / P3.4).

Re-runs the exact frozen pipeline of experiments/pacce_p3_topology.py with
deterministic bootstrap seeds, persists the full curve arrays, and computes
per-profile summaries: peak conditional effect, band half-widths, fraction of
the filtration grid where the 90% band excludes zero, and a profile-wise
studentized sup test p-value (same multiplier-bootstrap statistic as the band).

Outputs (results/pacce/)
------------------------
ctate_curves.parquet, itte_centers.parquet, ctate_itte_results_summary.csv,
ctate_bands.pdf (regenerated, deterministic), itte_prediction_bands.pdf
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.pacce_p0_audit import RES, ensure_sas_dir
from experiments.pacce_p3_topology import (
    BASELINE_COLS,
    PROFILES,
    _read_frames,
    design_row,
    fit_design,
    run_tate,
)
from tcda_uq.datasets.pacce import load_outcome_spec, load_pacce_from_frames
from tcda_uq.estimators import CTATEDRLearner
from tcda_uq.uq.asymptotic import ctate_confidence_band
from tcda_uq.uq.conformal import ITTEConformal
from sklearn.linear_model import LogisticRegression

from tcda_uq.uq.asymptotic.multiplier_bootstrap import _draw_multipliers

ALPHA = 0.10
SEED = 20240


def profile_sup_test(learner, xrow, d, n_boot, rng):
    """Studentized sup test of H0: tau_d(., x) = 0 (same statistic as the band)."""
    a = learner.weights(xrow, d=d)
    R = learner.residuals(d)
    center = learner.predict_dim(np.atleast_2d(xrow), d)[0]
    contrib = a[:, None] * R
    var = np.einsum("it,it->t", contrib, contrib)
    scale = np.sqrt(np.maximum(var, 1e-24))
    stat = float(np.max(np.abs(center) / scale))
    xi = _draw_multipliers("rademacher", (n_boot, a.shape[0]),
                           np.random.default_rng(rng))
    boot_sup = np.max(np.abs((xi @ contrib) / scale), axis=1)
    return stat, float(np.mean(boot_sup >= stat))


def main():
    sas = ensure_sas_dir()
    spec = load_outcome_spec("data/pacce/outcome_spec.yaml")
    demo, lesion = _read_frames(sas)
    data = load_pacce_from_frames(demo, lesion, spec)
    phi, tseq, A = data.phi, data.tseq, data.A
    X_raw = data.X[BASELINE_COLS].to_numpy(float)
    Xf, keep = fit_design(X_raw)
    print(f"n eligible = {len(A)}")

    cfr, _, _ = run_tate(phi[:, [0], :], A, Xf, tseq, seed=SEED)
    learner = CTATEDRLearner(n_basis=4).fit((phi[:, [0], :], A, Xf), tseq,
                                            cross_fit_result=cfr)

    ctate_rows, curve_rows = [], []
    for j, (pname, prow) in enumerate(PROFILES.items()):
        xrow = design_row([prow[c] for c in BASELINE_COLS], keep)
        cb = ctate_confidence_band(learner, xrow, d=0, alpha=ALPHA,
                                   n_boot=1999, rng=910000 + j)
        stat, pval = profile_sup_test(learner, xrow, 0, 1999, 920000 + j)
        support = int(((np.abs(X_raw - np.array([prow[c] for c in BASELINE_COLS]))
                        [:, [0, 4]] == 0).all(axis=1)).sum())
        exc = float(np.mean((cb.lower > 0) | (cb.upper < 0)))
        peak = float(np.max(np.abs(cb.center)))
        ctate_rows.append(dict(
            profile=pname, burden=prow["n_target_bl"],
            chemo_iri=prow["chemo_iri"], n_matching=support,
            peak_abs_tau=peak, t_at_peak=float(tseq[int(np.argmax(np.abs(cb.center)))]),
            mean_halfwidth=float(np.mean(cb.upper - cb.lower)) / 2.0,
            excursion_frac=exc, sup_stat=stat, sup_pvalue=pval,
            bonferroni_p=min(1.0, pval * len(PROFILES))))
        for t, c, lo, hi in zip(tseq, cb.center, cb.lower, cb.upper):
            curve_rows.append(dict(profile=pname, t=t, center=c,
                                   lower=lo, upper=hi))
    ct = pd.DataFrame(ctate_rows)
    ct.to_csv(RES / "ctate_itte_results_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_parquet(RES / "ctate_curves.parquet", index=False)
    print("== CTATE ==")
    print(ct.round(4).to_string(index=False))

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for axi, (pname, cur) in zip(
            axes.ravel(),
            pd.DataFrame(curve_rows).groupby("profile", sort=False)):
        cur = cur.sort_values("t")
        axi.fill_between(cur["t"], cur["lower"], cur["upper"], alpha=0.25)
        axi.plot(cur["t"], cur["center"], lw=2)
        axi.axhline(0, color="k", lw=0.8, ls="--")
        axi.set_title(pname, fontsize=9)
    fig.suptitle("CTATE profiles (90% simultaneous-in-t bands)")
    fig.tight_layout(); fig.savefig(RES / "ctate_bands.pdf"); plt.close(fig)

    model = ITTEConformal.fit((phi[:, [0], :], A, Xf), tseq, calib_frac=0.5,
                              n_basis=4, propensity_estimator=LogisticRegression(
                                  C=1.0, max_iter=2000),
                              propensity_feature_fn=lambda z: z[:, :1],
                              random_state=SEED + 3)
    itte_rows, itte_curve_rows = [], []
    for pname, prow in PROFILES.items():
        xrow = design_row([prow[c] for c in BASELINE_COLS], keep)
        lo, hi, ctr = model.band_bounds(xrow, ALPHA)
        lo, hi, ctr = lo[0], hi[0], ctr[0]
        itte_rows.append(dict(
            profile=pname,
            width_mean=float(np.mean(hi - lo)),
            width_median=float(np.median(hi - lo)),
            peak_abs_center=float(np.max(np.abs(ctr))),
            zero_outside_frac=float(np.mean((lo > 0) | (hi < 0))),
            unbounded=bool(np.isinf(hi - lo).any())))
        for t, c, l, h in zip(tseq, ctr, lo, hi):
            itte_curve_rows.append(dict(profile=pname, t=t, center=c,
                                        lower=l, upper=h))
    it = pd.DataFrame(itte_rows)
    print("== ITTE ==")
    print(it.round(4).to_string(index=False))
    pd.DataFrame(itte_curve_rows).to_parquet(
        RES / "itte_centers.parquet", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for axi, (pname, cur) in zip(
            axes.ravel(),
            pd.DataFrame(itte_curve_rows).groupby("profile", sort=False)):
        cur = cur.sort_values("t")
        axi.fill_between(cur["t"], cur["lower"], cur["upper"], alpha=0.25,
                         color="#ee7733")
        axi.plot(cur["t"], cur["center"], color="#ee7733", lw=2)
        axi.axhline(0, color="k", lw=0.8, ls="--")
        axi.set_title(pname, fontsize=9)
    fig.suptitle("ITTE prediction bands for a new patient at pre-specified profiles")
    fig.tight_layout(); fig.savefig(RES / "itte_prediction_bands.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
