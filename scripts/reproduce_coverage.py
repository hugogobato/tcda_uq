#!/usr/bin/env python
"""Reproduce the library's core coverage/width claims on the tri-oracle harness.

A small, self-contained, CPU-only driver that regenerates the headline
estimand x uncertainty numbers this library is built to deliver:

  * TATE  psi_d(t)     -- simultaneous **confidence** bands (multiplier bootstrap,
                          Liebl-Reimherr) at/above nominal; Pini-Vantini controls
                          the **interval-wise** error at a narrower width.
  * CTATE tau_d(t, x)  -- pointwise-in-x **confidence** band at a held-out x.
  * ITTE  delta_d(t)   -- overlap-stabilized conformal **prediction** band covering
                          a fresh individual draw, with the weak-overlap +inf atom
                          suppressed (finite-band share ~ 1).

This is the *library-level* reproducibility artifact (fast, a few minutes). The
full production tables and figures live in the research workspace's
``experiments/`` scripts, which are intentionally not shipped with the library.

Everything routes through the public/unified API::

    from tcda_uq import tate_band, ctate_band, itte_band, evaluate_coverage

Usage::

    python scripts/reproduce_coverage.py                 # default (~few min)
    python scripts/reproduce_coverage.py --quick         # fast smoke
    python scripts/reproduce_coverage.py --reps 40 --out coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from tcda_uq import ctate_band, evaluate_coverage, itte_band, tate_band
from tcda_uq.datasets import TriOracleSimulation
from tcda_uq.estimators import CTATEDRLearner, cross_fit
from tcda_uq.metrics import interval_wise_error
from tcda_uq.uq.conformal import ITTEConformal, make_weight_fn


def _prop():
    """A fast propensity learner (AIPW/weighted-CP stay valid via double robustness)."""
    return LogisticRegression(max_iter=500)


def tate_row(sim, truth, n, reps, alpha, seed0):
    """Simultaneous coverage/width for the three TATE confidence bands over reps."""
    mbb_c, lr_c, pv_iwe, mbb_w, lr_w, pv_w = ([] for _ in range(6))
    for r in range(reps):
        s = sim.sample(n, rng=seed0 + r)
        fit = cross_fit(s.observed, sim.tseq, n_basis=5, n_splits=2,
                        random_state=r, propensity_estimator=_prop())
        b_mbb = tate_band(fit, method="mbb", d=0, alpha=alpha, n_boot=400, rng=r)
        b_lr = tate_band(fit, method="lr", d=0, alpha=alpha, backend="python")
        b_pv = tate_band(fit, method="pv", d=0, alpha=alpha, n_boot=400, rng=r)
        mbb_c.append(evaluate_coverage(b_mbb, truth).coverage)
        lr_c.append(evaluate_coverage(b_lr, truth).coverage)
        pv_iwe.append(interval_wise_error(b_pv.lower, b_pv.upper, truth))
        mbb_w.append(b_mbb.mean_width())
        lr_w.append(b_lr.mean_width())
        pv_w.append(b_pv.mean_width())
    return {
        "TATE mbb (simult)": (np.mean(mbb_c), np.mean(mbb_w)),
        "TATE LR (simult)": (np.mean(lr_c), np.mean(lr_w)),
        "TATE PV (interval-wise err)": (np.mean(pv_iwe), np.mean(pv_w)),
    }


def ctate_row(sim, n, reps, alpha, seed0):
    """Simultaneous-in-t coverage/width for the CTATE confidence band at a held-out x."""
    x = sim.EX + 0.3
    truth = sim.true_ctate(x[None, :])[0, 0]
    cov, wid = [], []
    for r in range(reps):
        s = sim.sample(n, rng=seed0 + r)
        learner = CTATEDRLearner(n_basis=5).fit(
            s.observed, sim.tseq, n_splits=2, random_state=r, propensity_estimator=_prop()
        )
        band = ctate_band(learner, x, d=0, alpha=alpha, n_boot=400, rng=r)
        cov.append(evaluate_coverage(band, truth).coverage)
        wid.append(band.mean_width())
    return {"CTATE conf @x (simult)": (np.mean(cov), np.mean(wid))}


def itte_row(sim, n_train, n_test, alpha, seed0):
    """Prediction coverage/width + finite-band share for the overlap-stabilized ITTE band."""
    train = sim.sample(n_train, rng=seed0)
    test = sim.sample(n_test, rng=seed0 + 1)
    model = ITTEConformal.fit(
        train.observed, sim.tseq, n_basis=5, weight_fn=make_weight_fn("overlap"),
        random_state=0, propensity_estimator=_prop(),
    )
    lo, hi, _ = model.band_bounds(test.X, alpha=alpha, d=0)
    target = test.oracle_itte[:, 0, :]
    res = evaluate_coverage((lo, hi), target, level=1 - alpha)
    finite = np.isfinite(hi - lo).all(axis=1).mean()
    return {"ITTE overlap (predict)": (res.coverage, res.mean_width, finite)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reps", type=int, default=25, help="replicates per (n, band)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 500],
                    help="sample sizes n for the confidence-band sweep")
    ap.add_argument("--resolution", type=int, default=30, help="silhouette grid size")
    ap.add_argument("--alpha", type=float, default=0.10, help="1 - nominal coverage")
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument("--quick", action="store_true", help="fast smoke preset")
    ap.add_argument("--out", type=str, default=None, help="optional CSV output path")
    args = ap.parse_args()

    if args.quick:
        args.reps, args.sizes, args.resolution = 6, [200], 22

    print(f"tcda_uq coverage reproduction  (alpha={args.alpha}, nominal="
          f"{1 - args.alpha:.2f}, reps={args.reps}, res={args.resolution})\n")

    rows = []  # (n, label, value, width, extra)
    t0 = time.time()
    for n in args.sizes:
        sim = TriOracleSimulation(n_hom_dim=1, resolution=args.resolution,
                                  n_basis=5, noise_scale=0.3, seed=args.seed)
        truth = sim.true_tate()[0]
        results = {}
        results.update(tate_row(sim, truth, n, args.reps, args.alpha, args.seed + 5000))
        results.update(ctate_row(sim, n, args.reps, args.alpha, args.seed + 6000))
        results.update(itte_row(sim, max(2 * n, 400), 2 * n, args.alpha, args.seed + 7000))
        for label, vals in results.items():
            width = vals[1]
            extra = vals[2] if len(vals) > 2 else None
            rows.append((n, label, vals[0], width, extra))

    # markdown table
    print(f"| n | estimand / band | coverage or IWE | mean width | finite share |")
    print(f"|---|---|---|---|---|")
    for n, label, val, width, extra in rows:
        fs = "" if extra is None else f"{extra:.3f}"
        print(f"| {n} | {label} | {val:.3f} | {width:.4g} | {fs} |")
    print(f"\nDone in {time.time() - t0:.1f}s.")
    print("Note: coverage is a frequency over datasets; PV reports interval-wise "
          "error (target <= alpha), not simultaneous coverage.")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "band", "coverage_or_iwe", "mean_width", "finite_share"])
            for n, label, val, width, extra in rows:
                w.writerow([n, label, f"{val:.5f}", f"{width:.6g}",
                            "" if extra is None else f"{extra:.5f}"])
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
