#!/usr/bin/env python
"""Run small finite-grid Monte Carlo diagnostics for the public API.

The script estimates gridwise inclusion frequencies and width diagnostics for
TATE and fixed-x CTATE mean bands, plus ITTE prediction bands under full,
overlap, and matching covariate targets. For ITTE it supplies the simulator's
known propensity and evaluates each model on fresh draws from the same target.
The results are design-specific numerical diagnostics, not proof of theorem
assumptions or continuous-interval coverage.

Examples::

    python scripts/reproduce_coverage.py --quick
    python scripts/reproduce_coverage.py --reps 40 --out coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

from tcda_uq import ctate_band, evaluate_coverage, tate_band
from tcda_uq.datasets import TriOracleSimulation
from tcda_uq.estimators import CTATEDRLearner, cross_fit
from tcda_uq.uq.conformal import ITTEConformal


def _propensity_learner():
    """Learned propensity used for the mean-estimation diagnostic."""
    return LogisticRegression(max_iter=500)


def tate_row(sim, truth, n, reps, alpha, seed0):
    """Monte Carlo grid inclusion and mean width for the TATE band."""
    coverages, widths = [], []
    for r in range(reps):
        sample = sim.sample(n, rng=seed0 + r)
        fit = cross_fit(
            sample.observed,
            sim.tseq,
            n_basis=5,
            n_splits=2,
            random_state=r,
            stratify=False,
            propensity_estimator=_propensity_learner(),
        )
        band = tate_band(
            fit, method="mbb", d=0, alpha=alpha, n_boot=300, rng=r
        )
        coverages.append(evaluate_coverage(band, truth).coverage)
        widths.append(band.mean_width())
    return float(np.mean(coverages)), float(np.mean(widths))


def ctate_row(sim, n, reps, alpha, seed0):
    """Monte Carlo grid inclusion for the fixed-x CTATE mean band."""
    x = sim.EX + 0.3
    truth = sim.true_ctate(x[None, :])[0, 0]
    coverages, widths = [], []
    for r in range(reps):
        sample = sim.sample(n, rng=seed0 + r)
        learner = CTATEDRLearner(n_basis=5).fit(
            sample.observed,
            sim.tseq,
            n_splits=2,
            random_state=r,
            stratify=False,
            propensity_estimator=_propensity_learner(),
        )
        band = ctate_band(
            learner, x, d=0, alpha=alpha, n_boot=300, rng=r
        )
        coverages.append(evaluate_coverage(band, truth).coverage)
        widths.append(band.mean_width())
    return float(np.mean(coverages)), float(np.mean(widths))


def itte_rows(sim, n_train, n_test, reps, alpha, seed0):
    """Target-matched ITTE diagnostics with the known generating propensity."""
    targets = ("full", "overlap", "matching")
    scores = {target: {"coverage": [], "finite": [], "finite_width": []}
              for target in targets}

    for r in range(reps):
        train = sim.sample(n_train, rng=seed0 + 10 * r)
        for j, target in enumerate(targets):
            test = sim.sample_target(
                n_test, target=target, rng=seed0 + 10 * r + 1 + j
            )
            model = ITTEConformal.fit(
                train.observed,
                sim.tseq,
                n_basis=5,
                target=target,
                propensity_fn=sim.propensity_score,
                weights_known=True,
                random_state=seed0 + 10 * r + 5 + j,
            )
            lower, upper, _ = model.band_bounds(test.X, alpha=alpha, d=0)
            result = evaluate_coverage(
                (lower, upper), test.oracle_itte[:, 0, :], level=1.0 - alpha
            )
            finite = np.isfinite(upper - lower).all(axis=1)
            scores[target]["coverage"].append(result.coverage)
            scores[target]["finite"].append(float(finite.mean()))
            if finite.any():
                per_unit_width = (upper - lower).mean(axis=1)
                scores[target]["finite_width"].append(
                    float(per_unit_width[finite].mean())
                )

    rows = {}
    for target, values in scores.items():
        widths = values["finite_width"]
        rows[target] = (
            float(np.mean(values["coverage"])),
            float(np.mean(widths)) if widths else float("nan"),
            float(np.mean(values["finite"])),
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=25,
                        help="independent fitted samples per setting")
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 500],
                        help="sample sizes for mean-band diagnostics")
    parser.add_argument("--resolution", type=int, default=30,
                        help="fixed filtration-grid size")
    parser.add_argument("--alpha", type=float, default=0.10,
                        help="nominal error level")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed")
    parser.add_argument("--quick", action="store_true",
                        help="small tutorial-sized diagnostic")
    parser.add_argument("--out", type=str, default=None,
                        help="optional CSV output path")
    args = parser.parse_args()

    if args.quick:
        args.reps, args.sizes, args.resolution = 3, [200], 20

    rows = []
    start = time.time()
    for n in args.sizes:
        sim = TriOracleSimulation(
            n_hom_dim=1,
            resolution=args.resolution,
            n_basis=5,
            noise_scale=0.3,
            seed=args.seed,
        )
        tate_cov, tate_width = tate_row(
            sim, sim.true_tate()[0], n, args.reps, args.alpha, args.seed + 5000
        )
        rows.append((n, "full", "TATE Gaussian mean band", tate_cov,
                     tate_width, 1.0))

        ctate_cov, ctate_width = ctate_row(
            sim, n, args.reps, args.alpha, args.seed + 6000
        )
        rows.append((n, "fixed x", "CTATE mean band", ctate_cov,
                     ctate_width, 1.0))

        for target, values in itte_rows(
            sim,
            n_train=max(2 * n, 400),
            n_test=2 * n,
            reps=args.reps,
            alpha=args.alpha,
            seed0=args.seed + 7000,
        ).items():
            coverage, finite_width, finite_share = values
            rows.append((n, target, "ITTE prediction", coverage,
                         finite_width, finite_share))

    print("Finite-grid Monte Carlo diagnostics")
    print("ITTE widths are averaged over finite bands; all ITTE test draws match the declared target.")
    print("Estimated mean-band nuisances and finite simulation do not certify theorem assumptions.\n")
    print("| n | target | estimand / object | grid inclusion | mean width | finite share |")
    print("|---:|:---|:---|---:|---:|---:|")
    for n, target, label, inclusion, width, finite in rows:
        width_text = "nan" if not np.isfinite(width) else f"{width:.4g}"
        print(f"| {n} | {target} | {label} | {inclusion:.3f} | {width_text} | {finite:.3f} |")
    print(f"\nFinished in {time.time() - start:.1f}s.")

    if args.out:
        with open(args.out, "w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["n", "target", "estimand", "grid_inclusion",
                             "mean_width_finite", "finite_band_share"])
            writer.writerows(rows)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
