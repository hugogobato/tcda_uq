"""Phase 2.6 -- TATE confidence-band coverage vs width study.

Monte-Carlo coverage of the *marginal* estimand psi_d(t) for the three asymptotic
benchmark bands (all **confidence** bands: they cover the mean, not an individual
curve), against the tri-oracle truth, swept over sample size:

  * multiplier bootstrap  (simultaneous; TATE Cor 5.4)
  * Liebl-Reimherr FFSCB  (simultaneous, fair; roughness-adaptive)
  * Pini-Vantini IWT      (interval-wise)
  * pointwise z           (baseline; not simultaneous -- expected to undercover)

Reports simultaneous coverage (all-t), interval-wise error, and mean width per
homology dim. Nuisances are well specified (Fourier outcome model matched; a
logistic propensity with the true interaction features), so at moderate n the
simultaneous bands should be ~nominal and the pointwise baseline should
undercover -- the motivation for the finite-sample conformal side (Phase 4).

Run (from repo root, in the venv):
    .venv/bin/python -u experiments/phase2_coverage.py --reps 200 --alpha 0.10
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from tcda_uq.datasets import TriOracleSimulation
from tcda_uq.estimators import cross_fit
from tcda_uq.metrics import (
    covers_simultaneous,
    interval_wise_error,
    mean_width,
)
from tcda_uq.uq.asymptotic import (
    multiplier_bootstrap_band,
    liebl_reimherr_band,
    pini_vantini_band,
    eif_covariance,
    r_backend_available,
)


def _interactions(X):
    """True propensity features: [X, x1*x2, x0*x2] -> well-specified logistic."""
    extra = np.column_stack([X[:, 1] * X[:, 2], X[:, 0] * X[:, 2]])
    return np.hstack([X, extra])


def run(
    sim,
    n,
    reps,
    alpha,
    n_boot,
    lr_backend,
    rng0,
    *,
    rep_start=0,
    rep_stop=None,
    total_reps=None,
):
    truth = sim.true_tate()                       # [hom, res]
    n_hom = sim.n_hom_dim
    methods = ["pointwise", "multiplier", "liebl_reimherr", "pini_vantini"]
    cov = {m: np.zeros(n_hom) for m in methods}
    iwe = {m: np.zeros(n_hom) for m in methods}
    wid = {m: np.zeros(n_hom) for m in methods}

    if rep_stop is None:
        rep_stop = reps
    if total_reps is None:
        total_reps = reps
    if not 0 <= rep_start < rep_stop <= total_reps:
        raise ValueError(
            "rep shard must satisfy 0 <= rep_start < rep_stop <= total_reps"
        )
    n_local_reps = rep_stop - rep_start

    seeds = np.random.SeedSequence(rng0).spawn(total_reps)
    for r in range(rep_start, rep_stop):
        rng = np.random.default_rng(seeds[r])
        s = sim.sample(n, rng=rng)
        res = cross_fit(
            s.observed, sim.tseq, n_basis=sim.n_basis, n_splits=2,
            propensity_estimator=LogisticRegression(max_iter=500),
            propensity_feature_fn=_interactions,
            random_state=int(rng.integers(1 << 31)),
        )
        infl = res.influence()
        for d in range(n_hom):
            cov_d = eif_covariance(infl[d])
            var_d = np.diag(cov_d)
            sd = np.sqrt(var_d)
            est = res.aipw[d]
            seed_b = int(rng.integers(1 << 31))

            bands = {
                "pointwise": _pointwise(est, sd, n, sim.tseq, alpha),
                "multiplier": multiplier_bootstrap_band(
                    infl[d], sim.tseq, est, alpha=alpha, n_boot=n_boot,
                    variance=var_d, rng=seed_b),
                "liebl_reimherr": liebl_reimherr_band(
                    infl[d], sim.tseq, est, alpha=alpha, n_int=4, distribution="z",
                    backend=lr_backend, variance=var_d, covariance=cov_d),
                "pini_vantini": pini_vantini_band(
                    infl[d], sim.tseq, est, alpha=alpha, n_boot=n_boot,
                    variance=var_d, rng=seed_b),
            }
            for m, b in bands.items():
                cov[m][d] += float(covers_simultaneous(b.lower, b.upper, truth[d]))
                iwe[m][d] += interval_wise_error(b.lower, b.upper, truth[d])
                wid[m][d] += mean_width(b.lower, b.upper)

    for m in methods:
        cov[m] /= n_local_reps
        iwe[m] /= n_local_reps
        wid[m] /= n_local_reps
    return methods, cov, iwe, wid


def _pointwise(est, sd, n, tseq, alpha):
    from tcda_uq.metrics import Band

    half = norm.ppf(1 - alpha / 2) * sd / np.sqrt(n)
    return Band(tseq, est - half, est + half, center=est, level=1 - alpha,
                kind="confidence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[50, 100, 200, 500, 1000])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-boot", type=int, default=1500)
    ap.add_argument("--resolution", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--lr-backend", default="auto")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")
    base = args.reps // args.num_shards
    extra = args.reps % args.num_shards
    rep_start = args.shard_id * base + min(args.shard_id, extra)
    rep_stop = rep_start + base + int(args.shard_id < extra)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.append else "w"
        with args.output.open(mode) as f, contextlib.redirect_stdout(f):
            _run(args, rep_start, rep_stop)
    else:
        _run(args, rep_start, rep_stop)


def _run(args, rep_start, rep_stop):
    sim = TriOracleSimulation(
        n_hom_dim=2, resolution=args.resolution, n_basis=5, noise_scale=0.3, seed=0
    )
    shard = ""
    if args.num_shards > 1:
        shard = (
            f" | shard={args.shard_id}/{args.num_shards}"
            f" | reps={rep_stop - rep_start}/{args.reps}"
            f" | rep_range={rep_start}:{rep_stop}"
        )
    else:
        shard = f" | reps={args.reps}"
    print(f"Phase 2.6 coverage study | target={1-args.alpha:.2f}{shard} "
          f"| R backend={'yes' if r_backend_available() else 'no (python port)'}", flush=True)
    print(f"{'n':>5} {'method':>15} {'dim':>3} {'cov(simult)':>12} "
          f"{'iw_err':>8} {'mean_width':>11}", flush=True)
    for n in args.ns:
        t0 = time.time()
        methods, cov, iwe, wid = run(
            sim,
            n,
            args.reps,
            args.alpha,
            args.n_boot,
            args.lr_backend,
            args.seed + n,
            rep_start=rep_start,
            rep_stop=rep_stop,
            total_reps=args.reps,
        )
        for m in methods:
            for d in range(sim.n_hom_dim):
                print(f"{n:>5} {m:>15} {d:>3} {cov[m][d]:>12.3f} "
                      f"{iwe[m][d]:>8.3f} {wid[m][d]:>11.4f}", flush=True)
        if args.num_shards > 1:
            print(
                f"# n={n} shard={args.shard_id}/{args.num_shards} "
                f"reps={rep_start}:{rep_stop} total_reps={args.reps} "
                f"done in {time.time()-t0:.1f}s",
                flush=True,
            )
        else:
            print(f"# n={n} done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
