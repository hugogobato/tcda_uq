"""Phase 3.4 -- CTATE confidence-band coverage & recovery study.

Monte-Carlo evaluation of the functional DR-learner CTATE estimator and its
pointwise-in-x confidence band tau_d(., x), against the tri-oracle truth
``sim.true_ctate(x)``, swept over sample size. For a set of fixed held-out
covariate values ``x`` (the two subgroup means and their midpoint) we report,
per homology dim:

  * recovery       -- rel-L1 of tau_hat_d(., x) vs the oracle tau_d(., x) (bias+var);
  * cov(simult)    -- simultaneous-over-t coverage of the true tau_d(., x);
  * iw_err         -- interval-wise error (fraction of the domain excluded);
  * mean_width     -- mean band width.

Two bands: the multiplier-bootstrap **confidence** band (simultaneous in t) and a
**pointwise** z baseline (not simultaneous -> expected to undercover), mirroring
the Phase-2 story. Nuisances are well specified and the tri-oracle tau is linear
in x, so the linear second stage is well specified and the simultaneous band
should be ~nominal while recovery tightens as n grows.

Run (from repo root, in the venv):
    .venv/bin/python -u experiments/phase3_ctate_coverage.py --reps 200 --alpha 0.10

Sharding (parallelize reps across processes; combine with the Phase-2 aggregator):
    ... --num-shards 4 --shard-id 0 --output out/p3_s0.txt
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
from tcda_uq.estimators import CTATEDRLearner
from tcda_uq.metrics import Band, covers_simultaneous, interval_wise_error, mean_width
from tcda_uq.uq.asymptotic import ctate_confidence_band, ctate_pointwise_sd


def _interactions(X):
    """True propensity features: [X, x1*x2, x0*x2] -> well-specified logistic."""
    extra = np.column_stack([X[:, 1] * X[:, 2], X[:, 0] * X[:, 2]])
    return np.hstack([X, extra])


def _eval_points(sim):
    """Fixed held-out covariate values at which to band tau_d(., x)."""
    mid = 0.5 * (sim.mu1 + sim.mu2)
    return {"mu1": sim.mu1, "mid": mid, "mu2": sim.mu2}


def _pointwise_band(center, sd, tseq, alpha):
    half = norm.ppf(1 - alpha / 2) * sd
    return Band(tseq, center - half, center + half, center=center,
               level=1 - alpha, kind="confidence")


def run(sim, n, reps, alpha, n_boot, rng0, *,
        rep_start=0, rep_stop=None, total_reps=None):
    pts = _eval_points(sim)
    n_hom = sim.n_hom_dim
    methods = ["pointwise", "multiplier"]
    # accumulators keyed (point, method) -> [n_hom]
    rec = {p: np.zeros(n_hom) for p in pts}
    cov = {(p, m): np.zeros(n_hom) for p in pts for m in methods}
    iwe = {(p, m): np.zeros(n_hom) for p in pts for m in methods}
    wid = {(p, m): np.zeros(n_hom) for p in pts for m in methods}

    if rep_stop is None:
        rep_stop = reps
    if total_reps is None:
        total_reps = reps
    if not 0 <= rep_start < rep_stop <= total_reps:
        raise ValueError("rep shard must satisfy 0 <= rep_start < rep_stop <= total_reps")
    n_local = rep_stop - rep_start

    truth = {p: sim.true_ctate(x[None])[0] for p, x in pts.items()}  # [hom,res] each
    seeds = np.random.SeedSequence(rng0).spawn(total_reps)
    for r in range(rep_start, rep_stop):
        rng = np.random.default_rng(seeds[r])
        s = sim.sample(n, rng=rng)
        learner = CTATEDRLearner(n_basis=sim.n_basis).fit(
            s.observed, sim.tseq, n_splits=2,
            propensity_estimator=LogisticRegression(max_iter=500),
            propensity_feature_fn=_interactions,
            random_state=int(rng.integers(1 << 31)),
        )
        seed_b = int(rng.integers(1 << 31))
        for p, x in pts.items():
            tau_hat = learner.predict(x)                    # [hom, res]
            for d in range(n_hom):
                rel = (np.mean(np.abs(tau_hat[d] - truth[p][d]))
                       / (np.mean(np.abs(truth[p][d])) + 1e-9))
                rec[p][d] += rel
                sd = ctate_pointwise_sd(learner, x, d=d)
                bands = {
                    "pointwise": _pointwise_band(tau_hat[d], sd, sim.tseq, alpha),
                    "multiplier": ctate_confidence_band(
                        learner, x, d, alpha=alpha, n_boot=n_boot, rng=seed_b),
                }
                for m, b in bands.items():
                    cov[(p, m)][d] += float(
                        covers_simultaneous(b.lower, b.upper, truth[p][d]))
                    iwe[(p, m)][d] += interval_wise_error(b.lower, b.upper, truth[p][d])
                    wid[(p, m)][d] += mean_width(b.lower, b.upper)

    for p in pts:
        rec[p] /= n_local
        for m in methods:
            cov[(p, m)] /= n_local
            iwe[(p, m)] /= n_local
            wid[(p, m)] /= n_local
    return list(pts), methods, rec, cov, iwe, wid


def _emit(n, points, methods, rec, cov, iwe, wid, n_hom):
    for p in points:
        for d in range(n_hom):
            for m in methods:
                print(f"{n:>5} {p:>5} {m:>11} {d:>3} {rec[p][d]:>9.3f} "
                      f"{cov[(p, m)][d]:>12.3f} {iwe[(p, m)][d]:>8.3f} "
                      f"{wid[(p, m)][d]:>11.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[200, 500, 1000])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--n-boot", type=int, default=1500)
    ap.add_argument("--resolution", type=int, default=50)
    ap.add_argument("--n-basis", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260701)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")
    base, extra = divmod(args.reps, args.num_shards)
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
    sim = TriOracleSimulation(n_hom_dim=2, resolution=args.resolution,
                              n_basis=args.n_basis, noise_scale=0.3, seed=0)
    shard = (f" | shard={args.shard_id}/{args.num_shards} "
             f"| reps={rep_stop-rep_start}/{args.reps} | range={rep_start}:{rep_stop}"
             if args.num_shards > 1 else f" | reps={args.reps}")
    print(f"Phase 3.4 CTATE coverage | target={1-args.alpha:.2f}{shard}", flush=True)
    print(f"{'n':>5} {'x':>5} {'method':>11} {'dim':>3} {'recovery':>9} "
          f"{'cov(sim)':>12} {'iw_err':>8} {'mean_width':>11}", flush=True)
    for n in args.ns:
        t0 = time.time()
        points, methods, rec, cov, iwe, wid = run(
            sim, n, args.reps, args.alpha, args.n_boot, args.seed + n,
            rep_start=rep_start, rep_stop=rep_stop, total_reps=args.reps)
        _emit(n, points, methods, rec, cov, iwe, wid, sim.n_hom_dim)
        print(f"# n={n} done in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
