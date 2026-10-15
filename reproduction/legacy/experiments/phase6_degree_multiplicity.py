"""Phase 6.4 -- simultaneous ITTE bands across homology degree (H0/H1/H2).

A topological effect lives across several degrees at once (components, loops, voids). The
Phase-4 band is valid for each degree *separately*; a statement about the joint vector
``(delta_0, delta_1, delta_2)`` needs a multiplicity correction over degree. This experiment
compares three schemes (see :mod:`tcda_uq.uq.conformal.degree_multiplicity`):

  * ``bonferroni`` -- band each dim at ``alpha / n_dim`` and intersect (union bound);
  * ``sidak``      -- band each dim at ``1 - (1-alpha)^(1/n_dim)``;
  * ``joint``      -- fold the degree axis into the sup-norm: one per-arm score maxes over
    ``(d, t)``, so a single radius controls all degrees with **no** Bonferroni-over-``d``
    penalty (the topology-appropriate sharper scheme).

All three must achieve *joint* coverage (all degrees simultaneously) ``>= 1 - alpha``; the
question is the **total width** it costs. The band uses overlap-stabilized weights (Phase 6.5)
so the multiplicity comparison is not confounded by the weak-overlap ``+inf`` atom; an optional
``--overlap`` sweep shows how the per-dim schemes' level splitting reintroduces unbounded bands
under thin overlap while the joint scheme resists it.

Reported per (scheme): joint coverage (all dims), per-dim marginal coverage, per-dim + total
mean finite width, and joint inf% (any dim unbounded).

Runs on CPU (default 8 workers). Example::

    .venv/bin/python -u experiments/phase6_degree_multiplicity.py \
        --overlaps 1.0 2.5 --reps 200 --workers 8 --output-dir experiments/out/phase6
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import time

import numpy as np


def _limit_threads():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


_limit_threads()


def _interactions(X):
    X = np.asarray(X)
    return np.hstack([X, np.column_stack([X[:, 1] * X[:, 2], X[:, 0] * X[:, 2]])])


def _rep(task):
    n, m_test, seed, alpha, overlap, n_hom, cfg = task
    from sklearn.linear_model import LogisticRegression
    from tcda_uq.datasets import TriOracleSimulation
    from tcda_uq.uq.conformal import ITTEConformal, make_weight_fn, simultaneous_itte_bounds

    sim = TriOracleSimulation(n_hom_dim=n_hom, resolution=cfg["resolution"],
                              n_basis=cfg["n_basis"], noise_scale=cfg["noise"],
                              prop_scale=overlap, seed=0)
    rng = np.random.default_rng(seed)
    s = sim.sample(n, rng=rng)
    test = sim.sample(m_test, rng=rng)
    model = ITTEConformal.fit(
        s.observed, sim.tseq, n_basis=cfg["n_basis"],
        propensity_estimator=LogisticRegression(max_iter=500),
        propensity_feature_fn=_interactions, weight_fn=make_weight_fn("overlap"),
        random_state=0,
    )

    out = {}
    for scheme in ("joint", "bonferroni", "sidak"):
        bounds = simultaneous_itte_bounds(model, test.X, alpha, scheme=scheme)
        joint_cov = np.ones(test.X.shape[0], dtype=bool)
        joint_fin = np.ones(test.X.shape[0], dtype=bool)
        per_dim = {}
        for d, (lo, hi, _) in bounds.items():
            tgt = test.oracle_itte[:, d, :]
            fin = np.isfinite(hi).all(1) & np.isfinite(lo).all(1)
            cov = ((tgt >= lo) & (tgt <= hi)).all(1)
            cov = np.where(fin, cov, True)
            joint_cov &= cov
            joint_fin &= fin
            width = np.where(fin, (hi - lo).mean(1), np.nan)
            per_dim[d] = dict(cov=float(np.mean(cov)),
                              width=float(np.nanmean(width)) if fin.any() else np.nan)
        # total width across dims computed only on units finite in *every* dim
        total_w = 0.0
        ok = joint_fin.any()
        for d, (lo, hi, _) in bounds.items():
            w = (hi - lo).mean(1)
            total_w += float(np.mean(w[joint_fin])) if ok else np.nan
        out[scheme] = dict(joint_cov=float(np.mean(joint_cov)),
                           joint_inf=float(np.mean(~joint_fin)),
                           total_width=total_w, per_dim=per_dim)
    return out


def run(args):
    from multiprocessing import Pool
    cfg = dict(resolution=args.resolution, n_basis=args.n_basis, noise=args.noise)
    print(f"Phase 6.4 degree multiplicity | target={1-args.alpha:.2f} | n={args.n} "
          f"| n_hom={args.n_hom} | reps={args.reps} | m_test={args.m_test} "
          f"| overlap-stabilized band", flush=True)

    for overlap in args.overlaps:
        seeds = list(np.random.SeedSequence(args.seed).spawn(args.reps))
        tasks = [(args.n, args.m_test, s, args.alpha, overlap, args.n_hom, cfg) for s in seeds]
        rows, t0 = [], time.time()
        step = max(1, len(tasks) // 8)
        with Pool(processes=args.workers, initializer=_limit_threads) as pool:
            for i, r in enumerate(pool.imap_unordered(_rep, tasks, chunksize=1), 1):
                rows.append(r)
                if i % step == 0:
                    print(f"#   overlap {overlap}: {i}/{len(tasks)} ({time.time()-t0:.0f}s)",
                          flush=True)
        print(f"\n-- overlap = {overlap} --", flush=True)
        print(f"{'scheme':>11} {'joint_cov':>10} {'joint_inf%':>11} {'total_width':>12} "
              f"{'per-dim cov':>28} {'per-dim width':>28}", flush=True)
        for scheme in ("joint", "bonferroni", "sidak"):
            vs = [row[scheme] for row in rows]
            jc = np.mean([x["joint_cov"] for x in vs])
            ji = 100 * np.mean([x["joint_inf"] for x in vs])
            tw = np.nanmean([x["total_width"] for x in vs])
            covs = [np.mean([x["per_dim"][d]["cov"] for x in vs]) for d in range(args.n_hom)]
            wids = [np.nanmean([x["per_dim"][d]["width"] for x in vs]) for d in range(args.n_hom)]
            covs_s = " ".join(f"{c:.3f}" for c in covs)
            wids_s = " ".join(f"{w:.3f}" for w in wids)
            print(f"{scheme:>11} {jc:>10.3f} {ji:>11.1f} {tw:>12.3f} "
                  f"{covs_s:>28} {wids_s:>28}", flush=True)
        print(flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--m-test", type=int, default=200)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-hom", type=int, default=3, help="number of homology dims (H0..H2)")
    ap.add_argument("--n-basis", type=int, default=5)
    ap.add_argument("--resolution", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--overlaps", type=float, nargs="+", default=[1.0, 2.5])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260704)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    def _go():
        run(args)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "degree_multiplicity.txt"
        with out.open("w") as f, contextlib.redirect_stdout(f):
            _go()
        print(out.read_text())
    else:
        _go()


if __name__ == "__main__":
    main()
