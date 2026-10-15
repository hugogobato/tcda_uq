"""Phase 6.3 -- power-parameter ``r`` sensitivity of the ITTE conformal band.

The power exponent ``r`` in the weighted silhouette ``phi(t; D, r)`` controls *which*
homological features the summary emphasizes (large ``r`` up-weights long-lived, high-
persistence features; TATE Remark 1). It is a tuning dimension with **no scalar-CP analogue**:
moving ``r`` reshapes the whole score distribution and hence the band. This experiment measures
coverage and width of the ITTE prediction band as a function of ``r`` on the **topological**
(Level-B) DGP, where ``r`` genuinely acts through real persistence diagrams, and asks the
practical question: is there an ``r`` that **minimizes band width at fixed coverage**?

**Efficiency.** Persistent homology is computed **once** per unit (cached on the sample); each
``r`` only re-runs the cheap power-weighting + silhouette + conformal fit on the *same*
diagrams (:meth:`TopologicalCausalSimulation.silhouettes_at_r`). So the ``r`` sweep costs one
homology pass plus a handful of silhouette/regression passes.

The band uses **overlap-stabilized weights** (Phase 6.5) so the ``r`` effect is not confounded
by weak-overlap ``+inf`` atoms. Reported per (``r``, homology dim): coverage of a fresh oracle
ITTE, mean finite width, inf%, and the signal/noise proxy (mean ``|ITTE|`` vs aleatoric sd).

Runs on CPU (default 8 workers). Example::

    .venv/bin/python -u experiments/phase6_r_study.py \
        --rs 1 2 3 4 5 --reps 80 --n 250 --m-test 120 --workers 8 \
        --output-dir experiments/out/phase6
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
    n, m_test, seed, alpha, rs, n_basis, sim_cfg = task
    from tcda_uq.datasets import TopologicalCausalSimulation
    from tcda_uq.uq.conformal import ITTEConformal, make_weight_fn
    from sklearn.linear_model import LogisticRegression

    sim = TopologicalCausalSimulation(seed=0, **sim_cfg)
    rng = np.random.default_rng(seed)
    train = sim.sample(n, rng=rng)          # homology computed once, diagrams cached
    test = sim.sample(m_test, rng=rng)
    wfn = make_weight_fn("overlap")

    out = {}
    for r in rs:
        po_tr, _ = sim.silhouettes_at_r(train, r=r)
        phi_obs = po_tr[np.arange(n), train.A]                 # factual silhouettes at r
        po_te, itte_te = sim.silhouettes_at_r(test, r=r)
        model = ITTEConformal.fit(
            (phi_obs, train.A, train.X), sim.tseq, n_basis=n_basis,
            propensity_estimator=LogisticRegression(max_iter=500),
            propensity_feature_fn=_interactions, weight_fn=wfn, random_state=0,
        )
        per = {}
        for d in range(sim.n_hom_dim):
            lo, hi, _ = model.band_bounds(test.X, alpha, d=d)
            tgt = itte_te[:, d, :]
            fin = np.isfinite(hi).all(1) & np.isfinite(lo).all(1)
            cov = ((tgt >= lo) & (tgt <= hi)).all(1)
            cov = np.where(fin, cov, True)
            width = np.where(fin, (hi - lo).mean(1), np.nan)
            per[d] = dict(
                cov=float(np.mean(cov)),
                inf=float(np.mean(~fin)),
                width=float(np.nanmean(width)) if fin.any() else np.nan,
                signal=float(np.abs(tgt).mean()),
                noise=float(tgt.std(0).mean()),
            )
        out[r] = per
    return out


def run(args):
    from multiprocessing import Pool
    sim_cfg = dict(resolution=args.resolution, pts_per_loop=args.pts_per_loop,
                   noise=args.noise, prop_scale=args.overlap)
    seeds = list(np.random.SeedSequence(args.seed).spawn(args.reps))
    tasks = [(args.n, args.m_test, s, args.alpha, args.rs, args.n_basis, sim_cfg) for s in seeds]

    print(f"Phase 6.3 power-parameter r study | target={1-args.alpha:.2f} | n={args.n} "
          f"| overlap={args.overlap} | reps={args.reps} | m_test={args.m_test} "
          f"| topological (Level-B) DGP, overlap-stabilized band", flush=True)
    print(f"{'r':>4} {'dim':>3} {'cov':>7} {'inf%':>6} {'width':>8} {'signal':>8} "
          f"{'noise':>8} {'snr':>6}", flush=True)

    rows, t0 = [], time.time()
    step = max(1, len(tasks) // 10)
    with Pool(processes=args.workers, initializer=_limit_threads) as pool:
        for i, r in enumerate(pool.imap_unordered(_rep, tasks, chunksize=1), 1):
            rows.append(r)
            if i % step == 0:
                print(f"#   {i}/{len(tasks)} reps ({time.time()-t0:.0f}s)", flush=True)

    summary = []
    n_hom = 2
    for r in args.rs:
        for d in range(n_hom):
            vs = [row[r][d] for row in rows]
            cov = np.mean([x["cov"] for x in vs])
            inf = 100 * np.mean([x["inf"] for x in vs])
            width = np.nanmean([x["width"] for x in vs])
            sig = np.mean([x["signal"] for x in vs])
            noi = np.mean([x["noise"] for x in vs])
            snr = sig / noi if noi > 0 else np.nan
            print(f"{r:>4.1f} {d:>3} {cov:>7.3f} {inf:>6.1f} {width:>8.4f} {sig:>8.4f} "
                  f"{noi:>8.4f} {snr:>6.2f}", flush=True)
            summary.append((r, d, cov, inf, width, snr))
        print(flush=True)

    print("Width-minimizing r at coverage >= target (per dim):", flush=True)
    for d in range(n_hom):
        elig = [(r, w) for (r, dd, cov, inf, w, snr) in summary
                if dd == d and cov >= 1 - args.alpha and np.isfinite(w)]
        if elig:
            r_best, w_best = min(elig, key=lambda z: z[1])
            print(f"  dim {d}: r* = {r_best}  (width {w_best:.4f})", flush=True)
        else:
            print(f"  dim {d}: no r meets coverage target", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rs", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0])
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--m-test", type=int, default=120)
    ap.add_argument("--reps", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-basis", type=int, default=5)
    ap.add_argument("--resolution", type=int, default=60)
    ap.add_argument("--pts-per-loop", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.08)
    ap.add_argument("--overlap", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260703)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    def _go():
        run(args)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "r_study.txt"
        with out.open("w") as f, contextlib.redirect_stdout(f):
            _go()
        print(out.read_text())
    else:
        _go()


if __name__ == "__main__":
    main()
