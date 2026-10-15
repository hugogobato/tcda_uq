"""Phase 5 -- covariate-adaptive conformal coverage / width / adaptivity study.

Monte-Carlo evaluation of the Phase-5 covariate-adaptive ITTE prediction bands
against the marginal Phase-4 band, on a *heteroskedastic* tri-oracle DGP where the
aleatoric spread of ``delta`` varies with ``X`` (``--hetero``). This is the regime a
covariate-adaptive band must exploit: a good band should be **narrow where the
outcome is easy and wide where it is hard**, evening out conditional coverage --
while an X-blind marginal band cannot.

Per rep we fit on ``n`` train+calibration units and score a **fresh** test set with
oracle ITTE (both potential outcomes known) *and* the oracle local difficulty
``amp(X) = 1 + hetero * sigmoid(...)`` (``sim.noise_amplitude``). Test units are split
into ``--groups`` equal-count difficulty strata by ``amp(X)`` to probe *approximate
conditional* coverage.

Methods (``--methods``):
  * ``marginal``   -- Phase-4 :class:`ITTEConformal` (X-blind width). Baseline.
  * ``normalized`` -- Phase-5.1 :class:`AdaptiveITTEConformal`, local-scale only
    (mechanism A; exact marginal validity, X-dependent width).
  * ``localized``  -- adaptive with local-scale **and** kernel-localized quantile
    (mechanisms A+B; approximate conditional coverage).
  * ``meta``       -- Phase-5.2 :class:`ConformalMetaLearner` (Alaa 2023).

Reported per (n, method, dim):
  * cov         -- marginal ITTE coverage (simultaneous over t; target >= 1-alpha);
  * mean_width  -- mean finite band width;
  * inf%        -- % unbounded bands (weak-overlap atom);
  * cov_gap     -- max-minus-min coverage across difficulty strata (conditional
                   miscoverage; smaller = better);
  * w_corr      -- corr(band width, true difficulty) across test units (adaptivity;
                   ~0 for the marginal band, high for adaptive ones);
  * w_cv        -- coefficient of variation of the width across X (X-dependence).

Runs on CPU with a process pool (default 8 workers -- safe for ~16 GB RAM; each
worker single-threaded BLAS). Example::

    .venv/bin/python -u experiments/phase5_adaptive_coverage.py \
        --ns 500 1000 --reps 200 --hetero 3.0 --workers 8 \
        --methods marginal normalized localized meta \
        --output experiments/out/phase5/hetero.txt

    # homoskedastic control (adaptivity should mostly vanish)
    .venv/bin/python -u experiments/phase5_adaptive_coverage.py \
        --ns 1000 --reps 200 --hetero 0.0 --workers 8 \
        --output experiments/out/phase5/homo.txt
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import time

import numpy as np


# ----------------------------------------------------------------- worker setup
def _limit_threads():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


_limit_threads()


def _interactions(X):
    """Well-specified propensity features: [X, x1*x2, x0*x2] (the true logit)."""
    X = np.atleast_2d(X)
    extra = np.column_stack([X[:, 1] * X[:, 2], X[:, 0] * X[:, 2]])
    return np.hstack([X, extra])


# ------------------------------------------------------------------- one replicate
def _fit_method(method, observed, tseq, cal_idx, tr_idx, nuis):
    """Build a fitted predictor for ``method`` sharing the same nuisance + split."""
    from tcda_uq.uq.conformal import (
        ITTEConformal, AdaptiveITTEConformal, ConformalMetaLearner,
    )
    if method == "marginal":
        return ITTEConformal.fit(
            observed, tseq, nuisance=nuis, calib_index=cal_idx,
            propensity_feature_fn=_interactions, weighted=True,
        )
    if method in ("normalized", "localized"):
        return AdaptiveITTEConformal.fit(
            observed, tseq, nuisance=nuis, calib_index=cal_idx, train_index=tr_idx,
            propensity_feature_fn=_interactions, weighted=True,
            local_scale=True, localize=(method == "localized"),
        )
    if method == "meta":
        # the meta-learner does its own internal train/calib split + tau_hat
        return ConformalMetaLearner.fit(
            observed, tseq, propensity_feature_fn=_interactions,
            feature_fn=_interactions, random_state=0,
        )
    raise ValueError(method)


def _one_rep(task):
    (n, seed, alpha, n_basis, resolution, noise, hetero, prop_scale, methods,
     m_test, calib_frac, groups) = task

    from tcda_uq.datasets import TriOracleSimulation
    from tcda_uq.uq.conformal import fit_split_nuisances

    sim = TriOracleSimulation(n_hom_dim=2, resolution=resolution, n_basis=n_basis,
                              noise_scale=noise, hetero_scale=hetero,
                              prop_scale=prop_scale, seed=0)
    n_hom = sim.n_hom_dim
    rng = np.random.default_rng(seed)

    # shared train/calibration split + nuisance (so methods differ only by the band)
    s = sim.sample(n, rng=rng)
    phi, A, X = s.observed
    perm = rng.permutation(n)
    n_cal = int(round(calib_frac * n))
    cal_idx = np.sort(perm[:n_cal])
    tr_idx = np.sort(perm[n_cal:])
    nuis = fit_split_nuisances(
        (phi[tr_idx], A[tr_idx], X[tr_idx]), sim.tseq, n_basis=n_basis,
        propensity_feature_fn=_interactions,
    )

    # fresh test set with oracle ITTE + oracle difficulty
    st = sim.sample(m_test, rng=rng)
    amp = sim.noise_amplitude(st.X)                       # true local difficulty
    # difficulty strata (equal-count) for approximate conditional coverage
    edges = np.quantile(amp, np.linspace(0, 1, groups + 1))
    strat = np.clip(np.searchsorted(edges, amp, side="right") - 1, 0, groups - 1)

    out = {}
    for method in methods:
        model = _fit_method(method, s.observed, sim.tseq, cal_idx, tr_idx, nuis)
        for d in range(n_hom):
            lo, hi, _ = model.band_bounds(st.X, alpha, d=d)
            tgt = st.oracle_itte[:, d, :]
            finite = np.isfinite(hi).all(1) & np.isfinite(lo).all(1)
            covered = ((tgt >= lo) & (tgt <= hi)).all(1)
            width = np.where(finite, (hi - lo).mean(1), np.nan)
            # conditional-coverage gap across difficulty strata
            gcov = [covered[strat == g].mean() for g in range(groups)
                    if np.any(strat == g)]
            cov_gap = float(np.max(gcov) - np.min(gcov)) if gcov else np.nan
            wf = width[finite]
            af = amp[finite]
            w_corr = (float(np.corrcoef(wf, af)[0, 1])
                      if wf.size > 2 and wf.std() > 0 else np.nan)
            w_cv = float(wf.std() / wf.mean()) if wf.size and wf.mean() > 0 else np.nan
            out[(method, d)] = dict(
                cov=float(covered.mean()),
                width=float(np.nanmean(width)) if finite.any() else np.inf,
                inf=float((~finite).mean()),
                cov_gap=cov_gap, w_corr=w_corr, w_cv=w_cv,
            )
    return n, out


# ------------------------------------------------------------------- driver
def run(args, rep_range):
    from multiprocessing import Pool

    n_hom = 2
    tasks = []
    seeds = np.random.SeedSequence(args.seed).spawn(len(args.ns) * args.reps)
    k = 0
    for n in args.ns:
        for r in range(args.reps):
            seed = seeds[k]; k += 1
            if r in rep_range:
                tasks.append((n, seed, args.alpha, args.n_basis, args.resolution,
                              args.noise, args.hetero, args.overlap, tuple(args.methods),
                              args.m_test, args.calib_frac, args.groups))

    acc = {(n, m, d): [] for n in args.ns for m in args.methods for d in range(n_hom)}
    t0 = time.time()
    with Pool(processes=args.workers, initializer=_limit_threads) as pool:
        for i, (n, out) in enumerate(pool.imap_unordered(_one_rep, tasks, chunksize=1), 1):
            for (m, d), v in out.items():
                acc[(n, m, d)].append(v)
            if i % max(1, len(tasks) // 20) == 0:
                print(f"# {i}/{len(tasks)} reps done ({time.time()-t0:.0f}s)", flush=True)
    return acc, n_hom


def _emit(args, acc, n_hom):
    hdr = (f"Phase 5 adaptive conformal | target={1-args.alpha:.2f} | hetero={args.hetero} "
           f"| overlap={args.overlap} | reps={args.reps} | m_test={args.m_test} "
           f"| groups={args.groups}")
    print(hdr, flush=True)
    print(f"{'n':>5} {'method':>11} {'dim':>3} {'cov':>7} {'mean_width':>11} "
          f"{'inf%':>6} {'cov_gap':>8} {'w_corr':>7} {'w_cv':>6}", flush=True)
    for n in args.ns:
        for m in args.methods:
            for d in range(n_hom):
                rows = acc[(n, m, d)]
                if not rows:
                    continue
                cov = np.mean([x["cov"] for x in rows])
                wid_vals = [x["width"] for x in rows if np.isfinite(x["width"])]
                wid = np.mean(wid_vals) if wid_vals else np.inf
                inf = 100 * np.mean([x["inf"] for x in rows])
                gap = np.nanmean([x["cov_gap"] for x in rows])
                wc = np.nanmean([x["w_corr"] for x in rows])
                wcv = np.nanmean([x["w_cv"] for x in rows])
                print(f"{n:>5} {m:>11} {d:>3} {cov:>7.3f} {wid:>11.4f} {inf:>6.1f} "
                      f"{gap:>8.3f} {wc:>7.3f} {wcv:>6.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[500, 1000])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-basis", type=int, default=5)
    ap.add_argument("--resolution", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--hetero", type=float, default=3.0,
                    help="heteroskedasticity scale (0 => homoskedastic control)")
    ap.add_argument("--overlap", type=float, default=1.0,
                    help="propensity-logit scale (>1 => weaker overlap)")
    ap.add_argument("--methods", nargs="+",
                    default=["marginal", "normalized", "localized", "meta"],
                    choices=["marginal", "normalized", "localized", "meta"])
    ap.add_argument("--m-test", type=int, default=300)
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--groups", type=int, default=4,
                    help="difficulty strata for the conditional-coverage gap")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")
    base, extra = divmod(args.reps, args.num_shards)
    lo = args.shard_id * base + min(args.shard_id, extra)
    hi = lo + base + int(args.shard_id < extra)
    rep_range = range(lo, hi)

    def _go():
        acc, n_hom = run(args, rep_range)
        _emit(args, acc, n_hom)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f, contextlib.redirect_stdout(f):
            _go()
        print(args.output.read_text())
    else:
        _go()


if __name__ == "__main__":
    main()
