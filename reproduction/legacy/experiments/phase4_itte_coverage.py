"""Phase 4.5 -- ITTE conformal coverage / width study (the prediction side).

Monte-Carlo evaluation of the composed conformal **prediction** band for the
individual topological treatment effect ``delta_{i,d}(.)`` against the tri-oracle
truth ``sample.oracle_itte`` (both potential outcomes are known synthetically --
the only setting where per-unit ITTE coverage is even *defined*). Per rep we fit
the split-conformal predictor on ``n`` train+calibration units and score a
**fresh** test set, so the reported coverage is the honest frequentist target

    P( delta_{n+1,d}(.) in C(X_{n+1}, .) )   simultaneously over t.

Reported per (n, method, homology dim):
  * cov(itte)   -- fraction of fresh test units whose true delta_d(.) is covered
                   (simultaneously over t);
  * mean_width  -- mean band width;
  * inf%        -- % of test bands that are unbounded (weak-overlap +inf atom).

**Methods.**
  * ``conformal`` -- weighted composed CP (Layers 1-3). Target: >= 1-alpha.
  * ``confidence``-- the CTATE *confidence* band (Phase 3) used, wrongly, as an
    individual band. It covers the *mean* tau_d(.,x) (reported as ``cov(mean)``)
    but under-covers a *draw*, and gets **worse** as n grows (width ~ 1/sqrt n
    while delta keeps its aleatoric spread). This is the confidence-vs-prediction
    contrast made quantitative. (Enable with ``--baseline``.)

**Stress sweeps** (the knobs the semi-synthetic TATE datasets can't dial):
  * ``--overlap S`` scales the propensity logit (S>1 => pi->{0,1}, weak overlap).
  * ``--misspec {none,pi,mu,both}`` -- double-robustness stress: mis-specify the
    propensity (constant pi), the outcome model (constant mu), or both. Weighted
    CP is doubly robust: coverage should survive ``pi`` **or** ``mu`` wrong.

Runs on CPU with a process pool (default 8 workers -- safe for ~16 GB RAM; each
worker is single-threaded BLAS to avoid oversubscription). Example::

    .venv/bin/python -u experiments/phase4_itte_coverage.py \
        --ns 200 500 1000 --reps 200 --alpha 0.1 --workers 8 --baseline \
        --output experiments/out/phase4/wellspec.txt

    # double-robustness grid
    for MS in none pi mu both; do
      .venv/bin/python -u experiments/phase4_itte_coverage.py \
        --ns 800 --reps 200 --misspec $MS --workers 8 \
        --output experiments/out/phase4/dr_$MS.txt; done
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
    """Single-threaded BLAS per worker (process-level parallelism does the rest)."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


_limit_threads()


def _interactions(X):
    """Well-specified propensity features: [X, x1*x2, x0*x2] (the true logit)."""
    extra = np.column_stack([X[:, 1] * X[:, 2], X[:, 0] * X[:, 2]])
    return np.hstack([X, extra])


def _const_feat(X):
    """Mis-specified feature map: a single constant column (ignores X)."""
    return np.zeros((np.asarray(X).shape[0], 1))


class _Nuisance:
    """Minimal nuisance object for :class:`ITTEConformal` (predict_mu + prop_model).

    Decouples the outcome model (a callable ``mu_predict``) from the propensity
    model so the double-robustness stress test can degrade either independently.
    """

    def __init__(self, mu_predict, prop_model, tseq):
        self._mu_predict = mu_predict
        self.prop_model = prop_model
        self.tseq = np.asarray(tseq, dtype=float)

    def predict_mu(self, X):
        return self._mu_predict(np.atleast_2d(np.asarray(X, dtype=float)))


def _build_nuisance(train, tseq, n_basis, misspec):
    """Fit outcome + propensity nuisances on the training split.

    ``misspec in {none, pi, mu, both}`` degrades the corresponding nuisance to a
    constant (ignores X) -- a genuine, strong misspecification.
    """
    from sklearn.linear_model import LogisticRegression
    from tcda_uq.estimators.nuisance import (
        fit_functional_regression, predict_functional_regression, fit_propensity,
    )

    phi, A, X = train
    A = np.asarray(A)
    n_hom = phi.shape[-2]
    bad_mu = misspec in ("mu", "both")
    bad_pi = misspec in ("pi", "both")

    if bad_mu:
        # constant per-arm mean curves (outcome model ignores X)
        m0 = [phi[A == 0][:, d, :].mean(axis=0) for d in range(n_hom)]
        m1 = [phi[A == 1][:, d, :].mean(axis=0) for d in range(n_hom)]

        def mu_predict(Xe):
            m = Xe.shape[0]
            return [(np.tile(m0[d], (m, 1)), np.tile(m1[d], (m, 1))) for d in range(n_hom)]
    else:
        reg = fit_functional_regression(train, tseq, n_basis=n_basis)

        def mu_predict(Xe):
            return predict_functional_regression(reg, Xe, tseq)

    pi_feat = _const_feat if bad_pi else _interactions
    prop = fit_propensity(pi_feat(X), A, LogisticRegression(max_iter=500))
    return _Nuisance(mu_predict, prop, tseq), pi_feat


# ------------------------------------------------------------------- one replicate
def _one_rep(task):
    """One replicate: fit on n, evaluate on a fresh test set. Returns metric dict."""
    (n, seed, alpha, n_basis, resolution, noise, prop_scale, misspec,
     m_test, baseline, calib_frac) = task

    from tcda_uq.datasets import TriOracleSimulation
    from tcda_uq.uq.conformal import ITTEConformal

    sim = TriOracleSimulation(n_hom_dim=2, resolution=resolution, n_basis=n_basis,
                              noise_scale=noise, prop_scale=prop_scale, seed=0)
    n_hom = sim.n_hom_dim
    rng = np.random.default_rng(seed)

    # ---- train/calibration sample + split
    s = sim.sample(n, rng=rng)
    phi, A, X = s.observed
    perm = rng.permutation(n)
    n_cal = int(round(calib_frac * n))
    cal_idx = np.sort(perm[:n_cal])
    tr_idx = np.sort(perm[n_cal:])
    train = (phi[tr_idx], A[tr_idx], X[tr_idx])

    nuis, pi_feat = _build_nuisance(train, sim.tseq, n_basis, misspec)
    model = ITTEConformal.fit(
        s.observed, sim.tseq, nuisance=nuis, calib_index=cal_idx,
        propensity_feature_fn=pi_feat, weighted=True,
    )

    # ---- fresh test set with oracle ITTE
    st = sim.sample(m_test, rng=rng)
    out = {}
    for d in range(n_hom):
        lo, hi, _ = model.band_bounds(st.X, alpha, d=d)
        tgt = st.oracle_itte[:, d, :]
        finite = np.isfinite(hi).all(axis=1) & np.isfinite(lo).all(axis=1)
        covered = ((tgt >= lo) & (tgt <= hi)).all(axis=1)
        width = np.where(finite, (hi - lo).mean(axis=1), np.nan)
        out[("conformal", d)] = dict(
            cov=float(covered.mean()),
            width=float(np.nanmean(width)) if finite.any() else np.inf,
            inf=float((~finite).mean()),
            cov_mean=np.nan,
        )

    # ---- confidence baseline (CTATE band used as an individual band)
    if baseline:
        from sklearn.linear_model import LogisticRegression
        from scipy.stats import norm
        from tcda_uq.estimators import CTATEDRLearner
        from tcda_uq.uq.asymptotic import ctate_pointwise_sd

        learner = CTATEDRLearner(n_basis=n_basis).fit(
            s.observed, sim.tseq, n_splits=2,
            propensity_estimator=LogisticRegression(max_iter=500),
            propensity_feature_fn=_interactions,
            random_state=int(rng.integers(1 << 31)),
        )
        z = norm.ppf(1 - alpha / 2)
        tau_true = sim.true_ctate(st.X)                    # [m, hom, res] (the mean)
        for d in range(n_hom):
            center = learner.predict_dim(st.X, d)          # [m, res]
            half = np.stack([z * ctate_pointwise_sd(learner, st.X[j], d=d)
                             for j in range(st.X.shape[0])])  # [m, res]
            lo, hi = center - half, center + half
            cov_itte = ((st.oracle_itte[:, d, :] >= lo)
                        & (st.oracle_itte[:, d, :] <= hi)).all(axis=1).mean()
            cov_mean = ((tau_true[:, d, :] >= lo)
                        & (tau_true[:, d, :] <= hi)).all(axis=1).mean()
            out[("confidence", d)] = dict(
                cov=float(cov_itte), width=float((hi - lo).mean()),
                inf=0.0, cov_mean=float(cov_mean),
            )
    return n, out


# ------------------------------------------------------------------- driver
def run(args, rep_range):
    from multiprocessing import Pool

    methods = ["conformal"] + (["confidence"] if args.baseline else [])
    n_hom = 2
    tasks = []
    seeds = np.random.SeedSequence(args.seed).spawn(len(args.ns) * args.reps)
    k = 0
    for n in args.ns:
        for r in range(args.reps):
            seed = seeds[k]; k += 1
            if r in rep_range:
                tasks.append((n, seed, args.alpha, args.n_basis, args.resolution,
                              args.noise, args.overlap, args.misspec, args.m_test,
                              args.baseline, args.calib_frac))

    # accumulators keyed (n, method, dim) -> list of per-rep dicts
    acc = {(n, m, d): [] for n in args.ns for m in methods for d in range(n_hom)}
    t0 = time.time()
    with Pool(processes=args.workers, initializer=_limit_threads) as pool:
        for i, (n, out) in enumerate(pool.imap_unordered(_one_rep, tasks, chunksize=1), 1):
            for (m, d), v in out.items():
                acc[(n, m, d)].append(v)
            if i % max(1, len(tasks) // 20) == 0:
                print(f"# {i}/{len(tasks)} reps done ({time.time()-t0:.0f}s)", flush=True)
    return methods, acc, n_hom


def _emit(args, methods, acc, n_hom):
    hdr = (f"Phase 4.5 ITTE conformal | target={1-args.alpha:.2f} | overlap={args.overlap} "
           f"| misspec={args.misspec} | reps={args.reps} | m_test={args.m_test}")
    print(hdr, flush=True)
    print(f"{'n':>5} {'method':>11} {'dim':>3} {'cov(itte)':>10} {'cov(mean)':>10} "
          f"{'mean_width':>11} {'inf%':>6}", flush=True)
    for n in args.ns:
        for m in methods:
            for d in range(n_hom):
                rows = acc[(n, m, d)]
                if not rows:
                    continue
                cov = np.mean([x["cov"] for x in rows])
                covm_vals = [x["cov_mean"] for x in rows if not np.isnan(x["cov_mean"])]
                covm = np.mean(covm_vals) if covm_vals else np.nan
                wid_vals = [x["width"] for x in rows if np.isfinite(x["width"])]
                wid = np.mean(wid_vals) if wid_vals else np.inf
                inf = 100 * np.mean([x["inf"] for x in rows])
                covm_s = "     -" if np.isnan(covm) else f"{covm:>10.3f}"
                print(f"{n:>5} {m:>11} {d:>3} {cov:>10.3f} {covm_s} "
                      f"{wid:>11.4f} {inf:>6.1f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[200, 500, 1000])
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-basis", type=int, default=5)
    ap.add_argument("--resolution", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--overlap", type=float, default=1.0,
                    help="propensity-logit scale (>1 => weaker overlap)")
    ap.add_argument("--misspec", choices=["none", "pi", "mu", "both"], default="none")
    ap.add_argument("--m-test", type=int, default=200)
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--baseline", action="store_true",
                    help="also evaluate the CTATE confidence band as an ITTE band")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260701)
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
        methods, acc, n_hom = run(args, rep_range)
        _emit(args, methods, acc, n_hom)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f, contextlib.redirect_stdout(f):
            _go()
        # echo to console too
        print(args.output.read_text())
    else:
        _go()


if __name__ == "__main__":
    main()
