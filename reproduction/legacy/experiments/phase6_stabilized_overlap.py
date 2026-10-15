"""Phase 6.5 / 6.6 -- positivity-stabilized weighting vs naive, across shrinking overlap (HEADLINE).

Phase 4.5's decision gate named the weak-overlap ``+inf`` atom the dominant *practical*
failure of the ITTE conformal band and set a concrete target: at overlap scales 2.5 / 4.0
(where naive weighting gives ``inf%`` ~ 12% / 34%), cut ``inf%`` toward single digits while
keeping coverage ``>= 0.90`` and finite width roughly flat. This experiment tests the
Phase-6.5 stabilizers against that target.

**Design (the residual-reuse trick of Phase 4.5).** Everything the band depends on is a
function of the per-arm residual curves ``r^a`` and the estimated propensities ``pi_hat``
(calibration + test). So we fit the nuisances **once per replicate**, store the residuals and
``pi_hat``, and then evaluate *every* stabilizer (naive / clip / shrink / tilted(gamma) /
matching / overlap) by pure post-processing -- only the overlap sweep refits (it changes the
DGP). Each stabilizer is just a different, bounded weight map fed to the identical weighted
conformal quantile (see :mod:`tcda_uq.uq.conformal.stabilized_weights`).

**Estimand honesty.** The naive/clip/shrink methods target the *full* interventional
population; overlap/tilted/matching target a *tilted* population where treatment is not
near-deterministic (their coverage guarantee is exact for that target). Because all bands are
conservative, the full-population empirical coverage stays ``>= 0.90`` for every method, so the
comparison the reader cares about is honest: at equal (>=nominal) coverage, which methods keep
the band **bounded and tight** under thin overlap.

Reported per (overlap scale, method, homology dim):
  * cov(uncond)  -- coverage of a fresh oracle ITTE over the full-population test set;
  * cov(finite)  -- coverage conditional on the band being bounded;
  * inf%         -- share of unbounded (``+inf``) bands  <-- the Phase-6.5 target;
  * width        -- mean *finite* band width;
  * ESS/n_cal    -- Kish effective calibration size / raw calib count (weighting efficiency);
  * max_wt       -- mean over test units of the normalized test-atom weight (fires the atom
                    when it exceeds ``beta = alpha/2``).

Runs on CPU with a process pool (default 8 workers, single-threaded BLAS per worker; safe for
~16 GB RAM). Example::

    .venv/bin/python -u experiments/phase6_stabilized_overlap.py \
        --scales 1.0 2.5 4.0 --reps 200 --workers 8 \
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


# --------------------------------------------------------------- nuisance plumbing
def _interactions(X):
    X = np.asarray(X)
    return np.hstack([X, np.column_stack([X[:, 1] * X[:, 2], X[:, 0] * X[:, 2]])])


def _const_feat(X):
    return np.zeros((np.asarray(X).shape[0], 1))


class _Nuisance:
    def __init__(self, mu_predict, prop_model, tseq):
        self._mu_predict = mu_predict
        self.prop_model = prop_model
        self.tseq = np.asarray(tseq, dtype=float)

    def predict_mu(self, X):
        return self._mu_predict(np.atleast_2d(np.asarray(X, dtype=float)))


def _build_nuisance(train, tseq, n_basis, misspec):
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


# --------------------------------------------------------------- one fit -> raw materials
def _fit_raw(n, seed, *, resolution, n_basis, noise, prop_scale, misspec, m_test, calib_frac):
    """Fit once; return per-arm residual curves + calibration/test *pi_hat* (not weights).

    Storing ``pi_hat`` (not the naive weights) is what lets every stabilizer be applied by
    post-processing.
    """
    from tcda_uq.datasets import TriOracleSimulation

    sim = TriOracleSimulation(n_hom_dim=2, resolution=resolution, n_basis=n_basis,
                              noise_scale=noise, prop_scale=prop_scale, seed=0)
    n_hom = sim.n_hom_dim
    rng = np.random.default_rng(seed)

    s = sim.sample(n, rng=rng)
    phi, A, X = s.observed
    A = np.asarray(A)
    perm = rng.permutation(n)
    n_cal = int(round(calib_frac * n))
    cal_idx = np.sort(perm[:n_cal])
    tr_idx = np.sort(perm[n_cal:])
    train = (phi[tr_idx], A[tr_idx], X[tr_idx])

    nuis, pi_feat = _build_nuisance(train, sim.tseq, n_basis, misspec)

    def pi_of(Xe):
        return nuis.prop_model.predict_proba(pi_feat(np.atleast_2d(Xe)))[:, 1]

    Xc, Ac, phic = X[cal_idx], A[cal_idx], phi[cal_idx]
    mu_c = nuis.predict_mu(Xc)
    r_cal = [[None, None] for _ in range(n_hom)]
    pi_cal = [None, None]
    for a in (0, 1):
        mask = Ac == a
        pi_cal[a] = pi_of(Xc[mask])
        for d in range(n_hom):
            r_cal[d][a] = phic[mask, d, :] - mu_c[d][a][mask]

    st = sim.sample(m_test, rng=rng)
    mu_t = nuis.predict_mu(st.X)
    r_test = [[None, None] for _ in range(n_hom)]
    for a in (0, 1):
        for d in range(n_hom):
            r_test[d][a] = st.potential_outcomes[:, a, d, :] - mu_t[d][a]
    pi_test = pi_of(st.X)
    return dict(tseq=sim.tseq, n_hom=n_hom, r_cal=r_cal, pi_cal=pi_cal,
                r_test=r_test, pi_test=pi_test)


# --------------------------------------------------------------- post-processing
def _arm(r_cal, pi_cal, r_test, pi_test, arm, beta, weight_fn, *, mod="pointwise-sd"):
    """Evaluate one arm at level ``1-beta`` under weight map ``weight_fn``. Post-processing only."""
    from tcda_uq.uq.conformal.functional_cp import modulation, sup_norm_score
    from tcda_uq.uq.conformal.weighted_cp import weighted_conformal_radius

    r_cal = np.asarray(r_cal, float)
    r_test = np.atleast_2d(np.asarray(r_test, float))
    s = modulation(r_cal, mod)
    score_cal = np.atleast_1d(sup_norm_score(r_cal, s))
    w_cal = weight_fn(pi_cal, arm)
    w_test = weight_fn(pi_test, arm)
    k = np.atleast_1d(weighted_conformal_radius(score_cal, beta, w_cal, w_test))
    finite = np.isfinite(k)
    ess = float((w_cal.sum() ** 2) / np.sum(w_cal ** 2))
    max_test_w = float(np.mean(w_test / (w_cal.sum() + w_test)))
    return dict(k=k, s=s, finite=finite, ess=ess, ncal=len(score_cal), max_test_w=max_test_w)


def _itte(a1, a0, r1_test, r0_test):
    """ITTE coverage + finite mask + finite width from two evaluated arms."""
    r1 = np.atleast_2d(np.asarray(r1_test, float))
    r0 = np.atleast_2d(np.asarray(r0_test, float))
    half = a1["k"][:, None] * a1["s"][None, :] + a0["k"][:, None] * a0["s"][None, :]
    diff = np.abs(r1 - r0)
    finite = np.isfinite(half).all(axis=1)
    with np.errstate(invalid="ignore"):
        cov = (diff <= half).all(axis=1)
    cov = np.where(finite, cov, True)
    width = np.where(finite, 2.0 * half.mean(axis=1), np.nan)
    return cov, finite, width


def _rep(task):
    n, seed, alpha, cfg, methods = task
    from tcda_uq.uq.conformal import make_weight_fn
    raw = _fit_raw(n, seed, **cfg)
    beta = alpha / 2.0
    out = {name: {} for name in methods}
    for name, spec in methods.items():
        wfn = make_weight_fn(spec[0], **spec[1])
        for d in range(raw["n_hom"]):
            rc, pc = raw["r_cal"][d], raw["pi_cal"]
            rt, pt = raw["r_test"][d], raw["pi_test"]
            a1 = _arm(rc[1], pc[1], rt[1], pt, 1, beta, wfn)
            a0 = _arm(rc[0], pc[0], rt[0], pt, 0, beta, wfn)
            cov, fin, width = _itte(a1, a0, rt[1], rt[0])
            out[name][d] = dict(
                cov_uncond=float(np.mean(cov)),
                cov_finite=float(np.mean(cov[fin])) if fin.any() else np.nan,
                inf=float(np.mean(~fin)),
                width=float(np.nanmean(width)) if fin.any() else np.nan,
                ess=0.5 * (a1["ess"] + a0["ess"]), ncal=0.5 * (a1["ncal"] + a0["ncal"]),
                max_wt=0.5 * (a1["max_test_w"] + a0["max_test_w"]),
            )
    return out


# --------------------------------------------------------------- driver
def _methods(args):
    """Ordered dict name -> (stabilizer, params)."""
    m = {
        "naive": ("naive", {}),
        f"clip(eps={args.eps})": ("clip", dict(eps=args.eps)),
        f"shrink(cap={args.cap})": ("shrink", dict(cap=args.cap)),
        f"tilted(g={args.gamma})": ("tilted", dict(gamma=args.gamma)),
        "matching": ("matching", {}),
        "overlap": ("overlap", {}),
    }
    return m


def _seeds(args):
    return list(np.random.SeedSequence(args.seed).spawn(args.reps))


def _parallel(fn, tasks, workers):
    from multiprocessing import Pool
    out, t0 = [], time.time()
    step = max(1, len(tasks) // 10)
    with Pool(processes=workers, initializer=_limit_threads) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, tasks, chunksize=1), 1):
            out.append(r)
            if i % step == 0:
                print(f"#   {i}/{len(tasks)} reps ({time.time()-t0:.0f}s)", flush=True)
    return out


def run(args):
    methods = _methods(args)
    print(f"Phase 6.5/6.6 positivity-stabilized weighting | target={1-args.alpha:.2f} | n={args.n} "
          f"| misspec={args.misspec} | reps={args.reps} | m_test={args.m_test}", flush=True)
    print("Estimand: naive/clip/shrink target the full population; overlap/tilted/matching a "
          "tilted (overlap) population (exact coverage there). All bands conservative -> "
          "full-population coverage stays >= nominal; the story is inf% and width.\n", flush=True)
    print(f"{'scale':>6} {'method':>16} {'dim':>3} {'cov':>7} {'cov_fin':>8} {'inf%':>6} "
          f"{'width':>8} {'ESS/ncal':>9} {'max_wt':>7}", flush=True)
    summary = []
    for scale in args.scales:
        cfg = dict(resolution=args.resolution, n_basis=args.n_basis, noise=args.noise,
                   prop_scale=scale, misspec=args.misspec, m_test=args.m_test,
                   calib_frac=args.calib_frac)
        tasks = [(args.n, s, args.alpha, cfg, methods) for s in _seeds(args)]
        rows = _parallel(_rep, tasks, args.workers)
        for name in methods:
            for d in range(2):
                vs = [r[name][d] for r in rows]
                cov = np.mean([x["cov_uncond"] for x in vs])
                covf = np.nanmean([x["cov_finite"] for x in vs])
                inf = 100 * np.mean([x["inf"] for x in vs])
                width = np.nanmean([x["width"] for x in vs])
                ess = np.mean([x["ess"] for x in vs]); ncal = np.mean([x["ncal"] for x in vs])
                maxw = np.mean([x["max_wt"] for x in vs])
                print(f"{scale:>6.1f} {name:>16} {d:>3} {cov:>7.3f} {covf:>8.3f} {inf:>6.1f} "
                      f"{width:>8.3f} {ess/ncal:>9.3f} {maxw:>7.3f}", flush=True)
                summary.append((scale, name, d, cov, covf, inf, width, ess / ncal, maxw))
        print(flush=True)
    # headline delta at the Phase-4.5 target scales
    print("Headline (dim-averaged inf% and finite width; naive -> overlap):", flush=True)
    for scale in args.scales:
        def _avg(nm, j):
            xs = [row[j] for row in summary if row[0] == scale and row[1] == nm]
            return float(np.mean(xs))
        print(f"  scale {scale:>4.1f}: inf%  {_avg('naive',5):5.1f} -> {_avg('overlap',5):5.1f}   "
              f"width {_avg('naive',6):5.3f} -> {_avg('overlap',6):5.3f}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-basis", type=int, default=5)
    ap.add_argument("--resolution", type=int, default=40)
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    ap.add_argument("--misspec", choices=["none", "pi", "mu", "both"], default="none")
    ap.add_argument("--m-test", type=int, default=200)
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--eps", type=float, default=0.1, help="clip method: pi clip level")
    ap.add_argument("--cap", type=float, default=10.0, help="shrink method: weight cap")
    ap.add_argument("--gamma", type=float, default=0.5, help="tilted method: overlap-tilt exponent")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()

    def _go():
        run(args)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        out = args.output_dir / "stabilized_overlap.txt"
        with out.open("w") as f, contextlib.redirect_stdout(f):
            _go()
        print(out.read_text())
    else:
        _go()


if __name__ == "__main__":
    main()
