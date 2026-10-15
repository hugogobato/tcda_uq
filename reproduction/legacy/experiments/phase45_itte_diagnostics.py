"""Phase 4.5 -- tightness, usability and diagnostic audit of the ITTE conformal band.

Phase 4 validated the headline estimand/tool claim (conformal covers the individual
``delta_{i,d}(.)`` while confidence bands collapse onto the mean), but the MVP band
is *very* conservative: ``~0.997`` empirical coverage at a ``0.90`` target, and weak
overlap manifests as a rising share of **unbounded** (``+inf``) bands. This script
**decomposes** that conservatism and turns the weak-overlap failure into quantitative
design targets for the positivity-stabilized weighting of Phase 6.5.

The construction being audited (``tcda_uq.uq.conformal``) is, per homology dim ``d``:

  1. Layer 1 -- per arm ``a``, sup-norm score ``S_i = max_t |r_i(t)| / s(t)`` with
     ``r_i = phi^a_i - mu_hat_a(X_i)`` and modulation ``s(t)``; radius = weighted
     conformal quantile of ``{S_i}`` (Layer 2), band ``mu_hat_a(.,x) +/- k_a s_a(.)``.
  2. Layer 3 -- compose the two arms at level ``alpha/2`` each via Bonferroni +
     interval arithmetic: ``delta`` band centered at ``mu_hat_1 - mu_hat_0`` with
     half-width ``k_1 s_1(.) + k_0 s_0(.)``.

**Key structural fact this script exploits.** *Everything* the band coverage depends
on is a function of the **arm residual curves** ``r^a`` (calibration + test) and the
**propensity weights** ``w^a`` (calibration + test). Concretely, on a fresh test unit:

  * arm-``a`` coverage of ``phi^a``:      ``max_t |r^a_test(t)| / s_a(t) <= k_a``
  * ITTE coverage of ``delta``:           ``|r^1_test(t) - r^0_test(t)| <= k_1 s_1(t) + k_0 s_0(t)  for all t``

(the ITTE center error ``delta - (mu_hat_1 - mu_hat_0) = r^1_test - r^0_test``). So one
model fit yields, essentially free, *every* rung of the ladder and every modulation /
arm-split / level variant by pure post-processing of the stored residuals + weights --
only the resolution and overlap sweeps need refits. Each mode below is that
post-processing.

Modes (``--mode``):
  * ``decompose`` (4.5.1) -- the conservatism ladder: base per-arm CP, alpha->alpha/2
    inflation, Bonferroni union slack, interval-arithmetic slack, grid slack. Attributes
    the ``0.90 -> 0.997`` gap to named sources.
  * ``finite`` (4.5.2) -- finite-band diagnostics: unconditional vs finite-conditional
    coverage, finite-band share, width mean/median/quantiles, ESS of the weighted
    calibration law, max normalized test weight.
  * ``arms`` (4.5.3) -- arm-composition alternatives: Bonferroni(a/2,a/2) vs Sidak vs a
    width-optimized unequal split; the (un)availability of a joint/stacked score.
  * ``modulation`` (4.5.4) -- constant vs pointwise-sd vs lipschitz modulation, across
    grid resolutions, with/without the whole-interval grid slack.
  * ``overlap`` (4.5.5) -- inf%, effective calibration size (ESS), max normalized test
    weight and width across overlap scales -> the Phase 6.5 target.
  * ``all`` -- run every mode with its default config.

Runs on CPU with a process pool (default 8 workers -- safe for ~16 GB RAM; BLAS pinned
to one thread per worker). Example::

    .venv/bin/python -u experiments/phase45_itte_diagnostics.py --mode all \
        --reps 200 --workers 8 --output-dir experiments/out/phase45
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


# ----------------------------------------------------------------- nuisance plumbing
# (mirrors experiments/phase4_itte_coverage.py so the numbers here line up with the
#  Phase-4 production tables the audit is explaining.)
def _interactions(X):
    """Well-specified propensity features: [X, x1*x2, x0*x2] (the true logit)."""
    X = np.asarray(X)
    extra = np.column_stack([X[:, 1] * X[:, 2], X[:, 0] * X[:, 2]])
    return np.hstack([X, extra])


def _const_feat(X):
    """Mis-specified feature map: a single constant column (ignores X)."""
    return np.zeros((np.asarray(X).shape[0], 1))


class _Nuisance:
    """Minimal nuisance object (predict_mu + prop_model) for the arms."""

    def __init__(self, mu_predict, prop_model, tseq):
        self._mu_predict = mu_predict
        self.prop_model = prop_model
        self.tseq = np.asarray(tseq, dtype=float)

    def predict_mu(self, X):
        return self._mu_predict(np.atleast_2d(np.asarray(X, dtype=float)))


def _build_nuisance(train, tseq, n_basis, misspec):
    """Fit outcome + propensity nuisances on the training split (``misspec`` degrades one)."""
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


# ----------------------------------------------------------------- one fit -> raw materials
def _fit_residuals(n, seed, *, resolution, n_basis, noise, prop_scale, misspec,
                   m_test, calib_frac):
    """Fit once; return the raw per-arm residual curves + propensity weights.

    Returns a dict with, per homology dim ``d`` and arm ``a``:
        r_cal[d][a]  : calibration residual curves phi^a - mu_hat_a   [n_cal_a, res]
        w_cal[d][a]  : calibration propensity LR weights              [n_cal_a]   (same for all d)
        r_test[d][a] : test residual curves (oracle potential outcome) [m, res]
        w_test[a]    : test propensity LR weights                     [m]
    plus ``tseq``. Post-processing (all modes) is a pure function of these.
    """
    from tcda_uq.datasets import TriOracleSimulation
    from tcda_uq.uq.conformal.weighted_cp import propensity_weights

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

    # calibration residuals + weights, per arm
    Xc = X[cal_idx]
    Ac = A[cal_idx]
    phic = phi[cal_idx]
    mu_c = nuis.predict_mu(Xc)                       # list per dim of (mu0[m,res], mu1[m,res])
    r_cal = [[None, None] for _ in range(n_hom)]
    w_cal = [[None, None] for _ in range(n_hom)]
    for a in (0, 1):
        mask = Ac == a
        w_a = propensity_weights(pi_of(Xc[mask]), a)
        for d in range(n_hom):
            r_cal[d][a] = phic[mask, d, :] - mu_c[d][a][mask]
            w_cal[d][a] = w_a

    # fresh test set: oracle potential outcomes -> per-arm test residuals
    st = sim.sample(m_test, rng=rng)
    mu_t = nuis.predict_mu(st.X)
    r_test = [[None, None] for _ in range(n_hom)]
    for a in (0, 1):
        for d in range(n_hom):
            r_test[d][a] = st.potential_outcomes[:, a, d, :] - mu_t[d][a]
    w_test = [propensity_weights(pi_of(st.X), a) for a in (0, 1)]

    return dict(tseq=sim.tseq, n_hom=n_hom, r_cal=r_cal, w_cal=w_cal,
                r_test=r_test, w_test=w_test)


# ----------------------------------------------------------------- post-processing kernels
def _arm(r_cal, w_cal, r_test, w_test, beta, *, mod="pointwise-sd", floor=0.0,
         tseq=None, grid_L=0.0, weighted=True):
    """Evaluate one arm at level ``1-beta``. Pure post-processing of stored residuals.

    Returns dict: k[m] radii, s[res] modulation, cov[m] bool (phi^a covered),
    finite[m] bool, width[m] (mean band width, nan if infinite), ess (Kish ESS of
    calib weights), score_cal[n] calibration scores.
    """
    from tcda_uq.uq.conformal.functional_cp import (
        modulation, sup_norm_score, grid_discretization_slack)
    from tcda_uq.uq.conformal.weighted_cp import weighted_conformal_radius

    r_cal = np.asarray(r_cal, float)
    r_test = np.atleast_2d(np.asarray(r_test, float))
    s = modulation(r_cal, mod, floor=floor)
    score_cal = np.atleast_1d(sup_norm_score(r_cal, s))
    if grid_L > 0.0:
        score_cal = score_cal + grid_discretization_slack(tseq, s, grid_L)

    if not weighted:
        w_cal = np.ones_like(np.asarray(w_cal, float))
        w_test = np.ones_like(np.atleast_1d(np.asarray(w_test, float)))
    w_cal = np.asarray(w_cal, float)
    w_test = np.atleast_1d(np.asarray(w_test, float))

    k = np.atleast_1d(weighted_conformal_radius(score_cal, beta, w_cal, w_test))   # [m]
    test_score = np.atleast_1d(sup_norm_score(r_test, s))                          # [m]
    finite = np.isfinite(k)
    cov = test_score <= k                                                          # inf k -> always covered
    width = np.where(finite, 2.0 * k * s.mean(), np.nan)
    ess = float((w_cal.sum() ** 2) / np.sum(w_cal ** 2))
    max_test_w = float(np.max(w_test / (w_cal.sum() + w_test)))
    return dict(k=k, s=s, cov=cov, finite=finite, width=width, ess=ess,
                score_cal=score_cal, max_test_w=max_test_w)


def _itte_cov(a1, a0, r1_test, r0_test):
    """ITTE coverage + width from two evaluated arms (arm dicts + test residuals).

    Covered <=> |r1(t) - r0(t)| <= k1 s1(t) + k0 s0(t)  for all t.
    Returns (cov[m] bool, finite[m] bool, width[m] mean-width nan-if-inf).
    """
    r1 = np.atleast_2d(np.asarray(r1_test, float))
    r0 = np.atleast_2d(np.asarray(r0_test, float))
    half = a1["k"][:, None] * a1["s"][None, :] + a0["k"][:, None] * a0["s"][None, :]  # [m,res]
    diff = np.abs(r1 - r0)                                                            # [m,res]
    finite = np.isfinite(half).all(axis=1)
    with np.errstate(invalid="ignore"):
        cov = (diff <= half).all(axis=1)
    cov = np.where(finite, cov, True)          # infinite band trivially covers
    width = np.where(finite, 2.0 * half.mean(axis=1), np.nan)  # mean_t (upper - lower)
    return cov, finite, width


# ================================================================= MODE: decompose (4.5.1)
def _rep_decompose(task):
    n, seed, alpha, cfg = task
    raw = _fit_residuals(n, seed, **cfg)
    out = {}
    for d in range(raw["n_hom"]):
        rc, wc, rt, wt = raw["r_cal"][d], raw["w_cal"][d], raw["r_test"][d], raw["w_test"]
        kw = dict(mod="pointwise-sd", tseq=raw["tseq"], weighted=True)
        # per-arm at full alpha and at alpha/2
        a1_full = _arm(rc[1], wc[1], rt[1], wt[1], alpha, **kw)
        a0_full = _arm(rc[0], wc[0], rt[0], wt[0], alpha, **kw)
        a1_half = _arm(rc[1], wc[1], rt[1], wt[1], alpha / 2, **kw)
        a0_half = _arm(rc[0], wc[0], rt[0], wt[0], alpha / 2, **kw)
        # + grid slack variant (whole-interval guarantee) at alpha/2, L=2
        a1_gs = _arm(rc[1], wc[1], rt[1], wt[1], alpha / 2, grid_L=2.0, **kw)
        a0_gs = _arm(rc[0], wc[0], rt[0], wt[0], alpha / 2, grid_L=2.0, **kw)

        cov_itte, fin_itte, w_itte = _itte_cov(a1_half, a0_half, rt[1], rt[0])
        cov_itte_gs, _, w_itte_gs = _itte_cov(a1_gs, a0_gs, rt[1], rt[0])
        both = a1_half["cov"] & a0_half["cov"]

        out[d] = dict(
            arm_full=0.5 * float(a1_full["cov"].mean() + a0_full["cov"].mean()),
            arm_half=0.5 * float(a1_half["cov"].mean() + a0_half["cov"].mean()),
            both=float(np.mean(both)),
            itte=float(np.mean(cov_itte)),
            itte_gs=float(np.mean(cov_itte_gs)),
            width_arm_half=0.5 * float(np.nanmean(a1_half["width"]) + np.nanmean(a0_half["width"])),
            width_itte=float(np.nanmean(w_itte)),
            width_itte_gs=float(np.nanmean(w_itte_gs)),
            inf_itte=float(np.mean(~fin_itte)),
        )
    return n, out


def _run_decompose(args):
    cfg = _cfg(args)
    rows = _parallel(_rep_decompose, [(args.n, s, args.alpha, cfg) for s in _seeds(args)],
                     args.workers)
    acc = _accum(rows)
    tgt = 1 - args.alpha
    print(f"Phase 4.5.1 conservatism decomposition | target={tgt:.2f} | n={args.n} "
          f"| overlap={args.overlap} | misspec={args.misspec} | reps={args.reps} "
          f"| m_test={args.m_test}", flush=True)
    print("\nCoverage ladder (each row a nonconformity rung; gaps attribute the slack):",
          flush=True)
    hdr = f"{'dim':>3} {'arm@a':>8} {'arm@a/2':>8} {'both@a/2':>9} {'ITTE':>7} {'ITTE+grid':>10}"
    print(hdr, flush=True)
    for d in sorted(acc):
        m = acc[d]
        print(f"{d:>3} {m['arm_full']:>8.3f} {m['arm_half']:>8.3f} {m['both']:>9.3f} "
              f"{m['itte']:>7.3f} {m['itte_gs']:>10.3f}", flush=True)
    print("\nAttribution of the coverage gap (percentage points above the 0.90 target), "
          "averaged over dims:", flush=True)
    dims = sorted(acc)
    g_base = 100 * np.mean([acc[d]['arm_full'] - tgt for d in dims])
    g_level = 100 * np.mean([acc[d]['arm_half'] - acc[d]['arm_full'] for d in dims])
    g_union = 100 * np.mean([acc[d]['both'] - acc[d]['arm_half'] for d in dims])
    g_iarith = 100 * np.mean([acc[d]['itte'] - acc[d]['both'] for d in dims])
    g_total = 100 * np.mean([acc[d]['itte'] - tgt for d in dims])
    print(f"  base per-arm CP (sup-norm + modulation + weighting + discreteness) : {g_base:+6.2f} pp",
          flush=True)
    print(f"  alpha -> alpha/2 per-arm level inflation (half the Bonferroni budget): {g_level:+6.2f} pp",
          flush=True)
    print(f"  Bonferroni union: 'both covered' minus single-arm (a REDUCTION)      : {g_union:+6.2f} pp",
          flush=True)
    print(f"  interval-arithmetic slack (delta covered w/o both arms covered)      : {g_iarith:+6.2f} pp",
          flush=True)
    print(f"  {'-'*68}", flush=True)
    print(f"  total: empirical ITTE coverage minus target                         : {g_total:+6.2f} pp",
          flush=True)
    print("\nWidth + grid-slack effect (mean over dims):", flush=True)
    print(f"  mean per-arm width @ a/2      : {np.mean([acc[d]['width_arm_half'] for d in dims]):.4f}",
          flush=True)
    print(f"  mean ITTE width               : {np.mean([acc[d]['width_itte'] for d in dims]):.4f}",
          flush=True)
    print(f"  mean ITTE width + grid slack  : {np.mean([acc[d]['width_itte_gs'] for d in dims]):.4f} "
          f"(whole-interval guarantee, L=2)", flush=True)
    print(f"  ITTE inf%                     : {100*np.mean([acc[d]['inf_itte'] for d in dims]):.1f}",
          flush=True)


# ================================================================= MODE: finite (4.5.2)
def _rep_finite(task):
    n, seed, alpha, cfg = task
    raw = _fit_residuals(n, seed, **cfg)
    out = {}
    for d in range(raw["n_hom"]):
        rc, wc, rt, wt = raw["r_cal"][d], raw["w_cal"][d], raw["r_test"][d], raw["w_test"]
        kw = dict(mod="pointwise-sd", tseq=raw["tseq"], weighted=True)
        a1 = _arm(rc[1], wc[1], rt[1], wt[1], alpha / 2, **kw)
        a0 = _arm(rc[0], wc[0], rt[0], wt[0], alpha / 2, **kw)
        cov, fin, width = _itte_cov(a1, a0, rt[1], rt[0])
        finite_widths = width[fin]
        out[d] = dict(
            cov_uncond=float(np.mean(cov)),
            cov_finite=float(np.mean(cov[fin])) if fin.any() else np.nan,
            finite_share=float(np.mean(fin)),
            width_mean=float(np.mean(finite_widths)) if fin.any() else np.nan,
            width_med=float(np.median(finite_widths)) if fin.any() else np.nan,
            width_q90=float(np.quantile(finite_widths, 0.90)) if fin.any() else np.nan,
            ess1=a1["ess"], ess0=a0["ess"],
            ncal1=len(a1["score_cal"]), ncal0=len(a0["score_cal"]),
            maxw1=a1["max_test_w"], maxw0=a0["max_test_w"],
        )
    return n, out


def _run_finite(args):
    cfg = _cfg(args)
    rows = _parallel(_rep_finite, [(args.n, s, args.alpha, cfg) for s in _seeds(args)],
                     args.workers)
    acc = _accum(rows)
    print(f"Phase 4.5.2 finite-band diagnostics | target={1-args.alpha:.2f} | n={args.n} "
          f"| overlap={args.overlap} | misspec={args.misspec} | reps={args.reps} "
          f"| m_test={args.m_test}", flush=True)
    print(f"{'dim':>3} {'cov(uncond)':>12} {'cov(finite)':>12} {'finite%':>8} "
          f"{'w_mean':>8} {'w_med':>8} {'w_q90':>8} {'ESS/n_cal':>11} {'max_wt':>7}", flush=True)
    for d in sorted(acc):
        m = acc[d]
        ess = 0.5 * (m['ess1'] + m['ess0'])
        ncal = 0.5 * (m['ncal1'] + m['ncal0'])
        maxw = 0.5 * (m['maxw1'] + m['maxw0'])
        print(f"{d:>3} {m['cov_uncond']:>12.3f} {m['cov_finite']:>12.3f} "
              f"{100*m['finite_share']:>7.1f} {m['width_mean']:>8.3f} {m['width_med']:>8.3f} "
              f"{m['width_q90']:>8.3f} {ess:>6.0f}/{ncal:<4.0f} {maxw:>7.3f}", flush=True)
    print("\nESS = Kish effective calibration size (sum w)^2/sum w^2; n_cal = raw arm "
          "calib count; max_wt = mean over test units of w_new/(sum w_cal + w_new).",
          flush=True)


# ================================================================= MODE: arms (4.5.3)
def _run_arms(args):
    cfg = _cfg(args)
    # per-arm alpha allocations: (name, beta1, beta0)
    a = args.alpha
    sidak = 1 - (1 - a) ** 0.5
    splits = {
        "bonferroni a/2": (a / 2, a / 2),
        "sidak": (sidak, sidak),
        "unequal 0.3/0.7": (0.7 * a, 0.3 * a),
        "unequal 0.7/0.3": (0.3 * a, 0.7 * a),
    }
    rows = _parallel(_rep_arms, [(args.n, s, a, cfg, splits) for s in _seeds(args)],
                     args.workers)
    # rows: list of (n, {split -> {dim -> dict}})
    print(f"Phase 4.5.3 arm-composition alternatives | target={1-a:.2f} | n={args.n} "
          f"| overlap={args.overlap} | reps={args.reps} | m_test={args.m_test}", flush=True)
    print(f"{'split (beta1/beta0)':>20} {'dim':>3} {'cov(uncond)':>12} {'cov(finite)':>12} "
          f"{'finite%':>8} {'w_mean':>8}", flush=True)
    for name in splits:
        acc = {}
        for _, per in rows:
            for d, v in per[name].items():
                acc.setdefault(d, []).append(v)
        for d in sorted(acc):
            vs = acc[d]
            cu = np.mean([x['cov_uncond'] for x in vs])
            cf = np.mean([x['cov_finite'] for x in vs if not np.isnan(x['cov_finite'])])
            fs = 100 * np.mean([x['finite_share'] for x in vs])
            wm = np.mean([x['width_mean'] for x in vs if not np.isnan(x['width_mean'])])
            print(f"{name:>20} {d:>3} {cu:>12.3f} {cf:>12.3f} {fs:>8.1f} {wm:>8.3f}", flush=True)
    print("\nNote: arms calibrate on disjoint (treated / control) units, so no joint "
          "score across arms exists -- the alpha-split is the only Layer-3 lever, and "
          "(as the widths show) a small one; the interval arithmetic dominates.", flush=True)


def _rep_arms(task):
    n, seed, alpha, cfg, splits = task
    raw = _fit_residuals(n, seed, **cfg)
    per = {name: {} for name in splits}
    kw = dict(mod="pointwise-sd", tseq=raw["tseq"], weighted=True)
    for name, (b1, b0) in splits.items():
        for d in range(raw["n_hom"]):
            rc, wc, rt, wt = raw["r_cal"][d], raw["w_cal"][d], raw["r_test"][d], raw["w_test"]
            a1 = _arm(rc[1], wc[1], rt[1], wt[1], b1, **kw)
            a0 = _arm(rc[0], wc[0], rt[0], wt[0], b0, **kw)
            cov, fin, width = _itte_cov(a1, a0, rt[1], rt[0])
            per[name][d] = dict(
                cov_uncond=float(np.mean(cov)),
                cov_finite=float(np.mean(cov[fin])) if fin.any() else np.nan,
                finite_share=float(np.mean(fin)),
                width_mean=float(np.nanmean(width)) if fin.any() else np.nan,
            )
    return n, per


# ================================================================= MODE: modulation (4.5.4)
def _run_modulation(args):
    resolutions = args.resolutions or [20, 40, 80]
    mods = ["constant", "pointwise-sd", "lipschitz"]
    print(f"Phase 4.5.4 modulation audit | target={1-args.alpha:.2f} | n={args.n} "
          f"| overlap={args.overlap} | reps={args.reps} | m_test={args.m_test}", flush=True)
    print(f"{'res':>4} {'modulation':>13} {'grid_slack':>10} {'dim':>3} {'cov':>7} "
          f"{'finite%':>8} {'w_mean':>8}", flush=True)
    for res in resolutions:
        cfg = _cfg(args, resolution=res)
        tasks = [(args.n, s, args.alpha, cfg, mods) for s in _seeds(args)]
        rows = _parallel(_rep_modulation, tasks, args.workers)
        # rows: (n, {(mod, gs) -> {dim -> dict}})
        combos = sorted({k for _, per in rows for k in per})
        for (mod, gs) in combos:
            acc = {}
            for _, per in rows:
                for d, v in per[(mod, gs)].items():
                    acc.setdefault(d, []).append(v)
            for d in sorted(acc):
                vs = acc[d]
                cu = np.mean([x['cov_uncond'] for x in vs])
                fs = 100 * np.mean([x['finite_share'] for x in vs])
                wm = np.mean([x['width_mean'] for x in vs if not np.isnan(x['width_mean'])])
                gs_s = "on(L=2)" if gs else "off"
                print(f"{res:>4} {mod:>13} {gs_s:>10} {d:>3} {cu:>7.3f} {fs:>8.1f} "
                      f"{wm:>8.3f}", flush=True)
        print(flush=True)


def _rep_modulation(task):
    n, seed, alpha, cfg, mods = task
    raw = _fit_residuals(n, seed, **cfg)
    per = {}
    for mod in mods:
        for gs in (False, True):
            grid_L = 2.0 if gs else 0.0
            key = (mod, gs)
            per[key] = {}
            for d in range(raw["n_hom"]):
                rc, wc, rt, wt = raw["r_cal"][d], raw["w_cal"][d], raw["r_test"][d], raw["w_test"]
                kw = dict(mod=mod, tseq=raw["tseq"], grid_L=grid_L, weighted=True)
                a1 = _arm(rc[1], wc[1], rt[1], wt[1], alpha / 2, **kw)
                a0 = _arm(rc[0], wc[0], rt[0], wt[0], alpha / 2, **kw)
                cov, fin, width = _itte_cov(a1, a0, rt[1], rt[0])
                per[key][d] = dict(
                    cov_uncond=float(np.mean(cov)),
                    finite_share=float(np.mean(fin)),
                    width_mean=float(np.nanmean(width)) if fin.any() else np.nan,
                )
    return n, per


# ================================================================= MODE: overlap (4.5.5)
def _run_overlap(args):
    scales = args.scales or [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    print(f"Phase 4.5.5 overlap diagnostics | target={1-args.alpha:.2f} | n={args.n} "
          f"| misspec={args.misspec} | reps={args.reps} | m_test={args.m_test}", flush=True)
    print(f"{'scale':>6} {'dim':>3} {'cov(uncond)':>12} {'cov(finite)':>12} {'inf%':>6} "
          f"{'ESS':>6} {'ESS/n_cal':>9} {'max_wt':>7} {'w_finite':>9}", flush=True)
    summary = []
    for scale in scales:
        cfg = _cfg(args, overlap=scale)
        rows = _parallel(_rep_finite, [(args.n, s, args.alpha, cfg) for s in _seeds(args)],
                         args.workers)
        acc = _accum(rows)
        for d in sorted(acc):
            m = acc[d]
            ess = 0.5 * (m['ess1'] + m['ess0'])
            ncal = 0.5 * (m['ncal1'] + m['ncal0'])
            maxw = 0.5 * (m['maxw1'] + m['maxw0'])
            inf = 100 * (1 - m['finite_share'])
            print(f"{scale:>6.1f} {d:>3} {m['cov_uncond']:>12.3f} {m['cov_finite']:>12.3f} "
                  f"{inf:>6.1f} {ess:>6.0f} {ess/ncal:>9.3f} {maxw:>7.3f} "
                  f"{m['width_mean']:>9.3f}", flush=True)
            summary.append((scale, d, inf, ess, maxw, m['width_mean'],
                            m['cov_uncond'], m['cov_finite']))
    print("\nESS = Kish effective calibration size; ESS/n_cal is the weighting "
          "efficiency (1.0 = uniform / randomized). max_wt = mean max normalized test "
          "weight -> the +inf atom fires when it exceeds ~alpha/2. inf% and width under "
          "scales 2.5/4.0 are the Phase-6.5 stabilization targets.", flush=True)
    return summary


# ----------------------------------------------------------------- shared helpers
def _cfg(args, **override):
    cfg = dict(resolution=args.resolution, n_basis=args.n_basis, noise=args.noise,
               prop_scale=args.overlap, misspec=args.misspec, m_test=args.m_test,
               calib_frac=args.calib_frac)
    if "overlap" in override:
        cfg["prop_scale"] = override.pop("overlap")
    cfg.update(override)
    return cfg


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


def _accum(rows):
    """rows: list of (n, {dim -> metric dict}) -> {dim -> mean metric dict}."""
    per = {}
    for _, d_map in rows:
        for d, v in d_map.items():
            per.setdefault(d, []).append(v)
    acc = {}
    for d, vs in per.items():
        keys = vs[0].keys()
        acc[d] = {k: float(np.nanmean([x[k] for x in vs])) for k in keys}
    return acc


_MODES = {
    "decompose": _run_decompose,
    "finite": _run_finite,
    "arms": _run_arms,
    "modulation": _run_modulation,
    "overlap": _run_overlap,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=list(_MODES) + ["all"], default="all")
    ap.add_argument("--n", type=int, default=500, help="train+calib size (default 500)")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-basis", type=int, default=5)
    ap.add_argument("--resolution", type=int, default=40)
    ap.add_argument("--resolutions", type=int, nargs="+", default=None,
                    help="grid resolutions for the modulation audit (default 20 40 80)")
    ap.add_argument("--noise", type=float, default=0.3)
    ap.add_argument("--overlap", type=float, default=1.0)
    ap.add_argument("--scales", type=float, nargs="+", default=None,
                    help="overlap scales for the overlap sweep (default 1 1.5 2 2.5 3 4)")
    ap.add_argument("--misspec", choices=["none", "pi", "mu", "both"], default="none")
    ap.add_argument("--m-test", type=int, default=200)
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260702)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="write one <mode>.txt per mode here (else stdout)")
    args = ap.parse_args()

    modes = list(_MODES) if args.mode == "all" else [args.mode]
    for mode in modes:
        runner = _MODES[mode]
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out = args.output_dir / f"{mode}.txt"
            with out.open("w") as f, contextlib.redirect_stdout(f):
                runner(args)
            print(f"===== {mode} =====")
            print(out.read_text())
        else:
            print(f"\n===== {mode} =====")
            runner(args)


if __name__ == "__main__":
    main()
