"""Phase 7A.2 / 7A.3 / 7A.4 -- Level-B (topological) UQ-stress sweep.

The Level-A suite (Phases 2-6) generates silhouette *curves* directly, which is
perfect for testing the UQ math but **bypasses the topology map** ``Y -> D -> phi``.
This runner closes that gap: it drives the *actual* persistence -> silhouette
pipeline through :class:`~tcda_uq.datasets.TopologicalCausalSimulation` (point
clouds whose loop count is controlled by ``(X, A)``) and fills the
estimand x uncertainty matrix at the **topological** level:

  * **TATE**  (confidence): pointwise + multiplier-bootstrap AIPW bands vs the
    Monte-Carlo marginal truth ``E_X E[phi^1 - phi^0 | X]``;
  * **CTATE-mean** (confidence): pointwise-in-x DR-learner bands at held-out ``x``
    vs the Monte-Carlo conditional truth ``E[phi^1 - phi^0 | X = x]``;
  * **ITTE** (prediction): composed weighted conformal bands vs the per-unit
    oracle ``delta_i = phi^1 - phi^0`` on a fresh test set, plus the *wrong*
    TATE-confidence-band-as-ITTE baseline (the confidence-vs-prediction contrast).

Stress knobs swept across the sweep (the ones the semi-synthetic TATE datasets
cannot dial):

  * ``n``                -- sample size;
  * overlap / positivity -- ``prop_scale`` (>1 pushes pi(X) toward {0,1});
  * **diagram tail-heaviness** -- ``radius_tail_df`` inflates a few loops so the
    persistence diagram (hence the silhouette) grows a heavy tail (Phase 7A.3,
    topology-specific).

Because the DGP model is fixed (``seed=0``), the ground-truth objects depend only
on the scenario, not the sample; they are Monte-Carlo estimated **once per
scenario** (in parallel) and shared with every replicate. The replicate loop is
CPU-parallel with one BLAS thread per worker, so ``--workers`` is the real
process parallelism.

Examples::

    # Quick syntax/runtime check.
    .venv/bin/python -u experiments/phase7_level_b.py --preset smoke --workers 2

    # Main Level-B evidence, leaving one core free on an 8-core CPU.
    .venv/bin/python -u experiments/phase7_level_b.py --preset mvp --workers 7

    # Full sweep on a dedicated machine.
    .venv/bin/python -u experiments/phase7_level_b.py --preset full --workers 8
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import norm


def _limit_threads():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")


_limit_threads()


def _interactions(X):
    """Propensity features [X, x1*x2, x0*x2] (matches the DGP logit)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    extra = np.column_stack([X[:, 1] * X[:, 2], X[:, 0] * X[:, 2]])
    return np.hstack([X, extra])


def _const_feat(X):
    """Mis-specified propensity feature map: a single constant column (ignores X)."""
    return np.zeros((np.atleast_2d(np.asarray(X)).shape[0], 1))


# DGP defaults on TopologicalCausalSimulation (kept in sync so a scenario that
# does not touch a knob reproduces the baseline exactly).
_DEF_NOISE = 0.08
_DEF_SLOPE = 1.2


@dataclass(frozen=True)
class Scenario:
    name: str
    n: int
    prop_scale: float = 1.0
    radius_tail_df: float | None = None
    # nuisance misspecification (doubly-robust stress): degrade pi, mu, or both.
    misspec: str = "none"
    # signal-to-noise knob: point-cloud jitter (higher => noisier diagrams => lower SNR).
    noise: float = _DEF_NOISE
    # effect-heterogeneity-in-X knob: strength of the X-tilt of the loop count.
    loop_slope: float = _DEF_SLOPE
    note: str = ""

    @property
    def tail_label(self) -> str:
        return "off" if self.radius_tail_df is None else f"t{self.radius_tail_df:g}"


# Level-B scenarios spanning the 7A.3 knobs: sample size, overlap/positivity,
# diagram tail-heaviness, nuisance misspecification (pi/mu), signal-to-noise, and
# effect heterogeneity in X.
SCENARIOS = {
    "base": Scenario("base", n=500, note="baseline topological DGP"),
    "small_n": Scenario("small_n", n=200, note="small sample"),
    "large_n": Scenario("large_n", n=1000, note="large sample"),
    "weak_overlap": Scenario("weak_overlap", n=500, prop_scale=3.0,
                             note="thin positivity"),
    "very_weak_overlap": Scenario("very_weak_overlap", n=500, prop_scale=5.0,
                                  note="very thin positivity"),
    "heavy_tail": Scenario("heavy_tail", n=500, radius_tail_df=3.0,
                           note="heavy persistence-diagram tail"),
    "very_heavy_tail": Scenario("very_heavy_tail", n=500, radius_tail_df=2.5,
                                note="very heavy diagram tail"),
    # nuisance misspecification (doubly-robust stress) at n=800
    "misspec_pi": Scenario("misspec_pi", n=800, misspec="pi",
                           note="propensity misspecified (constant)"),
    "misspec_mu": Scenario("misspec_mu", n=800, misspec="mu",
                           note="outcome model misspecified (constant)"),
    "misspec_both": Scenario("misspec_both", n=800, misspec="both",
                             note="both nuisances misspecified"),
    # signal-to-noise: higher jitter blurs the loops (lower SNR)
    "low_snr": Scenario("low_snr", n=500, noise=0.22, note="high point-cloud jitter (low SNR)"),
    "high_snr": Scenario("high_snr", n=500, noise=0.03, note="low point-cloud jitter (high SNR)"),
    # effect heterogeneity in X: strength of the loop-count X-tilt (drives CTATE)
    "strong_hetero": Scenario("strong_hetero", n=500, loop_slope=2.6,
                              note="strong X-heterogeneity of the effect"),
    "weak_hetero": Scenario("weak_hetero", n=500, loop_slope=0.4,
                            note="weak X-heterogeneity of the effect"),
}

PRESET_SCENARIOS = {
    "smoke": ["base", "misspec_pi"],
    "mvp": ["small_n", "base", "weak_overlap", "heavy_tail"],
    "full": ["small_n", "base", "large_n", "weak_overlap", "very_weak_overlap",
             "heavy_tail", "very_heavy_tail",
             "misspec_pi", "misspec_mu", "misspec_both",
             "low_snr", "high_snr", "strong_hetero", "weak_hetero"],
}

PRESET_DEFAULTS = {
    "smoke": dict(reps=2, n_boot=50, resolution=25, m_test=30,
                  tate_nx=48, tate_nmc=2, n_eval=8, ctate_nmc=2, ctate_boot=100),
    "mvp": dict(reps=24, n_boot=300, resolution=40, m_test=120,
                tate_nx=256, tate_nmc=6, n_eval=32, ctate_nmc=8, ctate_boot=300),
    "full": dict(reps=80, n_boot=600, resolution=45, m_test=200,
                 tate_nx=384, tate_nmc=8, n_eval=48, ctate_nmc=12, ctate_boot=500),
}


def _build_sim(scenario: Scenario, args):
    from tcda_uq.datasets import TopologicalCausalSimulation

    return TopologicalCausalSimulation(
        n_cov=3,
        homology_dims=(0, 1),
        resolution=args.resolution,
        prop_scale=scenario.prop_scale,
        radius_tail_df=scenario.radius_tail_df,
        noise=scenario.noise,
        loop_slope=scenario.loop_slope,
        pts_per_loop=args.pts_per_loop,
        background=args.background,
        seed=0,
    )


class _Nuisance:
    """Minimal nuisance object for :class:`ITTEConformal` (functional outcome + propensity)."""

    def __init__(self, mu_predict, prop_model, tseq):
        self._mu_predict = mu_predict
        self.prop_model = prop_model
        self.tseq = np.asarray(tseq, dtype=float)

    def predict_mu(self, X):
        return self._mu_predict(np.atleast_2d(np.asarray(X, dtype=float)))


def _fit_nuisance(train, tseq, n_basis, misspec="none"):
    """Functional outcome regression + logistic propensity on a training split.

    ``misspec in {none, pi, mu, both}`` degrades the corresponding nuisance to a
    constant (ignores X) for the doubly-robust stress test. Returns
    ``(nuisance, pi_feat)`` -- the propensity feature map must also be used at
    predict time, so it is returned alongside the fitted object.
    """
    from sklearn.linear_model import LogisticRegression
    from tcda_uq.estimators.nuisance import (
        fit_functional_regression,
        fit_propensity,
        predict_functional_regression,
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
            m = np.atleast_2d(Xe).shape[0]
            return [(np.tile(m0[d], (m, 1)), np.tile(m1[d], (m, 1))) for d in range(n_hom)]
    else:
        reg = fit_functional_regression(train, tseq, n_basis=n_basis)

        def mu_predict(Xe):
            return predict_functional_regression(reg, Xe, tseq)

    pi_feat = _const_feat if bad_pi else _interactions
    prop = fit_propensity(pi_feat(X), A, LogisticRegression(max_iter=500))
    return _Nuisance(mu_predict, prop, tseq), pi_feat


def _cross_fit(sample, tseq, n_basis, rng, misspec="none"):
    """Two-fold cross-fitted AIPW scores + centered influence per homology dim."""
    from sklearn.model_selection import StratifiedKFold
    from tcda_uq.estimators.aipw import aipw_scores

    phi, A, X = sample
    A = np.asarray(A)
    n_hom = phi.shape[-2]
    splitter = StratifiedKFold(n_splits=2, shuffle=True,
                               random_state=int(rng.integers(1 << 31)))
    scores_parts = [[] for _ in range(n_hom)]
    for train_idx, test_idx in splitter.split(X, A):
        train = (phi[train_idx], A[train_idx], X[train_idx])
        nuis, pi_feat = _fit_nuisance(train, tseq, n_basis, misspec)
        X_test = X[test_idx]
        mu_hats = nuis.predict_mu(X_test)
        pi_hat = nuis.prop_model.predict_proba(pi_feat(X_test))[:, 1]
        s = aipw_scores(pi_hat, mu_hats, (phi[test_idx], A[test_idx], X_test))
        for d in range(n_hom):
            scores_parts[d].append(s[d])
    scores = [np.concatenate(parts, axis=0) for parts in scores_parts]
    aipw = [s.mean(axis=0) for s in scores]
    influence = [s - s.mean(axis=0, keepdims=True) for s in scores]
    return aipw, influence


def _pointwise_band(est, sd, n, alpha):
    half = norm.ppf(1 - alpha / 2) * sd / np.sqrt(n)
    return est - half, est + half


def _finite_width(lo, hi):
    finite = np.isfinite(lo).all(axis=1) & np.isfinite(hi).all(axis=1)
    width = np.where(finite, (hi - lo).mean(axis=1), np.nan)
    return finite, float(np.nanmean(width)) if finite.any() else np.inf


def _scenario_offset(name):
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name))


# ------------------------------------------------------------------ ground truth
def _scenario_truth(task):
    """Monte-Carlo ground truth for one scenario (run once, shared by all reps)."""
    scenario, args = task
    sim = _build_sim(scenario, args)
    base = 991 + _scenario_offset(scenario.name)
    tate = sim.true_tate_mc(n_x=args.tate_nx, n_mc=args.tate_nmc, rng=base)
    X_eval = sim._sample_covariates(args.n_eval, np.random.default_rng(base + 1))
    ctate = sim.true_ctate_mc(X_eval, n_mc=args.ctate_nmc, rng=base + 2)
    return scenario.name, (tate, X_eval, ctate)


# ------------------------------------------------------------------ one replicate
def _one_rep(task):
    scenario, rep_id, args, truth = task
    from tcda_uq.estimators import CTATEDRLearner
    from tcda_uq.metrics import covers_simultaneous, mean_width
    from tcda_uq.uq.asymptotic import (
        ctate_confidence_band,
        eif_covariance,
        multiplier_bootstrap_band,
    )
    from tcda_uq.uq.conformal import ITTEConformal

    tate_truth, X_eval, ctate_truth = truth
    sim = _build_sim(scenario, args)
    n_hom = sim.n_hom_dim
    rng = np.random.default_rng(args.seed + 100000 * rep_id + _scenario_offset(scenario.name))

    sample = sim.sample(scenario.n, rng=rng)
    observed = sample.observed

    rows = []

    # ---- TATE (confidence): pointwise + multiplier bootstrap vs MC marginal truth ----
    aipw, influence = _cross_fit(observed, sim.tseq, args.n_basis, rng, scenario.misspec)
    tate_bands = {}
    for d in range(n_hom):
        var_d = np.diag(eif_covariance(influence[d]))
        sd = np.sqrt(var_d)
        lo, hi = _pointwise_band(aipw[d], sd, scenario.n, args.alpha)
        rows.append(_row(scenario, "TATE", "confidence", "pointwise", d,
                         float(covers_simultaneous(lo, hi, tate_truth[d])),
                         mean_width(lo, hi), 0.0, scenario.note))

        band = multiplier_bootstrap_band(
            influence[d], sim.tseq, aipw[d], alpha=args.alpha,
            n_boot=args.n_boot, variance=var_d, rng=int(rng.integers(1 << 31)),
        )
        tate_bands[d] = (band.lower, band.upper)
        rows.append(_row(scenario, "TATE", "confidence", "multiplier", d,
                         float(covers_simultaneous(band.lower, band.upper, tate_truth[d])),
                         mean_width(band.lower, band.upper), 0.0, scenario.note))

    # ---- CTATE-mean (confidence): pointwise-in-x DR band at held-out x ----
    # (skipped under nuisance misspecification -- that is a DR test for the AIPW /
    # conformal objects, not for the CTATE second stage.)
    if args.ctate and scenario.misspec == "none":
        from sklearn.linear_model import LogisticRegression

        learner = CTATEDRLearner(n_basis=args.n_basis).fit(
            observed, sim.tseq, n_splits=2,
            propensity_estimator=LogisticRegression(max_iter=500),
            propensity_feature_fn=_interactions,
            random_state=int(rng.integers(1 << 31)),
        )
        for d in range(n_hom):
            covs, widths = [], []
            for j, x in enumerate(X_eval):
                cb = ctate_confidence_band(
                    learner, x, d, alpha=args.alpha, n_boot=args.ctate_boot,
                    rng=int(rng.integers(1 << 31)),
                )
                covs.append(bool(covers_simultaneous(cb.lower, cb.upper, ctate_truth[j, d])))
                widths.append(mean_width(cb.lower, cb.upper))
            rows.append(_row(scenario, "CTATE_mean", "confidence", "ctate_dr", d,
                             float(np.mean(covs)), float(np.mean(widths)), 0.0,
                             f"held-out x; {scenario.note}"))

    # ---- ITTE (prediction): composed weighted conformal vs per-unit oracle ----
    phi, A, X = observed
    perm = rng.permutation(scenario.n)
    n_cal = int(round(args.calib_frac * scenario.n))
    cal_idx = np.sort(perm[:n_cal])
    tr_idx = np.sort(perm[n_cal:])
    nuisance, pi_feat = _fit_nuisance((phi[tr_idx], A[tr_idx], X[tr_idx]),
                                      sim.tseq, args.n_basis, scenario.misspec)
    model = ITTEConformal.fit(
        observed, sim.tseq, nuisance=nuisance, calib_index=cal_idx,
        propensity_feature_fn=pi_feat, weighted=True,
    )
    test = sim.sample(args.m_test, rng=rng)
    for d in range(n_hom):
        lo, hi, _ = model.band_bounds(test.X, args.alpha, d=d)
        finite, width = _finite_width(lo, hi)
        covered = covers_simultaneous(lo, hi, test.oracle_itte[:, d, :])
        rows.append(_row(scenario, "ITTE", "prediction", "conformal", d,
                         float(covered.mean()), width, float((~finite).mean()),
                         scenario.note))

        blo, bhi = tate_bands[d]
        wrong = covers_simultaneous(blo, bhi, test.oracle_itte[:, d, :])
        rows.append(_row(scenario, "ITTE", "wrong_confidence", "multiplier_as_itte", d,
                         float(wrong.mean()), mean_width(blo, bhi), 0.0,
                         "TATE band applied to individual draws"))
    return rows


def _row(scenario, estimand, uncertainty, method, dim, coverage, width, inf_frac, note):
    return {
        "scenario": scenario.name,
        "n": scenario.n,
        "prop_scale": scenario.prop_scale,
        "tail": scenario.tail_label,
        "radius_tail_df": "" if scenario.radius_tail_df is None else scenario.radius_tail_df,
        "misspec": scenario.misspec,
        "noise": scenario.noise,
        "loop_slope": scenario.loop_slope,
        "estimand": estimand,
        "uncertainty": uncertainty,
        "method": method,
        "dim": dim,
        "coverage": float(coverage),
        "mean_width": float(width),
        "inf_pct": 100.0 * float(inf_frac),
        "note": note,
    }


# ------------------------------------------------------------------ aggregation
_KEYS = ["scenario", "n", "prop_scale", "tail", "radius_tail_df", "misspec",
         "noise", "loop_slope", "estimand", "uncertainty", "method", "dim", "note"]


def _aggregate(rows, reps):
    acc = {}
    for row in rows:
        acc.setdefault(tuple(row[k] for k in _KEYS), []).append(row)
    out = []
    for key, group in sorted(acc.items(), key=_sort_key):
        row = dict(zip(_KEYS, key))
        row["coverage"] = float(np.mean([r["coverage"] for r in group]))
        finite = [r["mean_width"] for r in group if np.isfinite(r["mean_width"])]
        row["mean_width"] = float(np.mean(finite)) if finite else np.inf
        row["inf_pct"] = float(np.mean([r["inf_pct"] for r in group]))
        row["reps"] = len(group)
        row["target_reps"] = reps
        out.append(row)
    return out


_ESTIMAND_ORDER = {"TATE": 0, "CTATE_mean": 1, "ITTE": 2}


def _sort_key(item):
    key = dict(zip(_KEYS, item[0]))
    return (_ESTIMAND_ORDER.get(key["estimand"], 9), key["scenario"],
            key["uncertainty"], key["method"], int(key["dim"]))


# ------------------------------------------------------------------ output
def _format_table(rows, alpha):
    lines = [
        f"Phase 7A.2/7A.4 Level-B topological UQ sweep | target={1-alpha:.2f}",
        f"{'scenario':>17} {'n':>5} {'tail':>5} {'estimand':>11} "
        f"{'uncertainty':>16} {'method':>18} {'dim':>3} {'cov':>7} "
        f"{'width':>10} {'inf%':>6} {'reps':>5}",
    ]
    for r in rows:
        width = "inf" if not np.isfinite(r["mean_width"]) else f"{r['mean_width']:.4f}"
        lines.append(
            f"{r['scenario']:>17} {int(r['n']):>5} {r['tail']:>5} {r['estimand']:>11} "
            f"{r['uncertainty']:>16} {r['method']:>18} {int(r['dim']):>3} "
            f"{r['coverage']:>7.3f} {width:>10} {r['inf_pct']:>6.1f} {int(r['reps']):>5}"
        )
    return "\n".join(lines) + "\n"


def _row_line(r):
    """One machine-parseable ROW line for the master-table builder."""
    bits = [r["scenario"], f"overlap={r['prop_scale']:g}", f"tail={r['tail']}"]
    if r["misspec"] != "none":
        bits.append(f"misspec={r['misspec']}")
    if abs(float(r["noise"]) - _DEF_NOISE) > 1e-9:
        bits.append(f"noise={float(r['noise']):g}")
    if abs(float(r["loop_slope"]) - _DEF_SLOPE) > 1e-9:
        bits.append(f"slope={float(r['loop_slope']):g}")
    scenario = ";".join(bits)  # no spaces -> one token for the ROW grammar
    width = "inf" if not np.isfinite(r["mean_width"]) else f"{r['mean_width']:.4f}"
    fields = {
        "level": "7A_Level_B",
        "estimand": r["estimand"],
        "uncertainty": r["uncertainty"],
        "method": r["method"],
        "n": r["n"],
        "scenario": scenario,
        "dim": r["dim"],
        "coverage": f"{r['coverage']:.4f}",
        "mean_width": width,
        "inf_pct": f"{r['inf_pct']:.1f}",
        "extra": f"scn={r['scenario']};reps={r['reps']}",
    }
    return "ROW " + " ".join(f"{k}={v}" for k, v in fields.items())


def _write_outputs(rows, output_dir, alpha):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "level_b.csv"
    md_path = output_dir / "level_b.md"
    txt_path = output_dir / "level_b.txt"

    fieldnames = _KEYS + ["coverage", "mean_width", "inf_pct", "reps", "target_reps"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| scenario | n | tail | estimand | uncertainty | method | dim | coverage | mean_width | inf% | reps |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        width = "inf" if not np.isfinite(r["mean_width"]) else f"{r['mean_width']:.4f}"
        md_lines.append(
            f"| {r['scenario']} | {r['n']} | {r['tail']} | {r['estimand']} | "
            f"{r['uncertainty']} | {r['method']} | {r['dim']} | "
            f"{r['coverage']:.3f} | {width} | {r['inf_pct']:.1f} | {r['reps']} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n")

    # human table + machine-parseable ROW lines in the same .txt
    txt = _format_table(rows, alpha) + "\n" + "\n".join(_row_line(r) for r in rows) + "\n"
    txt_path.write_text(txt)
    return csv_path, md_path, txt_path


# ------------------------------------------------------------------ driver
def _resolve_args(args):
    defaults = PRESET_DEFAULTS[args.preset]
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)
    if args.workers is None:
        args.workers = max(1, min(7, os.cpu_count() or 1))
    names = args.scenarios or PRESET_SCENARIOS[args.preset]
    missing = [name for name in names if name not in SCENARIOS]
    if missing:
        raise SystemExit(f"unknown scenario(s): {', '.join(missing)}")
    args.scenario_objects = [SCENARIOS[name] for name in names]
    return args


def run(args):
    from multiprocessing import Pool

    t0 = time.time()
    with Pool(processes=args.workers, initializer=_limit_threads) as pool:
        # 1) ground truth once per scenario (parallel over scenarios)
        truth = dict(pool.map(_scenario_truth,
                              [(s, args) for s in args.scenario_objects]))
        print(f"# truth ready for {len(truth)} scenarios ({time.time()-t0:.0f}s)", flush=True)

        # 2) replicates (parallel over scenario x rep)
        tasks = [
            (scenario, rep_id, args, truth[scenario.name])
            for scenario in args.scenario_objects
            for rep_id in range(args.reps)
        ]
        raw_rows = []
        for i, rows in enumerate(pool.imap_unordered(_one_rep, tasks, chunksize=1), 1):
            raw_rows.extend(rows)
            if i % max(1, len(tasks) // 20) == 0:
                print(f"# {i}/{len(tasks)} scenario-reps done ({time.time()-t0:.0f}s)",
                      flush=True)
    return _aggregate(raw_rows, args.reps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["smoke", "mvp", "full"], default="mvp")
    ap.add_argument("--scenarios", nargs="+")
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n-basis", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--m-test", type=int, default=None)
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--pts-per-loop", type=int, default=36)
    ap.add_argument("--background", type=int, default=6)
    ap.add_argument("--tate-nx", type=int, default=None, help="MC covariate draws for TATE truth")
    ap.add_argument("--tate-nmc", type=int, default=None, help="MC point-cloud draws per x for TATE truth")
    ap.add_argument("--n-eval", type=int, default=None, help="held-out x count for CTATE truth")
    ap.add_argument("--ctate-nmc", type=int, default=None, help="MC point-cloud draws per x for CTATE truth")
    ap.add_argument("--ctate-boot", type=int, default=None, help="bootstrap draws for CTATE bands")
    ap.add_argument("--no-ctate", dest="ctate", action="store_false",
                    help="skip the (slower) CTATE-mean rows")
    ap.set_defaults(ctate=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260703)
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/out/phase7_level_b"))
    args = _resolve_args(ap.parse_args())

    def _go():
        print(
            f"# Phase 7A.2 Level-B preset={args.preset} scenarios="
            f"{','.join(s.name for s in args.scenario_objects)} reps={args.reps} "
            f"workers={args.workers} n_boot={args.n_boot} ctate={args.ctate}",
            flush=True,
        )
        rows = run(args)
        print(_format_table(rows, args.alpha), end="", flush=True)
        csv_path, md_path, txt_path = _write_outputs(rows, args.output_dir, args.alpha)
        print(f"# wrote {csv_path}")
        print(f"# wrote {md_path}")
        print(f"# wrote {txt_path}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        log_path = args.output_dir / "level_b.log"
        with log_path.open("w") as f, contextlib.redirect_stdout(f):
            _go()
        print(log_path.read_text())
    else:
        _go()


if __name__ == "__main__":
    main()
