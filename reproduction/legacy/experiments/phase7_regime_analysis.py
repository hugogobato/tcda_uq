"""Phase 7A.5 regime analysis.

Compares where confidence bands and conformal prediction bands are reliable
across regimes the real datasets cannot dial: small n, large n, weak overlap,
heavy-tailed functional noise, and adaptive/neural nuisance fits.

The runner is CPU-parallel over Monte-Carlo replicates. BLAS threads are pinned
to one per worker, so ``--workers`` is the actual process parallelism.

Examples:
    # Quick syntax/runtime check.
    .venv/bin/python -u experiments/phase7_regime_analysis.py --preset smoke

    # Use seven workers on an 8-core CPU, leaving one core free.
    .venv/bin/python -u experiments/phase7_regime_analysis.py --preset mvp --workers 7

    # Use all 8 cores if the machine is dedicated to the run.
    .venv/bin/python -u experiments/phase7_regime_analysis.py --preset full --workers 8
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
    X = np.atleast_2d(np.asarray(X, dtype=float))
    extra = np.column_stack([X[:, 1] * X[:, 2], X[:, 0] * X[:, 2]])
    return np.hstack([X, extra])


@dataclass(frozen=True)
class Scenario:
    name: str
    n: int
    noise_scale: float = 0.3
    noise_df: float | None = None
    prop_scale: float = 1.0
    hetero_scale: float = 0.0
    nuisance: str = "functional"
    # level: "A" = functional-only DGP (silhouettes drawn directly); "B" = topological
    # DGP (real persistence -> silhouette pipeline in the loop, Phase 7A.2).
    level: str = "A"
    radius_tail_df: float | None = None
    note: str = ""


SCENARIOS = {
    # --- Level A: functional-only DGP (tests the UQ math) ---
    "small_n": Scenario("small_n", n=150, note="small sample"),
    "large_n": Scenario("large_n", n=1000, note="large sample"),
    "weak_overlap": Scenario("weak_overlap", n=500, prop_scale=4.0,
                             note="thin positivity"),
    "heavy_tail": Scenario("heavy_tail", n=500, noise_df=3.0,
                           note="Student-t functional noise"),
    "adaptive_rf": Scenario("adaptive_rf", n=500, nuisance="rf",
                            note="random-forest nuisances"),
    "neural_mlp": Scenario("neural_mlp", n=500, nuisance="mlp",
                           note="MLP nuisances"),
    "heterogeneous": Scenario("heterogeneous", n=500, hetero_scale=3.0,
                              note="X-dependent aleatoric spread"),
    # --- Level B: topological DGP (tests the topology-specific claims) ---
    "topological": Scenario("topological", n=500, level="B",
                            note="topological pipeline (Level B)"),
    "heavy_diagram_tail": Scenario("heavy_diagram_tail", n=500, level="B",
                                   radius_tail_df=3.0,
                                   note="heavy persistence-diagram tail (Level B)"),
}

PRESET_SCENARIOS = {
    "smoke": ["small_n", "heavy_tail"],
    "mvp": ["small_n", "large_n", "weak_overlap", "heavy_tail", "adaptive_rf",
            "topological"],
    "full": ["small_n", "large_n", "weak_overlap", "heavy_tail",
             "adaptive_rf", "neural_mlp", "heterogeneous",
             "topological", "heavy_diagram_tail"],
}

PRESET_DEFAULTS = {
    "smoke": dict(reps=2, n_boot=50, resolution=20, m_test=30, tate_nx=48, tate_nmc=2),
    "mvp": dict(reps=30, n_boot=300, resolution=40, m_test=120, tate_nx=256, tate_nmc=6),
    "full": dict(reps=100, n_boot=700, resolution=50, m_test=200, tate_nx=384, tate_nmc=8),
}


def _build_sim(scenario, args):
    """Build the Level-A tri-oracle or the Level-B topological DGP for a scenario."""
    if scenario.level == "B":
        from tcda_uq.datasets import TopologicalCausalSimulation

        return TopologicalCausalSimulation(
            n_cov=3,
            homology_dims=(0, 1),
            resolution=args.resolution,
            prop_scale=scenario.prop_scale,
            radius_tail_df=scenario.radius_tail_df,
            seed=0,
        )
    from tcda_uq.datasets import TriOracleSimulation

    return TriOracleSimulation(
        n_hom_dim=2,
        resolution=args.resolution,
        n_basis=args.n_basis,
        noise_scale=scenario.noise_scale,
        noise_df=scenario.noise_df,
        prop_scale=scenario.prop_scale,
        hetero_scale=scenario.hetero_scale,
        seed=0,
    )


def _scenario_offset(name):
    return sum((i + 1) * ord(ch) for i, ch in enumerate(name))


def _level_b_truth(task):
    """Monte-Carlo marginal TATE truth for a Level-B scenario (run once per scenario)."""
    scenario, args = task
    sim = _build_sim(scenario, args)
    truth = sim.true_tate_mc(n_x=args.tate_nx, n_mc=args.tate_nmc,
                             rng=991 + _scenario_offset(scenario.name))
    return scenario.name, truth


class _Nuisance:
    def __init__(self, mu_predict, prop_model, tseq):
        self._mu_predict = mu_predict
        self.prop_model = prop_model
        self.tseq = np.asarray(tseq, dtype=float)

    def predict_mu(self, X):
        return self._mu_predict(np.atleast_2d(np.asarray(X, dtype=float)))


def _fit_nuisance(train, tseq, n_basis, mode, seed, rf_trees, rf_depth, mlp_iter):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from tcda_uq.estimators.nuisance import (
        fit_functional_regression,
        fit_propensity,
        predict_functional_regression,
    )

    phi, A, X = train
    A = np.asarray(A)
    n_hom = phi.shape[-2]

    if mode == "functional":
        reg = fit_functional_regression(train, tseq, n_basis=n_basis)

        def mu_predict(Xe):
            return predict_functional_regression(reg, Xe, tseq)

        prop = fit_propensity(
            _interactions(X), A, LogisticRegression(max_iter=500)
        )
        return _Nuisance(mu_predict, prop, tseq), _interactions

    if mode == "rf":
        mu_models = []
        for d in range(n_hom):
            dim_models = []
            for a in (0, 1):
                mask = A == a
                model = RandomForestRegressor(
                    n_estimators=rf_trees,
                    max_depth=rf_depth,
                    min_samples_leaf=5,
                    n_jobs=1,
                    random_state=seed + 17 * d + a,
                )
                dim_models.append(model.fit(X[mask], phi[mask, d, :]))
            mu_models.append(tuple(dim_models))

        def mu_predict(Xe):
            return [(m0.predict(Xe), m1.predict(Xe)) for m0, m1 in mu_models]

        prop = RandomForestClassifier(
            n_estimators=rf_trees,
            max_depth=rf_depth,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=seed + 1009,
        ).fit(X, A)
        return _Nuisance(mu_predict, prop, tseq), None

    if mode == "mlp":
        mu_models = []
        for d in range(n_hom):
            dim_models = []
            for a in (0, 1):
                mask = A == a
                model = make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=mlp_iter,
                        early_stopping=True,
                        random_state=seed + 31 * d + a,
                    ),
                )
                dim_models.append(model.fit(X[mask], phi[mask, d, :]))
            mu_models.append(tuple(dim_models))

        def mu_predict(Xe):
            return [(m0.predict(Xe), m1.predict(Xe)) for m0, m1 in mu_models]

        prop = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=mlp_iter,
                early_stopping=True,
                random_state=seed + 2029,
            ),
        ).fit(X, A)
        return _Nuisance(mu_predict, prop, tseq), None

    raise ValueError(f"unknown nuisance mode: {mode}")


def _cross_fit(sample, tseq, n_basis, mode, rng, args):
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
        nuis, feat = _fit_nuisance(
            train,
            tseq,
            n_basis,
            mode,
            int(rng.integers(1 << 31)),
            args.rf_trees,
            args.rf_depth,
            args.mlp_iter,
        )
        X_test = X[test_idx]
        mu_hats = nuis.predict_mu(X_test)
        X_prop = feat(X_test) if feat is not None else X_test
        pi_hat = nuis.prop_model.predict_proba(X_prop)[:, 1]
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


def _one_rep(task):
    scenario, rep_id, args, level_b_truth = task
    from tcda_uq.metrics import covers_simultaneous, mean_width
    from tcda_uq.uq.asymptotic import eif_covariance, multiplier_bootstrap_band
    from tcda_uq.uq.conformal import ITTEConformal

    rng = np.random.default_rng(
        args.seed + 100000 * rep_id + _scenario_offset(scenario.name)
    )
    sim = _build_sim(scenario, args)
    sample = sim.sample(scenario.n, rng=rng)
    observed = sample.observed
    aipw, influence = _cross_fit(observed, sim.tseq, args.n_basis,
                                 scenario.nuisance, rng, args)

    rows = []
    # Level A has closed-form TATE truth; Level B uses the precomputed MC truth.
    truth = level_b_truth if level_b_truth is not None else sim.true_tate()
    tate_bands = {}
    for d in range(sim.n_hom_dim):
        cov_d = eif_covariance(influence[d])
        var_d = np.diag(cov_d)
        sd = np.sqrt(var_d)
        lo, hi = _pointwise_band(aipw[d], sd, scenario.n, args.alpha)
        rows.append(_metric_row(
            scenario, "TATE", "confidence", "pointwise", d,
            covers_simultaneous(lo, hi, truth[d]), mean_width(lo, hi), 0.0,
            args, extra=scenario.note,
        ))

        band = multiplier_bootstrap_band(
            influence[d],
            sim.tseq,
            aipw[d],
            alpha=args.alpha,
            n_boot=args.n_boot,
            variance=var_d,
            rng=int(rng.integers(1 << 31)),
        )
        tate_bands[d] = (band.lower, band.upper)
        rows.append(_metric_row(
            scenario, "TATE", "confidence", "multiplier", d,
            covers_simultaneous(band.lower, band.upper, truth[d]),
            mean_width(band.lower, band.upper), 0.0, args, extra=scenario.note,
        ))

    phi, A, X = observed
    perm = rng.permutation(scenario.n)
    n_cal = int(round(args.calib_frac * scenario.n))
    cal_idx = np.sort(perm[:n_cal])
    tr_idx = np.sort(perm[n_cal:])
    train = (phi[tr_idx], A[tr_idx], X[tr_idx])
    nuisance, feat = _fit_nuisance(
        train,
        sim.tseq,
        args.n_basis,
        scenario.nuisance,
        int(rng.integers(1 << 31)),
        args.rf_trees,
        args.rf_depth,
        args.mlp_iter,
    )
    model = ITTEConformal.fit(
        observed,
        sim.tseq,
        nuisance=nuisance,
        calib_index=cal_idx,
        propensity_feature_fn=feat,
        weighted=True,
    )
    test = sim.sample(args.m_test, rng=rng)
    for d in range(sim.n_hom_dim):
        lo, hi, _ = model.band_bounds(test.X, args.alpha, d=d)
        finite, width = _finite_width(lo, hi)
        covered = covers_simultaneous(lo, hi, test.oracle_itte[:, d, :])
        rows.append(_metric_row(
            scenario, "ITTE", "prediction", "conformal", d,
            float(covered.mean()), width, float((~finite).mean()), args,
            extra=scenario.note,
        ))

        blo, bhi = tate_bands[d]
        wrong_cov = covers_simultaneous(blo, bhi, test.oracle_itte[:, d, :])
        rows.append(_metric_row(
            scenario, "ITTE", "wrong confidence", "multiplier_as_itte", d,
            float(wrong_cov.mean()), mean_width(blo, bhi), 0.0, args,
            extra="TATE confidence band applied to individual draws",
        ))
    return rows


def _metric_row(scenario, estimand, uncertainty, method, dim, coverage,
                width, inf_frac, args, *, extra):
    return {
        "scenario": scenario.name,
        "level": scenario.level,
        "n": scenario.n,
        "nuisance": scenario.nuisance,
        "noise_df": "" if scenario.noise_df is None else scenario.noise_df,
        "prop_scale": scenario.prop_scale,
        "hetero_scale": scenario.hetero_scale,
        "estimand": estimand,
        "uncertainty": uncertainty,
        "method": method,
        "dim": dim,
        "coverage": float(coverage),
        "mean_width": float(width),
        "inf_pct": 100.0 * float(inf_frac),
        "extra": extra,
    }


def _aggregate(rows, reps):
    keys = [
        "scenario", "level", "n", "nuisance", "noise_df", "prop_scale",
        "hetero_scale", "estimand", "uncertainty", "method", "dim", "extra",
    ]
    acc = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        acc.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(acc.items()):
        row = dict(zip(keys, key))
        row["coverage"] = np.mean([r["coverage"] for r in group])
        finite_width = [r["mean_width"] for r in group if np.isfinite(r["mean_width"])]
        row["mean_width"] = np.mean(finite_width) if finite_width else np.inf
        row["inf_pct"] = np.mean([r["inf_pct"] for r in group])
        row["reps"] = len(group)
        row["target_reps"] = reps
        out.append(row)
    return out


def _format_table(rows, alpha):
    lines = [
        f"Phase 7A.5 regime analysis | target={1-alpha:.2f}",
        f"{'scenario':>15} {'n':>5} {'nuis':>10} {'estimand':>8} "
        f"{'uncertainty':>16} {'method':>18} {'dim':>3} {'cov':>7} "
        f"{'width':>10} {'inf%':>6} {'reps':>5}",
    ]
    for r in rows:
        width = "inf" if not np.isfinite(r["mean_width"]) else f"{r['mean_width']:.4f}"
        lines.append(
            f"{r['scenario']:>15} {int(r['n']):>5} {r['nuisance']:>10} "
            f"{r['estimand']:>8} {r['uncertainty']:>16} {r['method']:>18} "
            f"{int(r['dim']):>3} {r['coverage']:>7.3f} {width:>10} "
            f"{r['inf_pct']:>6.1f} {int(r['reps']):>5}"
        )
    return "\n".join(lines) + "\n"


def _write_outputs(rows, output_dir, alpha):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "regime_analysis.csv"
    md_path = output_dir / "regime_analysis.md"
    txt_path = output_dir / "regime_analysis.txt"

    fieldnames = [
        "scenario", "level", "n", "nuisance", "noise_df", "prop_scale",
        "hetero_scale", "estimand", "uncertainty", "method", "dim", "coverage",
        "mean_width", "inf_pct", "reps", "target_reps", "extra",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "| scenario | n | nuisance | estimand | uncertainty | method | dim | coverage | mean_width | inf% | reps |",
        "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        width = "inf" if not np.isfinite(r["mean_width"]) else f"{r['mean_width']:.4f}"
        md_lines.append(
            f"| {r['scenario']} | {r['n']} | {r['nuisance']} | {r['estimand']} | "
            f"{r['uncertainty']} | {r['method']} | {r['dim']} | "
            f"{r['coverage']:.3f} | {width} | {r['inf_pct']:.1f} | {r['reps']} |"
        )
    md_path.write_text("\n".join(md_lines) + "\n")
    txt_path.write_text(_format_table(rows, alpha=alpha))
    return csv_path, md_path, txt_path


def _resolve_args(args):
    defaults = PRESET_DEFAULTS[args.preset]
    for key in ("reps", "n_boot", "resolution", "m_test", "tate_nx", "tate_nmc"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
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
    raw_rows = []
    with Pool(processes=args.workers, initializer=_limit_threads) as pool:
        # Level-B scenarios need a Monte-Carlo TATE truth; compute it once each
        # (in parallel) and share with every replicate. Level-A truth is closed form.
        level_b = [s for s in args.scenario_objects if s.level == "B"]
        truth = {}
        if level_b:
            truth = dict(pool.map(_level_b_truth, [(s, args) for s in level_b]))
            print(f"# Level-B truth ready for {len(truth)} scenarios "
                  f"({time.time()-t0:.0f}s)", flush=True)

        tasks = [
            (scenario, rep_id, args, truth.get(scenario.name))
            for scenario in args.scenario_objects
            for rep_id in range(args.reps)
        ]
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
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--n-basis", type=int, default=5)
    ap.add_argument("--resolution", type=int, default=None)
    ap.add_argument("--m-test", type=int, default=None)
    ap.add_argument("--tate-nx", type=int, default=None,
                    help="MC covariate draws for the Level-B TATE truth")
    ap.add_argument("--tate-nmc", type=int, default=None,
                    help="MC point-cloud draws per x for the Level-B TATE truth")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260703)
    ap.add_argument("--rf-trees", type=int, default=80)
    ap.add_argument("--rf-depth", type=int, default=8)
    ap.add_argument("--mlp-iter", type=int, default=250)
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/out/phase7_regime"))
    args = _resolve_args(ap.parse_args())

    def _go():
        print(
            f"# Phase 7A.5 preset={args.preset} scenarios="
            f"{','.join(s.name for s in args.scenario_objects)} reps={args.reps} "
            f"workers={args.workers} n_boot={args.n_boot}",
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
        log_path = args.output_dir / "regime_analysis.log"
        with log_path.open("w") as f, contextlib.redirect_stdout(f):
            _go()
        print(log_path.read_text())
    else:
        _go()


if __name__ == "__main__":
    main()
