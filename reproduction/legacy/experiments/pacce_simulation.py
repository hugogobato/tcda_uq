"""PACCE-matched simulation study (Phase P2 / WP-P2.1 + WP-P2.2, gate G3).

Simulates known-truth longitudinal target-lesion trials with PACCE-like lesion
counts, visit schedule, strata, and observation defects; runs the frozen
pipeline (adapter geometry -> AIPW/multiplier-bootstrap TATE, DR-learner CTATE,
composed conformal ITTE) plus conventional SLD comparators; evaluates
calibration, power, and incremental value of topology.

Scenarios
---------
null             identical arms (no effect anywhere)
sld_only         treated shrink more on average (same lesion-level dispersion)
topo_favorable   treated response is heterogeneous across lesions; mean SLD
                 change matched to control (topology moves, SLD mean does not)
topo_null        all log-sizes translated by an arm-specific constant at ALL
                 times => persistence exactly invariant BY CONSTRUCTION while
                 absolute Delta-SLD differs (SLD-favorable falsifier)
nuisance_misp    sld_only + nonlinear/interaction covariate effects (linear nuisances)
irregular_visits wide week-12 timing jitter + unscheduled mid-window assessments
                 (the frozen visit-label rule drops the latter)
dropout_arm      week-12 observation probability differs by assigned arm and
                 baseline burden (observed-trajectory target != full-population)

Truths: the oracle ITTE delta_i = phi_i^1 - phi_i^0 is computed exactly per unit
(both potential outcomes are simulated), so the eligible-population TATE truth,
the SLD contrast, and large-Monte-Carlo CTATE profile curves are all known.

Usage
-----
python experiments/pacce_simulation.py --profile                     # smoke
python experiments/pacce_simulation.py --scenarios null,sld_only \
       --reps 500 --n 842 --shard 0 --num_shards 14                  # shard
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from tcda_uq.datasets.pacce import SPEC_V1, compute_phi
from tcda_uq.estimators import CTATEDRLearner, cross_fit
from tcda_uq.uq.asymptotic import ctate_confidence_band, multiplier_bootstrap_band
from tcda_uq.uq.conformal import ITTEConformal

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pacce_simulation"
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["null", "sld_only", "topo_favorable", "topo_null",
             "nuisance_misp", "irregular_visits", "dropout_arm"]

BASE_COLS = ["chemo_iri", "age10", "male", "ecog_ge1", "n_target_bl",
             "n_metsites_bl", "kras_wt", "kras_mut", "kras_missing",
             "ldh_high_bl"]

PROFILES = {
    "oxali_high_burden": dict(chemo_iri=0, age10=6.0, male=1, ecog_ge1=0,
                              n_target_bl=5, n_metsites_bl=2, kras_wt=1,
                              kras_mut=0, kras_missing=0, ldh_high_bl=0),
    "oxali_low_burden": dict(chemo_iri=0, age10=6.0, male=0, ecog_ge1=1,
                             n_target_bl=2, n_metsites_bl=1, kras_wt=0,
                             kras_mut=1, kras_missing=0, ldh_high_bl=0),
    "iri_median_burden": dict(chemo_iri=1, age10=7.0, male=1, ecog_ge1=0,
                              n_target_bl=3, n_metsites_bl=2, kras_wt=1,
                              kras_mut=0, kras_missing=0, ldh_high_bl=0),
}


def frozen_standardization() -> dict:
    with open(ROOT / "results" / "pacce" / "pacce_smoke.json") as f:
        return json.load(f)["standardization"]


# ==========================================================================
# generator
# ==========================================================================
class PacceSimulator:
    """Known-truth PACCE-like trial generator."""

    def __init__(self, scenario: str, n: int = 842, seed: int = 0,
                 std_consts: dict | None = None, resolution: int = 60,
                 hom: int = 0):
        self.scenario = scenario
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.resolution = resolution
        self.hom = hom                       # homology axis for the outcome
        self.spec = json.loads(json.dumps(SPEC_V1))
        self.spec["silhouette"]["resolution"] = resolution
        if std_consts is not None:
            self.spec["standardization"] = std_consts

    # ------------------------------------------------------------- covariates
    def draw_covariates(self, n=None):
        rng = self.rng
        n = self.n if n is None else n
        X = pd.DataFrame({
            "chemo_iri": (rng.random(n) < 0.20).astype(float),
            "age10": rng.normal(6.1, 1.2, n),
            "male": (rng.random(n) < 0.57).astype(float),
            "ecog_ge1": (rng.random(n) < 0.41).astype(float),
            "n_target_bl": np.clip(rng.poisson(2.1, n) + 1, 1, 8).astype(float),
            "n_metsites_bl": np.clip(rng.poisson(0.8, n) + 1, 1, 5).astype(float),
            "ldh_high_bl": (rng.random(n) < 0.20).astype(float),
            "kras_wt": np.zeros(n), "kras_mut": np.zeros(n),
            "kras_missing": np.zeros(n),
        })
        km = rng.random(n)
        X["kras_wt"] = (km < 0.49).astype(float)
        X["kras_mut"] = ((km >= 0.49) & (km < 0.83)).astype(float)
        X["kras_missing"] = (km >= 0.83).astype(float)
        return X[BASE_COLS]

    # ------------------------------------------------------------------ visits
    def draw_visit_days(self, m):
        rng = self.rng
        u0 = np.clip(rng.normal(-6, 2.5, m), -20, -0.01)
        u1 = (rng.uniform(41, 117, m) if self.scenario == "irregular_visits"
              else np.clip(rng.normal(81, 7, m), 41, 117))
        return u0, u1

    # --------------------------------------------------------- lesion dynamics
    def _unit_latents(self, X):
        """Shared unit/lesion latents (coupled potential outcomes)."""
        rng = self.rng
        m = len(X)
        J = X["n_target_bl"].to_numpy(int)
        jmax = int(J.max())
        lat = dict(
            m_i=rng.normal(3.3, 0.35, m),
            c_ij=rng.normal(0, 0.18, (m, jmax)),
            shrink_i=0.22 + rng.normal(0, 0.08, m),
            e_ij=rng.normal(0, 1.0, (m, jmax)),        # scaled per scenario
            big_ij=rng.random((m, jmax)) < 0.5,
            extra=(rng.random(m) < 0.30),
            u_extra=rng.uniform(20, 60, (m, jmax)),
            e_extra=rng.normal(0, 0.05, (m, jmax)),
        )
        return lat

    def _lesion_cloud(self, X, a: int, lat):
        """Potential-outcome clouds under arm ``a`` from shared latents."""
        rng = self.rng
        sc = self.scenario
        m = len(X)
        J = X["n_target_bl"].to_numpy(int)
        u0v, u1v = self._u0, self._u1
        age, iri, ntb = (X["age10"].to_numpy(float),
                         X["chemo_iri"].to_numpy(float),
                         X["n_target_bl"].to_numpy(float))
        clouds, meta = {}, []
        for i in range(m):
            j = int(J[i])
            z0 = lat["m_i"][i] + lat["c_ij"][i, :j]
            e = lat["e_ij"][i, :j] * (0.06 if not (sc == "topo_favorable" and a == 1)
                                      else 0.16)
            if sc in ("sld_only", "nuisance_misp"):
                shr = np.full(j, lat["shrink_i"][i] + 0.15 * a)
                z1 = z0 - shr + e
                if sc == "nuisance_misp":
                    tilt = 0.10 * np.sin(age[i]) \
                        + 0.08 * (iri[i] * (ntb[i] - 3) / 2.0)
                    z1 = z1 - a * tilt - (1 - a) * 0.05 * np.cos(age[i])
            elif sc == "topo_favorable":
                # treated wk-12 sizes are a mean-preserving SPREAD (mm scale) of
                # the control wk-12 sizes: sum(w12) identical by construction,
                # so the SLD contrast is exactly null while geometry moves.
                mm0 = np.exp(z0)
                if a == 1:
                    w12c = mm0 * np.exp(-lat["shrink_i"][i])
                    w12t = w12c.copy()
                    for p in range(j // 2):
                        ia, ib = 2 * p, 2 * p + 1
                        cbar = 0.5 * (w12c[ia] + w12c[ib])
                        d = 0.45 * cbar
                        w12t[ia], w12t[ib] = max(cbar - d, 1.0), cbar + d
                    z1 = np.log(np.maximum(w12t, 5.0))
                else:
                    z1 = z0 - lat["shrink_i"][i]
                    z1 = np.log(np.maximum(np.exp(z1), 5.0))
            elif sc == "topo_null":
                shift = 0.15 * a                       # whole-cloud translation
                z0 = z0 + shift                        # => persistence invariant
                z1 = z0 - lat["shrink_i"][i] + e
            else:                                      # null / irregular / dropout
                z1 = z0 - lat["shrink_i"][i] + e

            pts_u = np.concatenate([np.full(j, u0v[i]), np.full(j, u1v[i])]) / 117.0
            sizes = np.exp(np.concatenate([z0, z1]))
            pts_z = np.log(np.maximum(sizes, 5.0))
            if sc == "irregular_visits" and lat["extra"][i]:
                zu = z0 - 0.10 + lat["e_extra"][i, :j]
                pts_u = np.concatenate([pts_u, lat["u_extra"][i, :j] / 117.0])
                pts_z = np.concatenate([pts_z, np.log(np.maximum(np.exp(zu), 5.0))])
            clouds[i] = np.column_stack([pts_u, pts_z])
            meta.append((float(np.exp(z0).sum()), float(np.exp(z1).sum())))
        meta = pd.DataFrame(meta, columns=["bl_sld", "w12_sld"])
        return clouds, meta

    # -------------------------------------------------------------- eligibility
    def draw_eligibility(self, X):
        rng = self.rng
        m = len(X)
        burden = (X["n_target_bl"].to_numpy(float) - 3.0) / 2.0
        p0 = 0.75 - 0.05 * burden
        p1 = p0.copy()
        if self.scenario == "dropout_arm":
            p1 = 0.55 - 0.08 * burden
        return rng.random(m) < p0, rng.random(m) < p1

    # -------------------------------------------------------------------- draw
    def one_draw(self, n=None, override_row=None):
        """Simulate both potential outcomes + observation for ``n`` units."""
        rng = self.rng
        X = self.draw_covariates(n)
        if override_row is not None:                   # fixed-profile simulation
            for c in BASE_COLS:
                X[c] = float(override_row[c])
        self._u0, self._u1 = self.draw_visit_days(len(X))
        lat = self._unit_latents(X)

        phia, metas = {}, {}
        for a in (0, 1):
            clouds_a, meta_a = self._lesion_cloud(X, a, lat)
            pdict = compute_phi(clouds_a, dict(self.spec))[0]
            phia[a] = np.stack([pdict[s] for s in sorted(pdict)])[:, self.hom, :]
            metas[a] = meta_a
        delta = phia[1] - phia[0]

        A = rng.binomial(1, 0.5, len(X)).astype(float)
        R0, R1 = self.draw_eligibility(X)
        R = np.where(A == 1, R1, R0).astype(bool)
        # observed silhouette with a singleton homology axis: [n, 1, res]
        phi_obs = np.where(A[:, None] == 1.0, phia[1], phia[0])[:, None, :]

        bl1, w121 = metas[1]["bl_sld"].values, metas[1]["w12_sld"].values
        bl0, w120 = metas[0]["bl_sld"].values, metas[0]["w12_sld"].values
        self._u0 = self._u1 = None
        pct1 = 100 * (w121 - bl1) / bl1
        pct0 = 100 * (w120 - bl0) / bl0
        return dict(
            X=X, A=A, R=R, delta=delta, phi_obs=phi_obs,
            pct1=pct1, pct0=pct0,
            dsld_units=(w121 - bl1) - (w120 - bl0),      # mm contrast per unit
            dsld_obs=np.where(A == 1, (w121 - bl1), (w120 - bl0)),
            bl_sld=bl1,                                   # arm-1 baseline burden
        )


def _ctate_features(X):
    """Low-dimensional second-stage design: raw columns + binned burden."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    itb = BASE_COLS.index("n_target_bl")
    ntb = np.clip(X[:, itb], 1, 8)
    bins = np.column_stack([
        (ntb == 3).astype(float), (ntb == 4).astype(float),
        (ntb == 5).astype(float), (ntb >= 6).astype(float),
    ])
    return np.column_stack([X, bins])


# ==========================================================================
# estimation
# ==========================================================================
def _ols_hc1(y, Z):
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    resid = y - Z @ beta
    bread = np.linalg.pinv(Z.T @ Z)
    meat = (Z * resid[:, None] ** 2).T @ Z
    V = bread @ meat @ bread
    return beta, np.sqrt(np.clip(np.diag(V), 0, None))


def _fit_design(draw, min_support=15):
    """Eligible-sample design with deterministic dropping of rare dummies.

    Returns ``(F, A, phi, keep_mask)`` for the eligible sample; engineered
    columns (beyond ``BASE_COLS``) supported by fewer than ``min_support``
    units are dropped (prevents empty-fold singularity at small n). The mask
    must be applied to any evaluation row fed to models trained on ``F``.
    """
    R = draw["R"]
    F = _ctate_features(draw["X"][BASE_COLS].to_numpy(float)[R])
    keep = np.ones(F.shape[1], dtype=bool)
    counts = F.sum(axis=0)
    for j in range(len(BASE_COLS), F.shape[1]):
        c = min(counts[j], len(F) - counts[j])
        if c < min_support:
            keep[j] = False
    return F[:, keep], draw["A"][R], draw["phi_obs"][R], keep


def _design_row(raw_row, keep):
    """Feature row for evaluation under the training column mask."""
    return _ctate_features(np.atleast_2d(np.asarray(raw_row, dtype=float)))[0][keep]


def estimate_rep(draw, tseq, truth, *, alpha=0.10, seed=0, n_fresh=48):
    """Full estimation stack on one replication; returns a record dict."""
    # augmented design (binned burden); rare dummies dropped deterministically
    Xr, Ar, phi_r, keep = _fit_design(draw)

    prop_feat = lambda z: z[:, :1]                          # noqa: E731
    prop = LogisticRegression(C=1.0, max_iter=2000)
    sample = (phi_r, Ar, Xr)
    cfr = cross_fit(sample, tseq, n_basis=5, propensity_estimator=prop,
                    propensity_feature_fn=prop_feat,
                    n_splits=2, stratify=True, random_state=seed)
    band = multiplier_bootstrap_band(cfr.influence()[0], tseq, cfr.aipw[0],
                                     alpha=alpha, n_boot=999, rng=seed + 1)
    # Corollary-5.4 sup test of H0: psi_d == 0 (correct power notion for a
    # bump-shaped curve; 'band excludes 0 everywhere' is far too strict)
    from tcda_uq.uq.asymptotic import topological_effect_test

    ttest = topological_effect_test(cfr.influence()[0], cfr.aipw[0],
                                    n_boot=999, rng=seed + 5)
    psi_hat = cfr.aipw[0]
    psi_R = truth["psi_R"]

    rec = {
        "tate_covered": bool(((psi_R >= band.lower)
                              & (psi_R <= band.upper)).all()),
        "tate_width": float(np.mean(band.upper - band.lower)),
        "tate_pval": float(ttest["pvalue"]),
        "tate_reject": bool(ttest["pvalue"] < alpha),
        "tate_bias_l2": float(np.sqrt(np.mean((psi_hat - psi_R) ** 2))),
        "n_elig": int(len(Ar)),
    }

    # ---- SLD comparator (observed absolute Delta-SLD, mm; OLS HC1) ---------
    dsld_obs = draw["dsld_obs"][draw["R"]]
    Zd = np.column_stack([np.ones(len(Ar)), Ar, Xr])
    beta, se = _ols_hc1(dsld_obs, Zd)
    zc = 1.645
    lo, hi = beta[1] - zc * se[1], beta[1] + zc * se[1]
    rec.update(
        sld_est=float(beta[1]), sld_lo=float(lo), sld_hi=float(hi),
        sld_covered=bool(lo <= truth["dsld_elig"] <= hi),
        sld_excl0=bool(lo > 0 or hi < 0),
    )

    # ---- CTATE at pre-specified profiles ----------------------------------
    learner = CTATEDRLearner(n_basis=4).fit(sample, tseq, cross_fit_result=cfr)
    for pname, xv in truth.get("ctate_profiles", {}).items():
        if xv.get("curve") is None:
            continue
        cur = np.asarray(xv["curve"])
        xrow = _design_row([PROFILES[pname][c] for c in BASE_COLS], keep)
        cb = ctate_confidence_band(learner, xrow, d=0, alpha=alpha,
                                   n_boot=399, rng=seed + 2)
        rec[f"ctate_cov_{pname}"] = bool(((cur >= cb.lower)
                                          & (cur <= cb.upper)).all())
        rec[f"ctate_w_{pname}"] = float(np.mean(cb.upper - cb.lower))
        rec[f"ctate_rmse_{pname}"] = float(
            np.sqrt(np.mean((cb.center - cur) ** 2)))

    # ---- ITTE prediction band vs fresh oracle deltas ----------------------
    model = ITTEConformal.fit(sample, tseq, calib_frac=0.5, n_basis=4,
                              propensity_estimator=prop,
                              propensity_feature_fn=prop_feat,
                              random_state=seed + 3)
    dgp = draw["_dgp"]
    fresh_sim = PacceSimulator(dgp["scenario"], n=n_fresh, seed=seed + 4,
                               std_consts=dgp["std_consts"],
                               resolution=dgp["resolution"])
    fd = fresh_sim.one_draw()
    lo, hi, _ = model.band_bounds(
        _ctate_features(fd["X"][BASE_COLS].to_numpy(float))[:, keep], alpha)
    inside = ((fd["delta"] >= lo) & (fd["delta"] <= hi)).all(axis=1)
    widths = hi - lo
    rec.update(
        itte_cov=float(inside.mean()), itte_width_med=float(np.median(widths)),
        itte_unbounded=bool(np.isinf(widths).any()),
    )
    return rec


def scenario_truths(scenario, n_truth=6000, n_mc_profile=800, resolution=60,
                    std_consts=None, hom=0):
    """Truths: eligible-population TATE, SLD contrast, CTATE profile curves."""
    sim = PacceSimulator(scenario, n=n_truth // 2, seed=99,
                         std_consts=std_consts, resolution=resolution, hom=hom)
    d = sim.one_draw()
    delta, R = d["delta"], d["R"]

    truth = {"scenario": scenario, "resolution": resolution, "hom": hom,
             "psi_R": delta[R].mean(axis=0).tolist(),
             "psi_full": delta.mean(axis=0).tolist(),
             "dsld_elig": float(d["dsld_units"][R].mean()),
             "ctate_profiles": {}}
    for pname, prow in PROFILES.items():
        sim_p = PacceSimulator(scenario, n=n_mc_profile, seed=777,
                               std_consts=std_consts, resolution=resolution,
                               hom=hom)
        dp = sim_p.one_draw(override_row=prow)
        tau = dp["delta"].mean(axis=0)
        truth["ctate_profiles"][pname] = {"x": [prow[c] for c in BASE_COLS],
                                          "curve": tau.tolist()}
    return truth


# ==========================================================================
# driver
# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--n", type=int, default=842)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--resolution", type=int, default=60)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    scen_list = [s for s in args.scenarios.split(",") if s]
    std_consts = frozen_standardization()
    tseq = np.linspace(SPEC_V1["silhouette"]["interval"][0],
                       SPEC_V1["silhouette"]["interval"][1], args.resolution)

    tag = f"{args.tag}_" if args.tag else ""
    out_path = OUT / f"{tag}shard{args.shard}_of_{args.num_shards}.parquet"

    reps = list(range(args.seed0, args.seed0 + args.reps))
    my_reps = [r for k, r in enumerate(reps)
               if k % args.num_shards == args.shard % args.num_shards]

    rows = []
    for scen in scen_list:
        tp = OUT / f"truth_{scen}_res{args.resolution}.json"
        if not tp.exists():
            tr = scenario_truths(scen, resolution=args.resolution,
                                 std_consts=std_consts)
            with open(tp, "w") as f:
                json.dump(tr, f)
        with open(tp) as f:
            truth = json.load(f)
        print(f"[shard {args.shard}] {scen}: reps={len(my_reps)} "
              f"|psi_R|inf={np.abs(truth['psi_R']).max():.4f}", flush=True)

        for seed in my_reps:
            t0 = time.monotonic()
            try:
                sim = PacceSimulator(scen, n=args.n, seed=seed,
                                     std_consts=std_consts,
                                     resolution=args.resolution)
                d = sim.one_draw()
                d["_dgp"] = dict(scenario=scen, std_consts=std_consts,
                                 resolution=args.resolution)
                rec = estimate_rep(d, tseq, truth, alpha=args.alpha, seed=seed)
                rec.update(scenario=scen, seed=seed, n=args.n,
                           elapsed=time.monotonic() - t0)
            except Exception as e:
                rec = dict(scenario=scen, seed=seed, n=args.n,
                           error=f"{type(e).__name__}: {e}",
                           elapsed=time.monotonic() - t0)
            rows.append(rec)
        ok = pd.DataFrame([r for r in rows if r.get("scenario") == scen
                           and "error" not in r])
        if len(ok):
            print(f"  running: tate_cov={ok.tate_covered.mean():.3f} "
                  f"sld_cov={ok.sld_covered.mean():.3f} "
                  f"itte_cov={ok.itte_cov.mean():.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
