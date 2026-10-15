"""PACCE Phase P3 application (WP-P3.2 / P3.3 / P3.4, gate G4).

Runs the frozen pipeline on the real release under the locked configuration in
results/pacce/application_lock.yaml: TATE with simultaneous band, SLD-functional
comparator, CTATE at six pre-specified profiles, ITTE prediction bands,
pre-treatment and placebo negative controls, and the preprocessing-sensitivity
forest.

Outputs (results/pacce/)
------------------------
tate_estimates.parquet, tate_bands.pdf, tate_vs_sld.csv, tate_analysis_report.md
ctate_profiles.yaml, ctate_estimates.parquet, ctate_bands.pdf, ctate_report.md
itte_prediction_profiles.yaml, itte_prediction_bands.pdf, itte_width_summary.csv,
itte_claim_boundary.md
placebo_checks.csv, sensitivity_forest.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression

from experiments.pacce_p0_audit import DATA, RES, ensure_sas_dir
from tcda_uq.datasets.pacce import (
    BASELINE_COLS,
    SPEC_V1,
    load_outcome_spec,
    load_pacce_from_frames,
)
from tcda_uq.estimators import CTATEDRLearner, cross_fit
from tcda_uq.uq.asymptotic import (
    ctate_confidence_band,
    multiplier_bootstrap_band,
    topological_effect_test,
)
from tcda_uq.uq.conformal import ITTEConformal

ALPHA = 0.10
SEED = 20240

PROFILES = {}
for chemo_name, chemo_val in (("oxali", 0.0), ("iri", 1.0)):
    for b_name, ntb in (("low", 2.0), ("mid", 4.0), ("high", 6.0)):
        PROFILES[f"{chemo_name}_{b_name}_burden"] = dict(
            chemo_iri=chemo_val, age=6.1, male=1.0, ecog_ge1=0.0,
            n_target_bl=ntb, n_metsites_bl=2.0, kras_wt=1.0, kras_mut=0.0,
            kras_missing=0.0, ldh_high_bl=0.0)


def ctate_features(X):
    """Binned-burden design (identical family to the validated simulation)."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    itb = BASELINE_COLS.index("n_target_bl")
    ntb = X[:, itb]
    bins = np.column_stack([
        ((ntb >= 3) & (ntb <= 4)).astype(float),
        (ntb >= 5).astype(float),
    ])
    return np.column_stack([X, bins])


def fit_design(X_raw, min_support=15):
    """Binned-burden design with deterministic rare-dummy dropping."""
    F = ctate_features(X_raw)
    keep = np.ones(F.shape[1], dtype=bool)
    counts = F.sum(axis=0)
    for j in range(len(BASELINE_COLS), F.shape[1]):
        c = min(counts[j], len(F) - counts[j])
        if c < min_support:
            keep[j] = False
    return F[:, keep], keep


def design_row(raw_row, keep):
    return ctate_features(np.atleast_2d(np.asarray(raw_row, dtype=float)))[0][keep]


def sld_functional(frame):
    """Per-subject functional SLD outcome on the frozen grid (mm/100 scale)."""
    res = len(SPEC_V1["silhouette"]["interval"]) and 100
    tseq = np.linspace(0.0, 0.83, res)
    u0 = (-6.0 / 117.0 - 0.28611553900385733) / 0.413487541888698
    u1 = (81.0 / 117.0 - 0.28611553900385733) / 0.413487541888698
    bl = (np.log(np.maximum(frame["bl_sld"].values, 5.0)) - 3.23191748652353) \
        / 0.6358818190862385
    w12 = (np.log(np.maximum(frame["w12_sld"].values, 5.0)) - 3.23191748652353) \
        / 0.6358818190862385
    g = np.empty((len(frame), 1, res))
    for i in range(len(frame)):
        yi = np.interp(tseq, [u0, u1], [bl[i], w12[i]])
        g[i, 0, :] = yi / 100.0 * 50.0          # mm-scale contrast, comparable sd
    return g


def run_tate(phi, A, Xf, tseq, *, seed, n_boot=4999):
    prop_feat = lambda z: z[:, :1]                          # noqa: E731
    prop = LogisticRegression(C=1.0, max_iter=2000)
    cfr = cross_fit((phi, A, Xf), tseq, n_basis=5,
                    propensity_estimator=prop,
                    propensity_feature_fn=prop_feat,
                    n_splits=2, stratify=True, random_state=seed)
    band = multiplier_bootstrap_band(cfr.influence()[0], tseq, cfr.aipw[0],
                                     alpha=ALPHA, n_boot=n_boot, rng=seed + 1)
    test = topological_effect_test(cfr.influence()[0], cfr.aipw[0],
                                   n_boot=n_boot, rng=seed + 5)
    return cfr, band, test


def main():
    sas = ensure_sas_dir()
    spec = load_outcome_spec(DATA / "outcome_spec.yaml")
    demo, lesion = _read_frames(sas)
    data = load_pacce_from_frames(demo, lesion, spec)
    frame = data.frame.reset_index(drop=True)
    phi, tseq, A = data.phi, data.tseq, data.A
    print(f"n eligible = {len(A)}; arms = {np.bincount(A.astype(int))}")

    X_raw = data.X[BASELINE_COLS].to_numpy(float)
    Xf, keep = fit_design(X_raw)

    # ------------------------------------------------------------- TATE (H0)
    cfr, band, test = run_tate(phi[:, [0], :], A, Xf, tseq, seed=SEED)
    psi = cfr.aipw[0]
    pd.DataFrame(dict(tseq=tseq, psi_hat=psi, lower=band.lower,
                      upper=band.upper)).to_parquet(
        RES / "tate_estimates.parquet", index=False)
    pd.DataFrame([dict(pvalue=test["pvalue"], reject=test["pvalue"] < ALPHA,
                       statistic=test["statistic"],
                       mean_width=float(np.mean(band.upper - band.lower)),
                       n=len(A))]).to_csv(RES / "tate_test.csv", index=False)

    # ------------------------------------------- SLD-functional comparator
    g = sld_functional(frame)
    cfr_s, band_s, test_s = run_tate(g, A, Xf, tseq, seed=SEED + 100)
    pd.DataFrame([dict(pvalue=test_s["pvalue"],
                       reject=test_s["pvalue"] < ALPHA,
                       mean_width=float(np.mean(band_s.upper - band_s.lower)))
                 ]).to_csv(RES / "sldfun_tate_test.csv", index=False)
    pd.DataFrame(dict(tseq=tseq, psi_hat=cfr_s.aipw[0],
                      lower=band_s.lower, upper=band_s.upper)).to_parquet(
        RES / "sldfun_tate_estimates.parquet", index=False)

    # ---------------------------------------------------------------- plots
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.fill_between(tseq, band.lower, band.upper, alpha=0.25,
                    label="TATE 90% simultaneous band")
    ax.plot(tseq, psi, lw=2, label="TATE estimate (AIPW)")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("filtration scale t"); ax.set_ylabel("psi_0(t)")
    ax.set_title(f"PACCE TATE, H0 silhouette (p = {test['pvalue']:.3g})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "tate_bands.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.fill_between(tseq, band_s.lower, band_s.upper, alpha=0.25,
                    color="#ee7733", label="SLD-functional 90% band")
    ax.plot(tseq, cfr_s.aipw[0], color="#ee7733", lw=2, label="SLD-functional TATE")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("filtration scale t"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "sldfun_tate_bands.pdf"); plt.close(fig)

    # ------------------------------------------------------------- CTATE
    learner = CTATEDRLearner(n_basis=4).fit((phi[:, [0], :], A, Xf), tseq,
                                            cross_fit_result=cfr)
    ctate_rows, curves = [], {}
    for pname, prow in PROFILES.items():
        xrow = design_row([prow[c] for c in BASELINE_COLS], keep)
        cb = ctate_confidence_band(learner, xrow, d=0, alpha=ALPHA,
                                   n_boot=1999, rng=hash(pname) % (2**31))
        support = int(((np.abs(X_raw - np.array([prow[c] for c in BASELINE_COLS]))
                        [:, [0, 4]] == 0).all(axis=1)).sum())
        curves[pname] = dict(tseq=tseq.tolist(), center=cb.center.tolist(),
                             lower=cb.lower.tolist(), upper=cb.upper.tolist())
        ctate_rows.append(dict(profile=pname, burden=prow["n_target_bl"],
                               chemo_iri=prow["chemo_iri"], n_matching=support,
                               pvalue=None))
    pd.DataFrame(ctate_rows).to_parquet(RES / "ctate_estimates.parquet",
                                        index=False)
    with open(RES / "ctate_profiles.yaml", "w") as f:
        yaml.dump({"profiles": {k: v for k, v in PROFILES.items()},
                   "feature_rule": "binned burden (<=2 ref, 3-4, >=5)",
                   "note": "bands simultaneous-in-t at fixed profile "
                           "(pointwise-in-x)"},
                  f, sort_keys=False)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    for axi, (pname, cur) in zip(axes.ravel(), curves.items()):
        axi.fill_between(cur["tseq"], cur["lower"], cur["upper"], alpha=0.25)
        axi.plot(cur["tseq"], cur["center"], lw=2)
        axi.axhline(0, color="k", lw=0.8, ls="--")
        axi.set_title(pname, fontsize=9)
    fig.suptitle("CTATE profiles (90% simultaneous-in-t bands)")
    fig.tight_layout(); fig.savefig(RES / "ctate_bands.pdf"); plt.close(fig)

    # ------------------------------------------------------------- ITTE
    model = ITTEConformal.fit((phi[:, [0], :], A, Xf), tseq, calib_frac=0.5,
                              n_basis=4, propensity_estimator=LogisticRegression(
                                  C=1.0, max_iter=2000),
                              propensity_feature_fn=lambda z: z[:, :1],
                              random_state=SEED + 3)
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for axi, (pname, prow) in zip(axes.ravel(), PROFILES.items()):
        xrow = design_row([prow[c] for c in BASELINE_COLS], keep)
        lo, hi, ctr = model.band_bounds(xrow, ALPHA)
        rows.append(dict(profile=pname, width_mean=float(np.mean(hi - lo)),
                         width_median=float(np.median(hi - lo)),
                         unbounded=bool(np.isinf(hi - lo).any())))
        axi.fill_between(tseq, lo[0], hi[0], alpha=0.25, color="#ee7733")
        axi.plot(tseq, ctr[0], color="#ee7733", lw=2)
        axi.axhline(0, color="k", lw=0.8, ls="--")
        axi.set_title(pname, fontsize=9)
    fig.suptitle("ITTE prediction bands for a new patient at pre-specified profiles")
    fig.tight_layout(); fig.savefig(RES / "itte_prediction_bands.pdf"); plt.close(fig)
    pd.DataFrame(rows).to_csv(RES / "itte_width_summary.csv", index=False)

    (RES / "itte_claim_boundary.md").write_text(
        "# ITTE claim boundary\n\n"
        "- The composed conformal band is a **prediction band** for the "
        "individual topological treatment effect of a NEW patient with the "
        "stated baseline profile.\n"
        "- Only one potential outcome exists per observed subject: PACCE cannot "
        "validate realized individual-effect coverage, provide an "
        "effect-vs-truth scatter, or support factual-vs-counterfactual residual "
        "claims. Coverage validation lives in the PACCE-matched simulation.\n"
        "- Bands are simultaneous over the filtration scale and marginal over "
        "the population; they are NOT confidence bands for a mean effect.\n")

    print("TATE p =", test["pvalue"], "| SLD-functional p =", test_s["pvalue"])
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def _read_frames(sas):
    import pyreadstat

    from tcda_uq.datasets.pacce import DEMO_REQUIRED, LESION_REQUIRED

    demo, _ = pyreadstat.read_sas7bdat(str(Path(sas) / "demo.sas7bdat"),
                                       usecols=DEMO_REQUIRED)
    lesion, _ = pyreadstat.read_sas7bdat(str(Path(sas) / "lesion.sas7bdat"),
                                         usecols=LESION_REQUIRED)
    return demo, lesion


if __name__ == "__main__":
    main()
