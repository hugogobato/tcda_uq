"""PACCE Phase P3, WP-P3.5: negative controls and preprocessing sensitivity.

Controls (pre-specified in application_lock.yaml):
  * pre-treatment control -- baseline-only cloud geometry carries no treatment
    signal;
  * within-CHEMO-stratum label permutation -- placebo TATE rejects at ~ alpha.

Sensitivity forest: frozen-rule variants (size floor, visit rule, r exponent,
minimum points, H1 ablation) re-run through the identical TATE machinery.

Outputs: results/pacce/placebo_checks.csv, results/pacce/sensitivity_forest.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from experiments.pacce_p0_audit import DATA, RES, ensure_sas_dir
from experiments.pacce_p3_topology import ALPHA, SEED, ctate_features, fit_design
from tcda_uq.datasets.pacce import BASELINE_COLS, SPEC_V1, load_pacce_from_frames
from tcda_uq.estimators import cross_fit
from tcda_uq.uq.asymptotic import topological_effect_test


def _frames(sas):
    import pyreadstat

    from tcda_uq.datasets.pacce import DEMO_REQUIRED, LESION_REQUIRED

    demo, _ = pyreadstat.read_sas7bdat(str(Path(sas) / "demo.sas7bdat"),
                                       usecols=DEMO_REQUIRED)
    lesion, _ = pyreadstat.read_sas7bdat(str(Path(sas) / "lesion.sas7bdat"),
                                         usecols=LESION_REQUIRED)
    return demo, lesion


def tate_test(phi1, A, Xf, tseq, seed, n_boot=1999):
    prop_feat = lambda z: z[:, :1]                          # noqa: E731
    cfr = cross_fit((phi1, A, Xf), tseq, n_basis=5,
                    propensity_estimator=LogisticRegression(C=1.0,
                                                            max_iter=2000),
                    propensity_feature_fn=prop_feat,
                    n_splits=2, stratify=True, random_state=seed)
    t = topological_effect_test(cfr.influence()[0], cfr.aipw[0],
                                n_boot=n_boot, rng=seed + 5)
    return float(t["pvalue"])


def main():
    sas = ensure_sas_dir()
    demo, lesion = _frames(sas)
    base_spec = json.loads(json.dumps(SPEC_V1))
    tseq = np.linspace(0.0, 0.83, 100)

    # ---------------- pre-treatment control --------------------------------
    pre_spec = json.loads(json.dumps(base_spec))
    pre_spec["window"]["week12_visit"] = "__none__"        # baseline records only
    pre_spec["eligibility"]["min_distinct_visit_days"] = 1
    d_pre = load_pacce_from_frames(demo.copy(), lesion.copy(), pre_spec)
    Xpre, keep_pre = fit_design(d_pre.X[BASELINE_COLS].to_numpy(float))
    p_pre = tate_test(d_pre.phi[:, [0], :], d_pre.A, Xpre, tseq, SEED + 900)
    print(f"pre-treatment control p = {p_pre:.3f}")

    # ---------------- primary reference ------------------------------------
    d0 = load_pacce_from_frames(demo.copy(), lesion.copy(), base_spec)
    X0, _ = fit_design(d0.X[BASELINE_COLS].to_numpy(float))
    A0 = d0.A
    strata = d0.frame["CHEMO"].values

    # ---------------- placebo permutations ---------------------------------
    rng = np.random.default_rng(SEED + 901)
    n_perm = 200
    rows, t0 = [], time.monotonic()
    p_primary = None
    for k in range(n_perm + 1):
        if k == 0:
            Aperm = A0.copy()
            tag = "primary"
        else:
            Aperm = A0.copy()
            for s in np.unique(strata):
                m = strata == s
                Aperm[m] = rng.permutation(A0[m])
            tag = f"perm_{k:03d}"
        p = tate_test(d0.phi[:, [0], :], Aperm, X0, tseq, SEED + 902 + k,
                      n_boot=999)
        rows.append(dict(tag=tag, pvalue=p, reject=p < ALPHA))
        if k == 19 or (k and k % 40 == 0):
            print(f"  perm {k}/{n_perm} ({time.monotonic()-t0:.0f}s)", flush=True)
    pl = pd.DataFrame(rows)
    pl.to_csv(RES / "placebo_checks.csv", index=False)
    perm_rows = pl[pl.tag != "primary"]
    print(f"placebo rejection rate = {perm_rows['reject'].mean():.3f} "
          f"(alpha = {ALPHA}); primary p = "
          f"{pl.loc[pl.tag == 'primary', 'pvalue'].iloc[0]:.3f}")

    # ---------------- sensitivity forest ------------------------------------
    variants = []
    for floor in (3.0, 5.0, 10.0):
        variants.append(dict(tag=f"floor{floor:g}", floor=floor, rule="visit",
                             min_points=4, r=3.0, hom=0))
    variants.append(dict(tag="rule_day120", floor=5.0, rule="day",
                         min_points=4, r=3.0, hom=0))
    variants.append(dict(tag="minpts6", floor=5.0, rule="visit",
                         min_points=6, r=3.0, hom=0))
    for r in (2.0, 4.0):
        variants.append(dict(tag=f"r{r:g}", floor=5.0, rule="visit",
                             min_points=4, r=r, hom=0))
    variants.append(dict(tag="h1_ablation", floor=5.0, rule="visit",
                         min_points=4, r=3.0, hom=1))

    forest = []
    for v in variants:
        sp = json.loads(json.dumps(base_spec))
        sp["points"]["floor_mm"] = v["floor"]
        sp["window"]["rule"] = v["rule"]
        if v["rule"] == "day":
            sp["window"]["day_max"] = 120
        sp["eligibility"]["min_points"] = v["min_points"]
        sp["silhouette"]["r"] = v["r"]
        dv = load_pacce_from_frames(demo.copy(), lesion.copy(), sp)
        XV, keepv = fit_design(dv.X[BASELINE_COLS].to_numpy(float))
        p = tate_test(dv.phi[:, [v["hom"]], :], dv.A, XV, tseq, SEED + 700)
        forest.append(dict(variant=v["tag"], n_eligible=len(dv.A),
                           pvalue=p, reject=p < ALPHA))
        print(f"  {v['tag']}: n={len(dv.A)} p={p:.3f}", flush=True)
    fd = pd.DataFrame(forest)
    fd.to_csv(RES / "sensitivity_forest.csv", index=False)

    # append the two controls to the forest for one-stop reading
    pd.concat([fd, pd.DataFrame([
        dict(variant="pretreatment_control", n_eligible=len(d_pre.A),
             pvalue=p_pre, reject=p_pre < ALPHA),
        dict(variant="placebo_permutation_rate",
             n_eligible=len(A0),
             pvalue=float(perm_rows["reject"].mean()), reject=None),
    ])], ignore_index=True).to_csv(RES / "sensitivity_forest.csv", index=False)
    print("done.")


if __name__ == "__main__":
    main()
