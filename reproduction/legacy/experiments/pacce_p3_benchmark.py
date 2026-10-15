"""PACCE Phase P3, WP-P3.1: blinded application lock + conventional benchmark.

Runs BEFORE any treatment-group topology is computed. Freezes the analysis
configuration (application_lock.yaml), then reproduces the established PACCE
clinical direction from the release: modified-RECIST response summaries,
central-review PFS (Kaplan-Meier), and severe-AE incidence by assigned arm.

Outputs
-------
results/pacce/application_lock.yaml
results/pacce/conventional_benchmark.csv
results/pacce/benchmark_reconciliation.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.pacce_p0_audit import DATA, RES, ensure_sas_dir, read_domain

ALPHA = 0.10


def km_median(time, event):
    """Kaplan-Meier median (np.inf if not reached) with a tiny KM implementation."""
    t = np.asarray(time, float)
    e = np.asarray(event, int)
    ts = np.unique(t[e == 1])
    s = 1.0
    curve = [(0.0, 1.0)]
    for u in ts:
        at_risk = (t >= u).sum()
        deaths = ((t == u) & (e == 1)).sum()
        if at_risk > 0:
            s *= 1 - deaths / at_risk
        curve.append((u, s))
    for u, sv in curve:
        if sv <= 0.5:
            return u
    return np.inf


def logrank(t1, e1, t0, e0):
    """Two-sample log-rank statistic and chi-square(1) p-value."""
    t = np.concatenate([t1, t0])
    e = np.concatenate([e1, e0]).astype(int)
    g = np.concatenate([np.ones(len(t1)), np.zeros(len(t0))])
    O_E, V = 0.0, 0.0
    for u in np.unique(t[e == 1]):
        n = (t >= u).sum()
        n1 = ((t >= u) & (g == 1)).sum()
        d = ((t == u) & (e == 1)).sum()
        d1 = ((t == u) & (e == 1) & (g == 1)).sum()
        if n <= 1:
            continue
        O_E += d1 - d * n1 / n
        V += d * (n1 / n) * (1 - n1 / n) * (n - d) / (n - 1)
    chi2 = O_E**2 / V if V > 0 else 0.0
    from scipy.stats import chi2 as _chi2

    return float(O_E), float(chi2), float(_chi2.sf(chi2, 1))


def main():
    sas = ensure_sas_dir()

    # ------------------------------------------------------------------ lock
    lock = {
        "lock_version": "pacce-application-v1",
        "frozen_before": "any treatment-group topology computation",
        "population": "released 842-subject PACCE subset; eligible n=566 "
                      "(observed-trajectory target; see docs/pacce_estimand_lock.md)",
        "treatment": "assigned TRT; A=1 panitumumab+bev+chemo",
        "stratum": "CHEMO",
        "outcome": "data/pacce/outcome_spec.yaml v1.0 (H0 primary)",
        "estimators": {
            "TATE": "cross-fitted AIPW + multiplier-bootstrap simultaneous band",
            "CTATE": "functional DR-learner, binned-burden features; profiles "
                     "= CHEMO x burden terciles (6 pre-specified); bands "
                     "simultaneous-in-t, pointwise-in-profile",
            "ITTE": "composed weighted split-conformal prediction band "
                    "(naive randomized propensity weights)",
        },
        "alpha": ALPHA,
        "n_basis_outcome": 5,
        "n_basis_ctate": 4,
        "n_boot": 999,
        "seeds": {"pipeline": 20240, "placebo_permutations": 200},
        "negative_controls": [
            "baseline-only clouds (pre-treatment geometry): TATE must be null",
            "within-CHEMO-stratum label permutation: null rate ~ alpha",
        ],
    }
    import yaml

    with open(RES / "application_lock.yaml", "w") as f:
        yaml.dump(lock, f, sort_keys=False)

    # ------------------------------------------------------ conventional bench
    demo_cols = ["SUBJID", "TRT", "CHEMO"]
    demo = read_domain(sas, "demo", usecols=demo_cols + ["AGE", "SEX", "B_ECOG",
                                                         "B_LESNMT", "B_METANM",
                                                         "KRAS", "B_LDHYN"])
    ae_end = read_domain(sas, "a_eendpt", usecols=["SUBJID", "R12CRCA", "ROSCR",
                                                   "RCR", "PFSDYCR", "PFSCR"])
    ae = read_domain(sas, "ae", usecols=["SUBJID", "AESEV", "AESER"])

    demo["SUBJID"] = demo["SUBJID"].astype(int)
    arm = np.where(demo["TRT"].str.startswith("panit"), "P+B+C", "B+C")

    rows = []
    # week-12 central modified RECIST category distribution
    r12 = ae_end.set_index(ae_end["SUBJID"].astype(int))
    for a in ("B+C", "P+B+C"):
        ids = set(demo.loc[arm == a, "SUBJID"])
        sub = r12[r12.index.isin(ids)]
        cat = sub["R12CRCA"].replace("", "NE")
        n_eval = (~cat.isin(["NE", "UE"])).sum()
        resp = cat.isin(["CR", "PR"]).sum()
        dcr = cat.isin(["CR", "PR", "SD"]).sum()
        rows.append(dict(benchmark="week12_modified_RECIST", arm=a, n=len(ids),
                         value="response_rate", estimate=resp / max(n_eval, 1),
                         denominator=f"evaluable={n_eval}"))
        rows.append(dict(benchmark="week12_modified_RECIST", arm=a, n=len(ids),
                         value="disease_control_rate",
                         estimate=dcr / max(n_eval, 1),
                         denominator=f"evaluable={n_eval}"))
    # on-study response (central RECIST)
    for a in ("B+C", "P+B+C"):
        ids = set(demo.loc[arm == a, "SUBJID"])
        sub = r12[r12.index.isin(ids)]
        rows.append(dict(benchmark="on_study_response_central", arm=a, n=len(ids),
                         value="responder_rate",
                         estimate=float(sub["ROSCR"].mean()), denominator="all"))
    # PFS (central RECIST)
    e2 = ae_end.copy()
    e2["SUBJID"] = e2["SUBJID"].astype(int)
    e2 = e2.merge(demo[["SUBJID", "TRT"]], on="SUBJID", how="left")
    e2["arm"] = np.where(e2["TRT"].str.startswith("panit"), "P+B+C", "B+C")
    med = {}
    for a in ("B+C", "P+B+C"):
        s = e2[e2.arm == a]
        m = km_median(s["PFSDYCR"], s["PFSCR"])
        med[a] = m
        rows.append(dict(benchmark="PFS_central_KM_median_days", arm=a, n=len(s),
                         value="median_pfs_days", estimate=float(m),
                         denominator=""))
    oe, chi2, p = logrank(
        e2.loc[e2.arm == "P+B+C", "PFSDYCR"].values,
        e2.loc[e2.arm == "P+B+C", "PFSCR"].values,
        e2.loc[e2.arm == "B+C", "PFSDYCR"].values,
        e2.loc[e2.arm == "B+C", "PFSCR"].values)
    rows.append(dict(benchmark="PFS_logrank", arm="P+B+C vs B+C", n=int(len(e2)),
                     value="O-E (panitu minus bev)", estimate=oe,
                     denominator=f"chi2={chi2:.2f}, p={p:.4g}"))
    # severe AEs
    ae = ae.copy()
    ae["SUBJID"] = ae["SUBJID"].astype(int)
    ae_map = demo[["SUBJID"]].copy()
    ae_map["arm"] = arm
    sev_subj = set(ae.loc[ae["AESEV"].isin(["Severe", "Life threatening", "Fatal"]),
                          "SUBJID"])
    ser_subj = set(ae.loc[ae["AESER"] == "Y", "SUBJID"]) if "AESER" in ae else set()
    for a in ("B+C", "P+B+C"):
        ids = set(demo.loc[arm == a, "SUBJID"])
        rows.append(dict(benchmark="AE_grade_severe_plus", arm=a, n=len(ids),
                         value="subject_incidence",
                         estimate=len(ids & sev_subj) / len(ids),
                         denominator=f"n_severe={len(ids & sev_subj)}"))
        if ser_subj:
            rows.append(dict(benchmark="AE_serious", arm=a, n=len(ids),
                             value="subject_incidence",
                             estimate=len(ids & ser_subj) / len(ids),
                             denominator=f"n_serious={len(ids & ser_subj)}"))

    bench = pd.DataFrame(rows)
    bench.to_csv(RES / "conventional_benchmark.csv", index=False)

    # ------------------------------------------------------- reconciliation
    def get(bm, val, arm_):
        q = bench[(bench.benchmark == bm) & (bench.value == val) & (bench.arm == arm_)]
        return float(q["estimate"].iloc[0]) if len(q) else np.nan

    rr0, rr1 = get("week12_modified_RECIST", "response_rate", "B+C"), \
        get("week12_modified_RECIST", "response_rate", "P+B+C")
    pfs0, pfs1 = get("PFS_central_KM_median_days", "median_pfs_days", "B+C"), \
        get("PFS_central_KM_median_days", "median_pfs_days", "P+B+C")
    tox0, tox1 = get("AE_grade_severe_plus", "subject_incidence", "B+C"), \
        get("AE_grade_severe_plus", "subject_incidence", "P+B+C")

    checks = {
        "response_not_improved_by_panitumumab": bool(rr1 <= rr0 + 0.05),
        "pfs_not_improved_by_panitumumab": bool(pfs1 <= pfs0 * 1.05),
        "toxicity_higher_with_panitumumab": bool(tox1 >= tox0),
    }
    md = f"""# PACCE conventional benchmark reconciliation (WP-P3.1)

Population: released 842-subject subset (421 vs 421 by assigned TRT).
Published reference: Hecht et al., JCO 2009 -- adding panitumumab to
bevacizumab + chemotherapy produced **no PFS improvement** (numerically shorter
PFS) and **increased toxicity**, with similar response rates.

| Quantity | B+C | P+B+C | Published direction | Reproduced |
|---|---:|---:|---|---|
| Week-12 central modified-RECIST response rate | {rr0:.3f} | {rr1:.3f} | similar/lower with panitu | {checks['response_not_improved_by_panitumumab']} |
| Central KM median PFS (days) | {pfs0:.0f} | {pfs1:.0f} | no improvement (shorter) with panitu | {checks['pfs_not_improved_by_panitumumab']} |
| Severe+ AE incidence | {tox0:.3f} | {tox1:.3f} | higher with panitu | {checks['toxicity_higher_with_panitumumab']} |

Log-rank (panitu − bev): O−E = {oe:.1f}, chi2 = {chi2:.2f}, p = {p:.3g}.

Caveats: this is a **released subset** of PACCE (842 of 1053 randomized), so
numbers need not equal the publication exactly; only the direction is checked.
All counts are computed from the supplied SAS domains under the frozen
baseline dictionary; no topology enters this file.
"""
    (RES / "benchmark_reconciliation.md").write_text(md)
    print(md)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
