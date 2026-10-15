"""PACCE Phase P1, WP-P1.1: freeze the outcome/observation rule and document the
observation process.

Outputs
-------
data/pacce/outcome_spec.yaml                    final frozen construction
results/pacce/observation_process_report.pdf    flow + eligibility model (no outcome values)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.pacce_p0_audit import (
    DATA,
    RES,
    build_baseline_X,
    ensure_sas_dir,
    read_domain,
)

SPEC = {
    "spec_version": "pacce-outcome-v1.0-locked",
    "status": "LOCKED before treatment-group topology inspection",
    "estimand": "observed-trajectory TATE on the eligible set (R_i = 1); not ITT",
    "domain": "lesion.sas7bdat",
    "records": "LSTYPE == 'Central' and LSCAT == 'Target lesion'",
    "window": {"baseline": "DOSREFDY <= 0",
               "week12": "VISIT == 'Week 12' and DOSREFDY <= 117"},
    "points": {"u": "DOSREFDY / 117",
               "z": "log(max(LSLD, 5))",
               "floor_mm": 5.0,
               "standardization": "pooled z-score of (u, z) over the eligible pilot sample",
               "dedup": "deterministic 1e-6 jitter on exact duplicate coordinates"},
    "eligibility": {"min_distinct_visit_days": 2, "min_points": 4},
    "filtration": "alpha complex; radii = sqrt(alpha filtration); essential H0 dropped",
    "silhouette": {"r": 3.0, "resolution": 100, "interval": [0.0, 0.83],
                   "t_max_rule": "ceil(99th percentile of pooled finite deaths)"},
    "primary_summary": "H0 silhouette curve phi_{i,0}(t)",
    "ablation_summary": "H1 silhouette curve phi_{i,1}(t)",
    "n_eligible": 566,
    "treatment_blind_at_freeze": True,
}


def main():
    sas = ensure_sas_dir()
    les = read_domain(sas, "lesion")
    demo = read_domain(
        sas, "demo",
        usecols=["SUBJID", "TRT", "ATRT", "CHEMO", "AGE", "SEX", "B_ECOG",
                 "B_LESNMT", "B_METANM", "KRAS", "B_LDHYN"])
    ae = read_domain(sas, "a_eendpt", usecols=["SUBJID", "PFSDYCR", "PFSCR"])

    with open(DATA / "outcome_spec.yaml", "w") as f:
        yaml_dump(SPEC, f)

    # ---------------- observation process ---------------------------------
    d = demo.copy()
    d["SUBJID"] = d["SUBJID"].astype(int)
    d["arm"] = np.where(d["TRT"].str.startswith("panit"), "P+B+C", "B+C")

    ct = les[(les["LSTYPE"] == "Central") & (les["LSCAT"] == "Target lesion")
             & les["LSLD"].notna() & les["DOSREFDY"].notna()]
    bl_ids = set(ct.loc[ct["DOSREFDY"] <= 0, "SUBJID"].astype(int))
    w12_ids = set(ct.loc[(ct["VISIT"] == "Week 12")
                         & (ct["DOSREFDY"] <= 117), "SUBJID"].astype(int))
    both_ids = bl_ids & w12_ids

    dfp = pd.read_parquet(RES / "topology_pilot_subject_level.parquet")
    elig_ids = set(dfp["SUBJID"].astype(int))

    stages = [("Released", set(d["SUBJID"])), ("Baseline central target", bl_ids),
              ("+ Week-12 central target", both_ids), ("Eligible (frozen rule)", elig_ids)]
    rows = []
    for name, ids in stages:
        sub = d[d["SUBJID"].isin(ids)]
        rows.append(dict(stage=name, n=len(ids), n_bev=int((sub.arm == "B+C").sum()),
                         n_panitu=int((sub.arm == "P+B+C").sum()),
                         pct_of_842=round(100 * len(ids) / len(d), 1)))
    flow = pd.DataFrame(rows)
    print(flow.to_string(index=False))

    # eligibility logistic: R ~ X + A (coefficients only; no outcome values)
    Xb = build_baseline_X(d.set_index("SUBJID").loc[d["SUBJID"]]).reset_index(drop=True)
    Z = pd.DataFrame({"const": 1.0, **{c: Xb[c] for c in Xb.columns},
                      "arm_panitu": (d["arm"] == "P+B+C").astype(float).values})
    R = d["SUBJID"].isin(elig_ids).astype(float).values
    Zm = Z.values.astype(float)
    beta, *_ = np.linalg.lstsq(Zm, R - R.mean(), rcond=None) if False else (None,)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=4000, C=1e6).fit(Zm, R)
    coefs = pd.DataFrame({"term": Z.columns, "OR": np.exp(lr.coef_[0])})
    print(coefs.round(3).to_string(index=False))
    pi = lr.predict_proba(Zm)[:, 1]

    # week-12 timing (observation-process quantity, pooled and by-arm medians only)
    w12days = ct[(ct["VISIT"] == "Week 12") & (ct["DOSREFDY"] <= 117)][
        ["SUBJID", "DOSREFDY"]].drop_duplicates("SUBJID")
    w12days["SUBJID"] = w12days["SUBJID"].astype(int)
    w12days = w12days.merge(d[["SUBJID", "arm"]], on="SUBJID")
    med_by_arm = w12days.groupby("arm")["DOSREFDY"].median().to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    y = np.arange(len(flow))[::-1]
    axes[0].barh(y - 0.18, flow.n_bev, height=0.36, label="B+C", color="#4477aa")
    axes[0].barh(y + 0.18, flow.n_panitu, height=0.36, label="P+B+C", color="#ee7733")
    for yi, (nb, np_, ntot) in zip(y, flow[["n_bev", "n_panitu", "n"]].values):
        axes[0].text(max(nb, np_) + 6, yi, f"{ntot}", va="center", fontsize=9)
    axes[0].set_yticks(y); axes[0].set_yticklabels(flow.stage, fontsize=8)
    axes[0].set_xlabel("subjects"); axes[0].legend(fontsize=8)
    axes[0].set_title("Observation flow by assigned arm (counts only)")
    axes[1].hist(pi[R == 1], bins=25, alpha=0.6, density=True, label="eligible")
    axes[1].hist(pi[R == 0], bins=25, alpha=0.6, density=True, label="not eligible")
    axes[1].set_xlabel("P(eligible | X, A)")
    axes[1].set_title("Eligibility model (logistic)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "observation_process_report.pdf")

    with open(RES / "observation_process_summary.txt", "w") as f:
        f.write(flow.to_string(index=False) + "\n\n"
                + coefs.round(3).to_string(index=False)
                + f"\n\nweek-12 day median by arm: {med_by_arm}\n"
                f"eligibility-rate by arm: "
                f"{(R * (d.arm == 'P+B+C').values).sum() / (d.arm == 'P+B+C').sum():.3f} (P+B+C), "
                f"{(R * (d.arm == 'B+C').values).sum() / (d.arm == 'B+C').sum():.3f} (B+C)\n")


def yaml_dump(spec, f):
    import yaml
    yaml.dump(spec, f, sort_keys=False)


if __name__ == "__main__":
    main()
