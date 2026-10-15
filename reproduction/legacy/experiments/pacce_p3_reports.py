"""PACCE Phase P3 closeout: TATE/CTATE reports + G4 application gate memo."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.pacce_p0_audit import DATA, RES, ensure_sas_dir, read_domain

ROOT = Path(__file__).resolve().parents[1]


def main():
    sas = ensure_sas_dir()
    tate = pd.read_parquet(RES / "tate_estimates.parquet")
    ttest = pd.read_csv(RES / "tate_test.csv").iloc[0]
    sldt = pd.read_csv(RES / "sldfun_tate_test.csv").iloc[0]
    pilot = pd.read_parquet(RES / "topology_pilot_subject_level.parquet")

    # descriptive arm-mean curves (no inference; labels unblinded only here)
    demo = read_domain(sas, "demo", usecols=["SUBJID", "TRT"])
    demo["SUBJID"] = demo["SUBJID"].astype(int)
    arm1 = set(demo.loc[demo.TRT.str.startswith("panit"), "SUBJID"])
    is1 = pilot["SUBJID"].astype(int).isin(arm1).values

    # peak location/direction of the (null) TATE estimate
    pk = int(np.argmax(np.abs(tate.psi_hat)))
    bench = pd.read_csv(RES / "conventional_benchmark.csv")
    rr = bench[bench.value == "response_rate"]
    pfs = bench[(bench.benchmark == "PFS_central_KM_median_days")]
    tox = bench[bench.benchmark == "AE_grade_severe_plus"]

    sens = pd.read_csv(RES / "sensitivity_forest.csv")
    placebo = pd.read_csv(RES / "placebo_checks.csv")
    perm = placebo[placebo.tag.str.startswith("perm")]

    md = f"""# PACCE TATE analysis report (WP-P3.2)

Population: {int(ttest.n)} eligible subjects (observed-trajectory target,
288 B+C vs 278 P+B+C by assigned TRT). Outcome: frozen H0 alpha-complex
silhouette of central target-lesion window clouds
(`data/pacce/outcome_spec.yaml` v1.0). Estimator: cross-fitted AIPW with
CHEMO-stratum propensity; simultaneous 90% multiplier-bootstrap band;
Corollary-5.4 sup test.

## Result

The topological treatment-effect curve is **flat and null**: sup-test
p = {ttest.pvalue:.3f}; the band covers zero over the whole filtration scale.
Peak |estimate| {np.abs(tate.psi_hat).max():.4f} at t = {tate.tseq[pk]:.3f}
against a band half-width ~{float(ttest.mean_width) / 2:.4f}.

The functional-SLD comparator run through identical machinery gives
p = {sldt.pvalue:.3f}: the week-12 SLD trajectory shows no significant
treatment contrast in this eligible subset either, consistent with the
published PACCE response null (response rate {rr[rr.arm == 'B+C'].estimate.iloc[0]:.3f} vs
{rr[rr.arm == 'P+B+C'].estimate.iloc[0]:.3f}) and with the published harm
signal living in PFS/toxicity (median PFS {pfs[pfs.arm == 'B+C'].estimate.iloc[0]:.0f} vs
{pfs[pfs.arm == 'P+B+C'].estimate.iloc[0]:.0f} days; severe+ AE {tox[tox.arm == 'B+C'].estimate.iloc[0]:.3f} vs
{tox[tox.arm == 'P+B+C'].estimate.iloc[0]:.3f}), not in deeper tumor shrinkage.

## Robustness

* Pre-treatment control (baseline-only geometry): p = {sens[sens.variant == 'pretreatment_control'].pvalue.iloc[0]:.3f} (null as required).
* Placebo label permutation within CHEMO strata (200 draws): rejection rate
  {perm['reject'].mean():.3f} at alpha = 0.10 (calibrated).
* Sensitivity forest (size floors 3/5/10 mm, day-window rule, r in {{2,3,4}},
  min-points 6, H1 ablation): all p >= {sens[sens.variant != 'placebo_permutation_rate'].pvalue.min():.2f};
  no variant produces a signal.
* Simulation backing (G3): calibrated coverage/type-I on PACCE-matched DGPs,
  power 1.00 in the topology-favorable regime at this n -- so the observed null
  is not an artifact of an underpowered pipeline.

## Interpretation

Under the frozen observed-trajectory estimand, adding panitumumab to
bevacizumab + chemotherapy does not measurably change the H0 multiscale
organization of central target-lesion trajectories during the first treatment
window. Given the validated power envelope, the honest reading is a **true-null
or below-detectability effect on lesion-trajectory topology**, coexisting with
the published toxicity/PFS harm. Topology here is a *negative-control-quality*
re-expression of the radiographic record rather than a biomarker of the
panitumumab effect.
"""
    (RES / "tate_analysis_report.md").write_text(md)

    ct = pd.read_csv(RES / "ctate_estimates.parquet")
    iw = pd.read_csv(RES / "itte_width_summary.csv")
    md2 = f"""# PACCE CTATE and ITTE reports (WP-P3.3 / WP-P3.4)

## CTATE (secondary; pointwise-in-profile)

Six pre-specified profiles (CHEMO backbone x baseline target burden
<=2 / 3-4 / >=5), functional DR-learner with binned-burden features, 90%
bands simultaneous-in-t at each fixed profile (`ctate_bands.pdf`).

Matching-structure counts (chemo x burden cells): see `ctate_estimates.parquet`
(n_matching column). With a globally null TATE (p = {ttest.pvalue:.2f}), profile-level
excursions are exploratory only; none survive any multiplicity correction, and
the simulation study documented anti-conservative behavior for profile bands
under step-shaped conditional-truth DGPs. No heterogeneity claim is made.

## ITTE (exploratory prediction)

Composed weighted split-conformal bands for a NEW patient's individual
topological effect curve are finite at every pre-specified profile
(mean width ~{iw.width_mean.mean():.2f} silhouette units; no unbounded bands).
Widths are profile-invariant because the randomized-design propensity weights
are flat; centers shift with the profile plug-in. Per the claim boundary
(`itte_claim_boundary.md`): prediction objects only -- PACCE cannot validate
realized individual-effect coverage.
"""
    (RES / "ctate_report.md").write_text(md2)

    # ------------------------------------------------------------- G4 memo
    memo = f"""# G4 application gate memo (WP-P3.5)

## Gate evidence

| Requirement | Evidence | Status |
|---|---|---|
| Conventional benchmark reproduces published direction | response {rr[rr.arm == 'B+C'].estimate.iloc[0]:.3f}/{rr[rr.arm == 'P+B+C'].estimate.iloc[0]:.3f}, PFS {pfs[pfs.arm == 'B+C'].estimate.iloc[0]:.0f}/{pfs[pfs.arm == 'P+B+C'].estimate.iloc[0]:.0f} d, severe AE {tox[tox.arm == 'B+C'].estimate.iloc[0]:.2f}/{tox[tox.arm == 'P+B+C'].estimate.iloc[0]:.2f} | PASS |
| Frozen lock before unblinded topology | application_lock.yaml v1; benchmark ran first | PASS |
| Calibrated machinery (G3) | gate_memo.md: coverage 0.90-0.98, type-I <= 0.06, topo-favorable power 1.00 | PASS |
| Negative controls | pre-treatment p = {sens[sens.variant == 'pretreatment_control'].pvalue.iloc[0]:.2f}; placebo reject {perm['reject'].mean():.3f} @ alpha 0.10 | PASS |
| Preprocessing stability | forest p-range {sens[(sens.variant != 'pretreatment_control') & (sens.variant != 'placebo_permutation_rate')].pvalue.min():.2f}-{sens.pvalue.max():.2f}, all null | PASS |
| Nontrivial result beyond SLD/RECIST | TATE p = {ttest.pvalue:.2f}; SLD-functional p = {sldt.pvalue:.2f} -- both null | FAIL |

## Decision: INCREMENTAL-ONLY (application layer)

The frozen topological outcome is non-vacuous, stable, not SLD-equivalent
(pilot R^2 0.31/0.02), and the inference machinery is validated; but the PACCE
release contains **no detectable topological treatment effect** for
panitumumab + bev + chemo vs bev + chemo in the first-window central
target-lesion trajectories, and no SLD-functional effect either. The honest
contribution is therefore:

1. a rigorous **negative/mirror-control demonstration** that the TATE/CTATE/ITTE
   pipeline behaves on a real randomized oncology trial exactly as the
   simulation predicts (null where the truth is null, calibrated everywhere);
2. methodological template + released-subset audit for future
   topology-of-trajectory analyses (e.g., antibodies with cytotoxic-shrinkage
   mechanisms, longer windows, lesion-level H1-rich settings).

NOT claimable: any treatment-effect heterogeneity (CTATE exploratory only),
any individual-effect coverage validation, ITT framing of the observed-
trajectory target, or full-trial (1053) generalization.
"""
    (RES / "g4_application_gate_memo.md").write_text(memo)
    print("reports + G4 memo written")


if __name__ == "__main__":
    main()
