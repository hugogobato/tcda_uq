# Phase 7 Verdict

Synthetic Level A is the coverage backbone because it has known TATE, CTATE, and ITTE truth.
Real-data rows are demonstrations unless the row explicitly says it has synthetic counterfactual truth.

ITTE: conformal prediction averages 0.996 coverage in parsed rows, while the confidence-band baseline averages 0.0475 for individual curves.
Level B (real persistence -> silhouette pipeline): ITTE conformal averages 0.9966 coverage across the topological sweep, while the asymptotic TATE multiplier band averages 0.807 -- near nominal for the marginal mean but degrading under weak overlap, since the functional-linear nuisance is misspecified for the topological DGP and its finite-sample bias eats into the narrower (pointwise) confidence widths. The pointwise CTATE-mean band under-covers more sharply (0.1106) because its linear second stage cannot fit the step-function loop-count conditional mean. The distribution-free conformal prediction band stays valid on the same misspecified nuisance -- the topology-specific motivation for the conformal side.
Weak overlap: at overlap scale 4.0, stabilized overlap weighting cuts dim-averaged unbounded bands from 34% to 0% in the parsed Phase 6 table.
Use asymptotic confidence bands for TATE and CTATE means; their widths target estimation error and shrink with n.
Use conformal prediction bands for ITTE or CTATE-level draws; confidence bands are the wrong object for individual curves.
Use adaptive conformal when aleatoric spread varies with X, and stabilized weighting when overlap is thin.
GEOM-Drugs Phase 7B.3 is intentionally absent here because the dataset was not ready.
