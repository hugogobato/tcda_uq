# Changelog

## 0.1.0 - 2026-10-11

This is a breaking mathematical and API migration from the 0.0.1 library. The revised defaults and interfaces make each inferential target explicit and match the revised statistical formulation used in the project.

- Mean confidence bands use independently randomized or prespecified folds and Gaussian, unstudentized multiplier calibration by default. The Gaussian process and nuisance conditions remain assumptions; the default does not make them automatic.
- Pini–Vantini interval-wise inference is exposed as a hypothesis test with adjusted p-values. It is not returned as a confidence band. A confidence set requires a separately justified inversion.
- ITTE conformal prediction declares a target population (`full`, `overlap`, or `matching`). The held-out test population and the calibration weights must correspond to that target; training estimates are fixed independently before calibration. Nonconstant scales and estimated weights are fit on training data, while the default unit scale is fixed by the score design.
- Known likelihood ratios give a finite-sample marginal lower bound under the split and identification assumptions. Estimated ratios require a total-variation qualification for the intended target. Independence of the fitted weights alone does not establish exact coverage or double robustness.
- Silhouette utilities use plain normalized tents with exponent `r >= 1` by default. Values with `r < 1` are experimental and outside the stated topology result.
- Coverage and width summaries describe the prespecified grid. They do not certify coverage between grid points.

The revision reproduction runner preserves the earlier paper-run source in its own frozen environment. Use the instructions in [the reproduction guide](docs/revision_reproduction.md) when reproducing those runs; current package defaults are not a substitute for the frozen source behavior.
