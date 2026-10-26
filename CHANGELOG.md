# Changelog

## PACCE co-author corrections - 2026-10-26

Added a portable aggregate-only clinical reproduction entry point using authenticated frozen sources, explicit age 61 profiles, constant-one scores fixed before calibration, silhouette support intersections, and optional independently randomized folds. Published only corrected aggregate CTATE and fold summaries; no clinical prediction-output files, patient records, or identifiers are added. Historical age 6.1 profile tables and legacy scripts remain available as superseded provenance. The current library already uses training-frozen conformal scales; this update connects those conditions to the corrected clinical reproduction procedure. Added a staged-file release guard and clinical-only optional dependency.

## Paper submission snapshot - 2026-10-15

Added aggregate clinical mean outputs, retained historical experiment results, and a paper-to-results guide. The manuscript and arXiv submission files are maintained separately from this repository. Clinical ITTE curves, patient records, and subject identifiers remain excluded. The frozen production sources and all 11,200 production replications are preserved. The Python API and software version remain 0.1.0.

## 0.1.0 - 2026-10-11

This is a breaking mathematical and API migration from the 0.0.1 library. The revised defaults and interfaces make each inferential target explicit and match the revised statistical formulation used in the project.

- Mean confidence bands use independently randomized or prespecified folds and Gaussian, unstudentized multiplier calibration by default. The Gaussian process and nuisance conditions remain assumptions; the default does not make them automatic.
- Pini–Vantini interval-wise inference is exposed as a hypothesis test with adjusted p-values. It is not returned as a confidence band. A confidence set requires a separately justified inversion.
- ITTE conformal prediction declares a target population (`full`, `overlap`, or `matching`). The held-out test population and the calibration weights must correspond to that target; training estimates are fixed independently before calibration. Nonconstant scales and estimated weights are fit on training data, while the default unit scale is fixed by the score design.
- Known likelihood ratios give a finite-sample marginal lower bound under the split and identification assumptions. Estimated ratios require a total-variation qualification for the intended target. Independence of the fitted weights alone does not establish exact coverage or double robustness.
- Silhouette utilities use plain normalized tents with exponent `r >= 1` by default. Values with `r < 1` are experimental and outside the stated topology result.
- Coverage and width summaries describe the prespecified grid. They do not certify coverage between grid points.

The revision reproduction runner preserves the earlier paper-run source in its own frozen environment. Use the instructions in [the reproduction guide](docs/revision_reproduction.md) when reproducing those runs; current package defaults are not a substitute for the frozen source behavior.
