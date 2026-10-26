# Corrected clinical PACCE reproduction, 26 October 2026

The current entry point is `reproduction/pacce_coauthor_revision.py`. It reproduces the co-author revision while keeping the original numerical implementation fixed in an authenticated source snapshot. Historical `legacy/experiments/pacce_*.py` scripts are retained for provenance and must not be treated as the current clinical procedure.

## What changed

The old prediction implementation estimated a pointwise residual scale from calibration outcomes and reused it inside its scores. This did not satisfy the training-frozen score assumption. The corrected procedure fits nuisance regressions using the training half, fixes every score scale at one before calibration, computes calibration supremum residual scores, and composes armwise weighted bands with errors 0.05 each (nominal 90% under the theorem's assumptions). The current library defaults already enforce training-frozen scales; the new entry point explicitly implements the corrected clinical calculation.

The historical query profiles supplied age 6.1 although the adapter records age in years. Corrected profiles use age 61, with male sex, ECOG 0, two metastatic sites, wild-type KRAS, and normal LDH, crossing chemotherapy backbone and target-lesion counts 2, 4, 6. The population-average analysis does not query those rows and is unchanged. The corrected six-row CTATE summary and five-fold sensitivity are under `reproduction/results/clinical_aggregates/coauthor_revision_20261026/`; old clinical outputs are preserved and labeled as superseded.

The representation remains the historical H0, r=3, sqrt(2)-scaled silhouette, using 100 equally spaced points on [0, 0.83] and pooled coordinate scaling. Alpha-radius births are nonnegative, so standard tents lie in [0, t], silhouettes in [0, sqrt(2)t], and their differences in [-sqrt(2)t, sqrt(2)t]. Intersecting a composed band with this support preserves whatever coverage it already has. Corrected unrestricted mean full widths are 1.639–1.730; support intersection gives 1.053–1.090. These are standardized geometric-radius units, not millimeters or a direct estimate of individual-effect variability. No clinical prediction-output file is published.

The main adjusted fit uses treatment-stratified folds. The optional independent-fold sensitivity changes allocation while preserving the representation and working models. Across five independent allocations, the mean bands include zero at every node, with p-values 0.787–0.979. Neither those diagnostics nor support intersection certifies nuisance rates, fixes selection, or makes learned ratios exact. Clinical interpretation remains an adjusted observed-group comparison, without a claim of causal absence, equivalence, or conditional prediction coverage.

## Run with separately authorized inputs

Install the pinned scientific dependencies and the clinical SAS reader:

```sh
uv pip install -e ".[clinical]"
python reproduction/pacce_coauthor_revision.py --help
python reproduction/pacce_coauthor_revision.py --sas-dir /approved/pacce/sas --out-dir /approved/private/pacce-coauthor-20261026 --include-fold-sensitivity
```

The input directory must contain `demo.sas7bdat` and `lesion.sas7bdat`, obtained under the investigator's own authorization. The output directory must resolve outside the checkout. No clinical dataset is downloaded or embedded by this command. Computation uses a single process with one BLAS/OpenMP thread to avoid oversubscription.

Only aggregate diagnostics are written. Subject identifiers, actual individual trajectories, individual prediction bands, per-unit scores, and split membership arrays are never output by this entry point. The complete local outputs of older clinical adapters may contain restricted objects and must not be copied into the public tree.

The fixture mode exercises the same code with explicitly synthetic frames and small resampling budgets; its values are not clinical results. See the CLI help for its options. CI uses fixtures and aggregate checks only, without clinical inputs.

## Public verification

```sh
python reproduction/verify_publication.py
python reproduction/verify_pacce_release.py
```

The first command checks published artifact checksums and allowed clinical schemas. The second inspects staged Git blobs, including nested archives and notebook outputs, for files outside the release scope. The clinical dataset and individual clinical prediction results are excluded; simulation results remain available and are not hidden through `.gitignore`.
