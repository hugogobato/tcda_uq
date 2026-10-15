# Clinical aggregate release

`main_cohort/` contains aggregate results for the manuscript's 566-subject observed-cohort analysis. Grid curves are exported from the aggregate Parquet tables to CSV. `ctate_profile_summary.csv` is the CTATE-only six-row source table, renamed from a misleading local filename that mentioned ITTE. `ctate_profiles.yaml` defines the queried working-regression profiles; these are not individual patient records.

`split_sample/` contains aggregate results for six held-out-scaling runs. The `pacce_pilot_grid1_basis*` runs use endpoint 1.0 and support the appendix's held-out sensitivity analysis. The `pacce_selected_basis*` runs use endpoint 0.83 and are retained as supplementary sensitivity runs; they do not produce the paper's main 566-subject estimate.

Each split-sample directory contains only `adjusted_selected_contrast.csv`, `fixed_profile_diagnostics.csv`, `selection_weight_sensitivity.csv`, and sanitized `scope.json` and `manifest.json`. Identifier lists, patient manifests, prediction-curve files, input-file fingerprints and local SAS/output paths have been removed. All PACCE ITTE output files are excluded.

The analysis is conditional on observed eligibility and working statistical models; these files do not establish equivalence, causal identification after observation selection, or individual counterfactual coverage. See [the paper results guide](../../../docs/paper_results.md) and [data availability](../../../docs/data_availability.md) for provenance and access to authorized source data.
