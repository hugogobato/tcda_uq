# Paper results and reproduction map

This guide accompanies *Uncertainty Quantification for Topological Treatment Effects*, by Hugo G. Souto and Ioannis Diamantis. The manuscript and its arXiv submission package are maintained separately. The directories below contain public code and numerical evidence, not an arXiv submission.

## Final controlled experiments

| Study | Saved results | Supporting material |
| --- | --- | --- |
| Bounded one-bar design, n = 200, 500, 1000 | `reproduction/results/primary_production/bounded_n*` | Manifest, 2,000 replications per setting, summaries, and truth arrays |
| Topological base and scale-4 design | `reproduction/results/primary_production/topological_base` and `topological_scale4_v2` | 200 replications per setting; use v2 for the scale-4 final production result |
| CT slice-pool benchmark, r = 1 | `reproduction/results/primary_production/sars_r1` | 200 replications; exact derived pools in `reproduction/results/empirical_pool_inputs` |
| Functional bias, three-degree design, weighting and target sensitivities | `reproduction/results/remaining_production/01_*` through `04_*` | Eleven configurations, each with 200 replications |
| Topological exponents r = 1, 2, 4, 5 | `reproduction/results/remaining_production/05_*` through `08_*` | 200 replications per exponent; r = 3 is the primary base run |
| ORBIT population-reference experiment | `reproduction/results/remaining_production/09_orbit_1000_points` | 200 replications with 1,000-point generated clouds |
| Seven PACCE-matched simulations | `reproduction/results/remaining_production/10_*` through `16_*` | Generated data, 200 replications each; native mean-zero replay sidecars are not extra replications |

There are 6,600 primary and 4,600 remaining fitted-sample replications, for a total of 11,200. [The reproduction guide](revision_reproduction.md) explains the exact source archives and fresh-run commands. `reproduction/validation/remaining_run_summary.csv` lists all 23 remaining configurations; `remaining_evidence_ledger.csv` and `remaining_oracle_precision.csv` contain aggregate performance and reference-precision diagnostics.

`python reproduction/verify_saved_results.py` checks the frozen source hashes, checkpoint identities and replay provenance. `python reproduction/verify_publication.py` checks the additional publication artifact hashes and the clinical aggregate allowlist. Neither command reruns the experiments or certifies a mathematical theorem.

## Clinical PACCE aggregates

Public clinical results are under [`reproduction/results/clinical_aggregates`](../reproduction/results/clinical_aggregates/README.md).

| Paper analysis | Public artifacts | Interpretation |
| --- | --- | --- |
| Main selected cohort, 566 eligible subjects | `main_cohort/tate_estimates.csv`, `tate_test.csv` | Adjusted observed-group topological mean contrast; the saved sup-test p-value is approximately 0.982 |
| Log-summed-diameter comparator | `main_cohort/sldfun_tate_estimates.csv`, `sldfun_tate_test.csv` | Distinct constructed response; p-value approximately 0.175 |
| Corrected six fixed CTATE profiles, age 61 | `coauthor_revision_20261026/ctate_profile_summary.csv` | Current conditional-mean working-regression table; smallest sup-test p-value 0.432, all Bonferroni values 1.000; no model-free conditional guarantee |
| Independent-fold sensitivity | `coauthor_revision_20261026/independent_fold_sensitivity.csv` | Original stratified fit plus five independent two-fold fits; aggregate mean diagnostics only |
| Superseded age 6.1 CTATE profiles | `main_cohort/ctate_curves.csv`, `ctate_profile_summary.csv`, `ctate_profiles.yaml` | Historical age-unit error; retained for provenance, not the current manuscript table |
| Clinical benchmarks and controls | Other `main_cohort/*.csv` | Aggregate clinical benchmarks, baseline summaries, placebo and sensitivity diagnostics |
| Held-out scaling sensitivity, 419 eligible analysis subjects | `split_sample/pacce_pilot_grid1_basis5` | TATE curve, fixed-profile diagnostics, sensitivity results, and sanitized provenance; different representation and population from the main 566-subject fit |
| Second-stage basis sensitivity | `split_sample/pacce_pilot_grid1_basis3`, `pacce_pilot_grid1_basis9` | Same held-out design, different CTATE working basis |
| Additional endpoint sensitivity | `split_sample/pacce_selected_basis3`, `pacce_selected_basis5`, `pacce_selected_basis9` | Supplemental runs using endpoint 0.83; these are not the manuscript's main 566-subject fit |

The two PACCE figures in `reproduction/figures/` show aggregate mean contrasts and confidence bands. Clinical ITTE curves, per-subject arrays, identifier lists and patient-level manifests are excluded. The source dataset is not redistributed; see [data availability](data_availability.md).

The current corrected clinical entry point is documented in [the PACCE correction guide](pacce_coauthor_revision.md). It uses age 61 in years and a score scale fixed at one before calibration. Historical calibration-derived modulation was part of the old scores, not merely a separate diagnostic. Its clinical prediction widths are superseded; no individual clinical prediction results are published.

The original main-cohort calculations use the retained adapters in `reproduction/legacy/experiments/pacce_*.py`. The held-out calculations use `revision_20261010/rerun/pacce_reanalysis.py` inside the already published `reproduction/sources/run_sources_5fdce0b48fe341aee1f1b31312ce5a35a27872577e5ed2ed44d25fc3cbf45625.zip`. The sanitized result manifests identify the source hashes, software versions, seeds and settings. Authorized investigators must run these clinical adapters in a private workspace: their complete local output includes patient manifests and prediction objects that are not part of the public allowlist.

## Retained historical diagnostics

`reproduction/legacy/out/` preserves earlier console records, aggregate tables and image-pool/ORBIT outputs. `reproduction/legacy/figures/` retains their figures, and `reproduction/legacy/experiments/` retains the available adapter source snapshot. These files support earlier benchmark and diagnostic panels described in the paper, including confidence-versus-prediction, smoothing, weak-overlap and topology stress examples. They predate the final production program; some use older conventions or interfaces. Their archived console interpretations are historical and must be read with the paper's corrected inferential scope.

The adapter snapshot is retained for transparency. It does not have the complete source-hash and checkpoint provenance of the final frozen runs, and executing it against the revised public package may require the historical API. Use the final frozen runs for the documented 11,200-replication reproduction path.

For the earlier functional TATE table, use the seven `legacy/out/phase7/phase2_shard*.txt` records and `legacy/verified/verified_tate_aggregate.csv`. For the earlier CTATE table, use the four equal-sized `legacy/out/phase3/shard*.txt` records, their `aggregated.txt`, and `legacy/verified/verified_ctate_aggregate.csv`. The separate historical `legacy/out/phase7/phase3_aggregated.txt` averages rounded summaries across unequal shard sizes and is retained only as an obsolete record; it is not the source of the final table. The Level-B stress table is backed by `legacy/out/phase7_level_b/level_b.csv` and its log. The confidence-versus-prediction illustration is generated directly by `legacy/experiments/phase5_figures.py`.

`reproduction/results/exploratory/production/` retains the superseded scale-4 attempt and a runtime pilot. `reproduction/results/exploratory/runtime_pilots/` preserves the subsequent timing pilots. These runs are not included in the final replication totals and should not be pooled with production estimates. Additional retained result figures, including unused diagnostic panels, are under `reproduction/figures/`; the figure filenames distinguish these panels from final production results.

`reproduction/publication_files.json` maps the additional public artifacts to their original relative paths and records SHA-256 hashes. Converted clinical curve CSVs retain 17 significant digits. No experiment results are excluded through `.gitignore`.
