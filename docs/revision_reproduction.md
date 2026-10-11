# Reproducing the revision studies

The repository's `reproduction/` directory contains the frozen-source runner and records for the revision studies. Use its source bundle and manifest when reproducing the paper runs. The current `tcda_uq` package has a revised API and defaults; it is not a replacement for the frozen implementation used by those runs.

From the repository root, inspect available studies and runner options with:

```bash
python reproduction/run_revision.py --list
```

To run the 23 additional study configurations on a local machine:

```bash
python reproduction/run_revision.py --study all --workers 4
```

The six primary settings use their own original source archives, mapped in `reproduction/sources/primary_source_mapping.json`. Extract the mapped archive into a separate workspace before running its `revision_20261010/rerun/revised_experiments.py`; its CLI accepts the scientific parameters recorded in the corresponding saved manifest. Reproduction needs the same source archive, dependency versions, scientific parameters and seeds, with a new output path. The SARS benchmark also uses the derived plain-tent arrays and metadata in `reproduction/results/empirical_pool_inputs/`. Those vectors fix the available 1,252/1,229-slice empirical pools and contain no images or patient identifiers. The clinical PACCE reanalysis requires the authorized original SAS inputs, which are not distributed here.

The exact source archive and checksums are recorded in `reproduction/revision_source.zip` and `reproduction/sources.json`. The self-contained notebooks in `reproduction/colab/` are optional execution presets. The recorded outputs are organized under `reproduction/results/primary_production/` and `reproduction/results/remaining_production/`. Check each preset's manifest for its status; the presence of a results directory alone does not mean that every configured run has completed.

The saved production program is complete: 6,600 primary replications and 4,600 additional replications across 23 configurations, for 11,200 fitted samples. The 1,400 PACCE mean-zero test sidecars replay the mean stage of the same simulated samples and are not extra replications. Pilot runs and API execution checks are recorded separately and excluded. Independent summaries and reference-precision diagnostics are in `reproduction/validation/`; Monte Carlo standard errors use the fitted sample as the replication unit. Finite Monte Carlo references introduce additional error in the topological, ORBIT and PACCE mean diagnostics.

Verify the saved checkpoint identities, archive hashes and PACCE replay provenance with:

```bash
python reproduction/verify_saved_results.py
```

Fresh local outputs are saved in `reproduction/results/new_runs/`, separately from the published records. The 16 Colab notebooks are self-contained fresh-run templates; their initial execution status does not describe the already completed local records. Each notebook retains checkpoints and downloads one ZIP, with a safe fallback outside Colab.

To repeat the PACCE mean-stage tests after producing complete fresh runs, extract the exact source bundle and pass both paths explicitly:

```bash
python -m zipfile -e reproduction/revision_source.zip /tmp/tcda_uq_mean_replay_source
python reproduction/augment_pacce_mean_tests.py --source-root /tmp/tcda_uq_mean_replay_source --runs-root reproduction/results/new_runs/remaining_production --require-complete --workers 2
```

The replay verifies agreement with each original mean fit before saving the native absolute-supremum zero-mean rejection event. It preserves the original checkpoint. Mean confidence calibration always uses the known generator propensity in these controlled runs; the known/learned/constant weight sensitivity changes prediction weights only.

The result records include synthetic studies and a historical aggregate PACCE standardization JSON from an existing cohort analysis. Those cohort-scale summaries are legacy aggregate values, not new clinical evidence. Raw clinical data and subject identifiers are not part of the public reproduction materials. The manuscript draft and PDF are maintained separately and are not included in this code and reproducibility publication.

The frozen runner, source archive, checksums, and recorded outputs define the reproducibility path. Tutorials and the small `scripts/reproduce_coverage.py` diagnostic demonstrate the current public API and are not substitutes for the frozen runs or their recorded result files.
