# Colab reproduction presets

These 16 self-contained notebooks request 23 scientific configurations, with
200 replications per configuration. Upload any notebook to Colab and use Run
All. No repository clone, selector edit or clinical-data upload is required.
Each notebook includes its checksummed source, dependency installation, resume
option and a safe single-ZIP output download. The initial NOT RUN status
describes a fresh notebook execution.

All 23 configurations were completed locally. Saved production records are in
`../results/remaining_production/`, with original independent validation in
`../validation/`. These fitted samples are part of the 11,200-record production
program. Pilot runs and deterministic mean-stage replays are excluded.

| Notebook | Fixed request |
| --- | --- |
| `01_functional_step_bias` | Functional generator with the treatment-arm step perturbation, n=500, degree 2, known propensity, and full/overlap/matching targets. |
| `02_functional_degree3_base` | Functional base generator, n=500, degree 3, known propensity, and full/overlap/matching targets. |
| `03_functional_weight_target_sensitivities` | Six configurations: propensity scales 1 and 4 crossed with known, learned, and misspecified prediction weights. Each uses n=500, degree 2, and all six target modes. |
| `04_topological_scale4_weight_target_sensitivities` | Three configurations crossing known, learned, and misspecified prediction weights at propensity scale 4. Each uses topological r=3, n=500, 40 points per loop, degree 2, and all six target modes. |
| `05_topological_r1` | Topological generator at r=1, n=500, 40 points per loop, degree 2, propensity scale 1, known weights, and full/overlap/matching targets. |
| `06_topological_r2` | Topological generator at r=2, n=500, 40 points per loop, degree 2, propensity scale 1, known weights, and full/overlap/matching targets. |
| `07_topological_r4` | Topological generator at r=4, n=500, 40 points per loop, degree 2, propensity scale 1, known weights, and full/overlap/matching targets. |
| `08_topological_r5` | Topological generator at r=5, n=500, 40 points per loop, degree 2, propensity scale 1, known weights, and full/overlap/matching targets. |
| `09_orbit_1000_points` | ORBIT generator with n=300, 1,000 points, degree 2, r=3, propensity scale 1, known weights, and full/overlap/matching targets. |
| `10_pacce_null` | Controlled PACCE simulation scenario null, n=842, degree 1, 40 points, scale 1, known weights, and full/overlap/matching targets. This uses the embedded simulation generator and requires no clinical upload. |
| `11_pacce_sld_only` | Controlled PACCE simulation scenario sld_only, n=842, degree 1, 40 points, scale 1, known weights, and full/overlap/matching targets. This uses the embedded simulation generator and requires no clinical upload. |
| `12_pacce_topo_favorable` | Controlled PACCE simulation scenario topo_favorable, n=842, degree 1, 40 points, scale 1, known weights, and full/overlap/matching targets. This uses the embedded simulation generator and requires no clinical upload. |
| `13_pacce_topo_null` | Controlled PACCE simulation scenario topo_null, n=842, degree 1, 40 points, scale 1, known weights, and full/overlap/matching targets. This uses the embedded simulation generator and requires no clinical upload. |
| `14_pacce_nuisance_misp` | Controlled PACCE simulation scenario nuisance_misp, n=842, degree 1, 40 points, scale 1, known weights, and full/overlap/matching targets. This uses the embedded simulation generator and requires no clinical upload. |
| `15_pacce_irregular_visits` | Controlled PACCE simulation scenario irregular_visits, n=842, degree 1, 40 points, scale 1, known weights, and full/overlap/matching targets. This uses the embedded simulation generator and requires no clinical upload. |
| `16_pacce_dropout_arm` | Controlled PACCE simulation scenario dropout_arm, n=842, degree 1, 40 points, scale 1, known weights, and full/overlap/matching targets. This uses the embedded simulation generator and requires no clinical upload. |

From a repository checkout, use `python reproduction/run_revision.py --list`
and `python reproduction/run_revision.py --study all --workers 4` for fresh
local runs. See `docs/revision_reproduction.md` for source scope, primary runs,
mean-stage test replay and validation. Generated outputs remain visible to Git.

The shared notebook budget is 8.5 hours with two workers, leaving time for ZIP
packaging. Incomplete commands and checkpoint counts are reported explicitly.
The ORBIT preset takes about 2–3 hours locally with two workers; Colab timing
can differ. Use disjoint shard IDs only after measuring runtime, and preserve
all scientific parameters and seeds across shards.

These are finite-grid diagnostics. Estimated weights, numerical-reference
error, working-model bias and real clinical selection retain the qualifications
in the reproduction guide and statistical-scope documentation.

Run `python reproduction/presets/build_colab_remaining.py` to regenerate the
notebooks and this directory's single ZIP from the embedded source template.
