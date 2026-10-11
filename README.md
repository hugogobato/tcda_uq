# tcda_uq

`tcda_uq` contains Python tools and reproducibility materials for uncertainty quantification of topological treatment effects. The library represents each outcome by a prespecified persistence pipeline and a power-weighted silhouette on a fixed filtration grid. It supports three different targets:

| Target | Quantity | Statistical object |
|---|---|---|
| TATE | Population mean effect curve, `psi_d(t)` | Asymptotic confidence band |
| CTATE | Conditional mean effect curve, `tau_d(t, x)`, at a fixed `x` | Asymptotic confidence band for that mean curve |
| ITTE | A new unit's individual effect curve, `delta_d(t)` | Split-conformal prediction band, marginal over the declared target population |

Confidence bands for means and prediction bands for individual curves answer different questions. The computed summaries are on the prespecified grid. Grid inclusion does not by itself establish coverage over a continuous interval; see [targets, assumptions, and scope](docs/targets_and_assumptions.md).

The package also includes the revision's frozen-source rerun materials and all 11,200 completed production replications. The [reproduction guide](docs/revision_reproduction.md) explains the saved evidence, its validation, and the 16 self-contained Colab presets. The rerun environment preserves the source used for the paper runs; the current package implements the revised defaults and interface.

## Install

Install from GitHub:

```bash
uv venv --python 3.10
uv pip install "tcda_uq @ git+https://github.com/hugogobato/tcda_uq.git"
```

For local development:

```bash
git clone https://github.com/hugogobato/tcda_uq.git
cd tcda_uq
uv venv --python 3.10
uv pip install -e ".[dev]"
```

The optional `[data]` extra supports the SARS-CoV-2 image example. `[graphs]` installs additional graph-related libraries. The package's core methods do not require either extra.

Conda users can create the development environment from the repository root:

```bash
conda env create -f environment.yml
conda activate tcda_uq
```

`pyproject.toml` defines the Python and package dependencies. The notebooks use small synthetic examples by default; the CT image tutorial reads local data only when `TCDA_UQ_SARS_ROOT` is set.

## Quick start: TATE mean confidence band

The TATE band uses Gaussian multipliers and an unstudentized supremum by default. Its asymptotic guarantee requires the process, nuisance-estimation, and nondegeneracy conditions stated in the accompanying theory. Folds are independently randomized by default; the example makes that choice explicit.

```python
from sklearn.linear_model import LogisticRegression

from tcda_uq import evaluate_coverage, tate_band
from tcda_uq.datasets import TriOracleSimulation
from tcda_uq.estimators import cross_fit

sim = TriOracleSimulation(seed=0, resolution=40)
sample = sim.sample(400, rng=1)
fit = cross_fit(
    sample.observed,
    sim.tseq,
    n_basis=5,
    n_splits=2,
    random_state=2,
    stratify=False,
    propensity_estimator=LogisticRegression(max_iter=500),
)

band = tate_band(fit, method="mbb", d=0, alpha=0.10, n_boot=500, rng=3)
result = evaluate_coverage(band, sim.true_tate()[0])
print(result.coverage)  # inclusion on sim.tseq for this synthetic example
print(band.mean_width())
```

The value returned by `evaluate_coverage` is gridwise. A single synthetic run illustrates the API; it does not validate the asymptotic assumptions.

## Interval-wise testing

The Pini–Vantini interface returns adjusted and unadjusted p-values for a null hypothesis on the mean curve. It is a testing procedure, not a confidence band. Without a separately implemented inversion, report p-values, type-I error, and power rather than confidence-band coverage or width.

```python
from tcda_uq import tate_test

iwt = tate_test(fit, method="iwt", d=0, alpha=0.05, n_boot=500, rng=4)
adjusted_p_values = iwt.p_adjusted
rejected_grid_points = iwt.significant(adjusted=True)
```

The test's guarantees concern its prespecified grid and its stated interval-wise error control. They do not imply simultaneous confidence coverage over the continuum.

## Quick start: ITTE prediction for a declared target

The example below uses the simulator's known propensity and draws test units from the same full-population target used for calibration. `ITTEConformal.fit` separates training and calibration data internally. Its default unit score scale is fixed in advance; a nonconstant scale must be estimated using training data before calibration. With known likelihood ratios and the stated split assumptions, the guarantee is a finite-sample lower bound on marginal prediction coverage, not exact equality and not coverage conditional on every fixed `x`.

```python
from tcda_uq import evaluate_coverage
from tcda_uq.datasets import TriOracleSimulation
from tcda_uq.uq.conformal import ITTEConformal

sim = TriOracleSimulation(seed=7, resolution=40)
train = sim.sample(900, rng=8)
test = sim.sample_target(250, target="full", rng=9)

model = ITTEConformal.fit(
    train.observed,
    sim.tseq,
    n_basis=5,
    target="full",
    propensity_fn=sim.propensity_score,
    weights_known=True,
    random_state=10,
)
lo, hi, _ = model.band_bounds(test.X, alpha=0.10, d=0)
result = evaluate_coverage((lo, hi), test.oracle_itte[:, 0, :], level=0.90)
print(model.target_, model.weights_known_)
print(result.coverage)  # empirical inclusion on the declared target and grid
```

The `target` argument supports `"full"`, `"overlap"`, and `"matching"`. Draw evaluation data with `sample_target` from the same target used by the model, for example `sample_target(..., target="overlap")` when the model was fitted with `target="overlap"`. Changing the weights changes the population being predicted. Estimated propensity ratios target an estimated distribution; the revised guarantee for the intended population is qualified by the total-variation discrepancy of those normalized ratios. Independent fitting alone does not make estimated ratios exact, and the package does not claim double robustness for weighted conformal coverage. Bounded weights can still yield an infinite radius at finite sample sizes because the target point contributes mass at infinity.

## Tutorials and reproduction

The five source-generated notebooks cover TATE confidence bands, fixed-`x` CTATE mean inference, target-matched ITTE prediction, ORBIT point clouds, and a SARS-CoV-2 CT image pipeline. They are checked in as clean source notebooks without saved execution output. Rebuild their JSON from the Python cell sources with:

```bash
python scripts/build_notebooks.py
```

Execute them locally with `make notebooks`. The CT image example may use images from a local directory, but it constructs treatment assignment and potential outcomes synthetically. It does not estimate clinical treatment effects or evaluate counterfactual coverage in observed patients.

The small simulation diagnostic can be run with:

```bash
python scripts/reproduce_coverage.py --quick
python scripts/reproduce_coverage.py --out coverage.csv
```

It reports finite-grid Monte Carlo inclusion frequencies and width diagnostics. Its ITTE comparisons use known simulator propensities and target-matched test samples. These outputs are numerical diagnostics for the specified design, not proofs of the theorem assumptions.

## Package layout

```text
api.py         estimand-specific confidence, testing, prediction, and evaluation API
datasets/      synthetic data and optional modality loaders
estimators/    AIPW estimation, independent cross-fitting, and CTATE regression
metrics/       bands, gridwise inclusion, widths, and plotting helpers
silhouette/    persistence diagram / point cloud / image to silhouette tools
uq/            asymptotic mean inference and split-conformal prediction
scripts/       simulation diagnostics and notebook source builder
notebooks/     executable tutorials with no embedded datasets or saved outputs
reproduction/  frozen source and paper-run artifacts for revision reproducibility
docs/          target assumptions, guarantee scope, and reproduction guide
```

## Citation and license

See [CITATION.cff](CITATION.cff) for the software citation metadata. The package is released under the MIT License; see [LICENSE](LICENSE).
