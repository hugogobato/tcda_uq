# tcda_uq

`tcda_uq` is a self-contained Python library for uncertainty quantification of
topological treatment effects.

It supports three estimands:

| Estimand | Meaning | Uncertainty object |
|---|---|---|
| TATE `psi_d(t)` | marginal average topological effect | confidence band |
| CTATE `tau_d(t, x)` | conditional average topological effect | confidence band for the mean; adaptive prediction band for a draw |
| ITTE `delta_{i,d}(t)` | individual topological treatment effect | conformal prediction band |

The package includes:

- persistence-to-silhouette utilities for point clouds, diagrams, and images;
- plug-in, IPW, and AIPW topological treatment-effect estimators;
- cross-fitting and functional nuisance regression;
- multiplier-bootstrap, Liebl-Reimherr, and Pini-Vantini confidence bands;
- functional conformal prediction, causal weighted conformal prediction, and
  positivity-stabilized ITTE bands;
- synthetic oracle datasets for testing coverage.

This repository is only the reusable library. Research plans, manuscripts,
large experiment outputs, and data downloads are intentionally not included.

## Install

Install directly from GitHub:

```bash
uv venv --python 3.10
uv pip install "tcda_uq @ git+https://github.com/hugogobato/tcda_uq.git"
```

For development from a local clone:

```bash
git clone https://github.com/hugogobato/tcda_uq.git
cd tcda_uq
uv venv --python 3.10
uv pip install -e ".[dev]"
```

Optional extras from a local clone:

```bash
uv pip install -e ".[dev,data]"    # SARS-CoV-2 image loader
uv pip install -e ".[dev,graphs]"  # graph dependencies for future GEOM-style work
```

Optional extras directly from GitHub:

```bash
uv pip install "tcda_uq[data] @ git+https://github.com/hugogobato/tcda_uq.git"
uv pip install "tcda_uq[graphs] @ git+https://github.com/hugogobato/tcda_uq.git"
```

Conda users can instead create the environment from `environment.yml` (it
pip-installs the package; `pyproject.toml` remains the single source of truth for
pinned versions, and the fragile `pyg` conda pin is intentionally dropped - see
the file header):

```bash
conda env create -f environment.yml
conda activate tcda_uq
```

## Testing

Two lanes. The default run is fast (smoke + unit + API tests); the statistical
coverage-property tests drive the full nuisance pipeline over many replicates and
are marked `slow`:

```bash
pytest              # fast: smoke + unit + unified-API tests
pytest -m slow      # coverage-property regression tests on the tri-oracle harness
```

## Quickstart: TATE Confidence Band

```python
from tcda_uq.datasets import TriOracleSimulation
from tcda_uq.estimators import cross_fit
from tcda_uq.uq.asymptotic import multiplier_bootstrap_band

sim = TriOracleSimulation(seed=0)
sample = sim.sample(300, rng=0)
fit = cross_fit(sample.observed, sim.tseq, n_basis=5)

d = 1
band = multiplier_bootstrap_band(
    fit.influence()[d],
    sim.tseq,
    fit.aipw[d],
    alpha=0.10,
)

print(bool(band.covers(sim.true_tate()[d])))
print(band.mean_width())
```

## Quickstart: ITTE Prediction Band

```python
from tcda_uq.datasets import TriOracleSimulation
from tcda_uq.uq.conformal import ITTEConformal, make_weight_fn

sim = TriOracleSimulation(seed=0)
train = sim.sample(800, rng=1)
test = sim.sample(200, rng=2)

model = ITTEConformal.fit(
    train.observed,
    sim.tseq,
    n_basis=5,
    weight_fn=make_weight_fn("overlap"),
)

lo, hi, center = model.band_bounds(test.X, alpha=0.10, d=1)
target = test.oracle_itte[:, 1, :]
coverage = ((target >= lo) & (target <= hi)).all(axis=1).mean()

print(coverage)
```

## Unified Interface

A thin, non-breaking convenience layer bands and evaluates all three estimands
through one surface (the per-method functions/classes remain the reference API):

```python
from tcda_uq import tate_band, ctate_band, itte_band, evaluate_coverage

# TATE confidence band (method in {"mbb", "pv", "lr"}), then coverage vs truth
band = tate_band(cross_fit_result, method="mbb", d=1, alpha=0.10)
print(evaluate_coverage(band, sim.true_tate()[1]))
# CoverageResult(coverage=... @ level=0.900 ..., mean_width=..., n=1, confidence)

band = ctate_band(ctate_learner, x, d=1, alpha=0.10)   # confidence, per-x
band = itte_band(itte_model, x, d=1, alpha=0.10)       # prediction
```

`evaluate_coverage` accepts a `Band` or a raw `(lower, upper)` tuple and a single
truth curve or a batch, returning simultaneous coverage + mean width.

## Reproducing The Coverage Claims

A self-contained, CPU-only script regenerates the headline coverage/width table
for all three estimands on the tri-oracle harness:

```bash
python scripts/reproduce_coverage.py --quick      # fast smoke
python scripts/reproduce_coverage.py              # default (a few minutes)
python scripts/reproduce_coverage.py --out coverage.csv
```

(The full production tables and figures live in the research workspace, not in
this library repo.)

## Tutorials

Executable notebooks under `notebooks/` - one per estimand and one per dataset:

| Notebook | Focus |
|---|---|
| `01_tate_confidence_bands.ipynb` | TATE confidence bands (mbb / PV / LR) |
| `02_ctate_bridge.ipynb` | CTATE DR-learner + confidence-vs-prediction at x |
| `03_itte_conformal.ipynb` | ITTE conformal prediction + positivity stabilization |
| `04_orbit_pipeline.ipynb` | ORBIT point clouds -> persistence -> silhouette -> UQ |
| `05_sarscov2_demo.ipynb` | SARS-CoV-2 CT images (needs the `[data]` extra) |

Execute them in place with `make notebooks` (or
`jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb`).
Refresh the notebook JSON from the checked-in sources with
`python scripts/build_notebooks.py`.

## Package Layout

```text
api.py         unified tate_band / ctate_band / itte_band / evaluate_coverage layer
datasets/      synthetic oracle DGPs and lightweight data helpers
estimators/    plug-in, IPW, AIPW, cross-fitting, CTATE DR-learner
metrics/       band, coverage, width, and plotting helpers
silhouette/    persistence diagram / point cloud / image to silhouette tools
uq/
  asymptotic/  confidence-band methods
  conformal/   conformal prediction methods
tests/         fast unit/API tests + `slow` coverage-property tests
scripts/       reproduce_coverage.py and build_notebooks.py
notebooks/     executable tutorials (one per estimand, one per dataset)
```

## Notes On Optional Backends

The Liebl-Reimherr confidence band has a self-contained NumPy implementation.
If you want to compare against the original FFSCB R implementation, install or
clone FFSCB separately and set `FFSCB_R_DIR` to its `R/` directory. FFSCB is not
vendored in this library repository.

## License

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`.
