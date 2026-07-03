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

```bash
uv venv --python 3.10
uv pip install -e ".[dev]"
```

Optional extras:

```bash
uv pip install -e ".[dev,data]"    # SARS-CoV-2 image loader
uv pip install -e ".[dev,graphs]"  # graph dependencies for future GEOM-style work
```

Run the smoke tests:

```bash
.venv/bin/python -m pytest
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

## Package Layout

```text
datasets/      synthetic oracle DGPs and lightweight data helpers
estimators/    plug-in, IPW, AIPW, cross-fitting, CTATE DR-learner
metrics/       band, coverage, width, and plotting helpers
silhouette/    persistence diagram / point cloud / image to silhouette tools
uq/
  asymptotic/  confidence-band methods
  conformal/   conformal prediction methods
tests/         fast smoke tests
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
