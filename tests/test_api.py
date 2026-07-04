"""Tests for the non-breaking unified convenience layer (``tcda_uq.api``)
and the composed-band helpers that share its band interface.

Uses one small cross-fit fixture (module-scoped) so the class-based estimators
are exercised end-to-end without repeated fitting.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import tcda_uq
from tcda_uq import (
    CoverageResult,
    ctate_band,
    evaluate_coverage,
    itte_band,
    tate_band,
)
from tcda_uq.datasets import TriOracleSimulation
from tcda_uq.estimators import CTATEDRLearner, cross_fit
from tcda_uq.metrics import Band
from tcda_uq.uq.conformal import ITTEConformal, make_weight_fn, simultaneous_itte_bounds


@pytest.fixture(scope="module")
def harness():
    sim = TriOracleSimulation(n_hom_dim=2, resolution=24, n_basis=5, seed=0)
    sample = sim.sample(220, rng=1)
    fit = cross_fit(
        sample.observed, sim.tseq, n_basis=5, n_splits=2, random_state=0,
        propensity_estimator=LogisticRegression(max_iter=500),
    )
    return sim, sample, fit


# ------------------------------------------------------------- lazy re-exports
def test_top_level_lazy_exports():
    for name in ("tate_band", "ctate_band", "itte_band", "evaluate_coverage", "CoverageResult"):
        assert name in dir(tcda_uq)
        assert getattr(tcda_uq, name) is not None
    with pytest.raises(AttributeError):
        tcda_uq.does_not_exist


# ---------------------------------------------------------- evaluate_coverage
def test_evaluate_coverage_values():
    tseq = np.linspace(0, 1, 20)
    truth = np.sin(2 * np.pi * tseq)
    band = Band(tseq, truth - 0.5, truth + 0.5, center=truth, level=0.9, kind="confidence")
    r = evaluate_coverage(band, truth)
    assert isinstance(r, CoverageResult)
    assert r.coverage == 1.0 and r.mean_width == pytest.approx(1.0)
    assert r.level == 0.9 and r.gap == pytest.approx(0.1) and r.kind == "confidence"
    assert r.n_targets == 1
    bad = truth.copy()
    bad[5] += 10.0
    assert evaluate_coverage(band, np.vstack([truth, bad])).coverage == 0.5
    # tuple form + explicit level
    r2 = evaluate_coverage((band.lower, band.upper), truth, level=0.9)
    assert r2.coverage == 1.0 and r2.level == 0.9
    with pytest.raises(TypeError):
        evaluate_coverage((band.lower,), truth)


def test_evaluate_coverage_batched_bands():
    # per-unit (batched) bands vs per-unit truths broadcast row-by-row
    lower = np.array([[-1.0, -1.0], [0.0, 0.0]])
    upper = np.array([[1.0, 1.0], [2.0, 2.0]])
    truth = np.array([[0.5, -0.5], [5.0, 1.0]])           # row 0 inside, row 1 outside
    r = evaluate_coverage((lower, upper), truth)
    assert r.coverage == 0.5 and r.n_targets == 2


# ------------------------------------------------------------------ TATE band
@pytest.mark.parametrize("method", ["mbb", "multiplier", "pini_vantini", "pv", "lr", "ffscb"])
def test_tate_band_dispatch(harness, method):
    _, _, fit = harness
    kw = dict(backend="python") if method in ("lr", "ffscb") else {}
    band = tate_band(fit, method=method, d=1, alpha=0.10, **kw)
    assert isinstance(band, Band)
    assert band.kind == "confidence"
    assert band.level == pytest.approx(0.90)
    assert np.all(band.upper >= band.lower)
    assert band.mean_width() > 0


def test_tate_band_list_and_bad_method(harness):
    _, _, fit = harness
    bands = tate_band(fit, method="mbb", alpha=0.10)       # d=None -> list per hom dim
    assert len(bands) == 2 and all(isinstance(b, Band) for b in bands)
    with pytest.raises(ValueError):
        tate_band(fit, method="not-a-method")


# ----------------------------------------------------------------- CTATE band
def test_ctate_band_confidence_and_prediction_guard(harness):
    sim, sample, fit = harness
    learner = CTATEDRLearner(n_basis=5).fit(
        sample.observed, sim.tseq, cross_fit_result=fit
    )
    x = sim.EX
    band = ctate_band(learner, x, d=1, alpha=0.10)
    assert isinstance(band, Band) and band.kind == "confidence"
    assert np.all(band.upper >= band.lower)
    bands = ctate_band(learner, x, alpha=0.10)             # list per hom dim
    assert len(bands) == 2
    with pytest.raises(ValueError):
        ctate_band(learner, x, method="prediction")


# ------------------------------------------------------------------ ITTE band
def test_itte_band_and_degree_multiplicity(harness):
    sim, sample, _ = harness
    model = ITTEConformal.fit(
        sample.observed, sim.tseq, n_basis=5,
        weight_fn=make_weight_fn("overlap"), random_state=0,
        propensity_estimator=LogisticRegression(max_iter=500),
    )
    x = sim.EX
    band = itte_band(model, x, d=1, alpha=0.10)
    assert isinstance(band, Band) and band.kind == "prediction"
    assert np.all(band.upper >= band.lower)
    assert len(itte_band(model, x, alpha=0.10)) == 2      # list form

    # joint (sup-over-degree) bounds must be at least as wide as the per-dim band
    X = np.atleast_2d(x)
    joint = simultaneous_itte_bounds(model, X, 0.10, scheme="joint")
    for d in range(model.n_hom_dim):
        lo_j, hi_j, _ = joint[d]
        lo_m, hi_m, _ = model.band_bounds(X, 0.10, d=d)
        assert np.all((hi_j - lo_j) >= (hi_m - lo_m) - 1e-9)
    # bonferroni scheme also returns one entry per dim
    bonf = simultaneous_itte_bounds(model, X, 0.10, scheme="bonferroni")
    assert set(bonf) == {0, 1}
