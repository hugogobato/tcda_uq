import numpy as np
import pytest

from tcda_uq.datasets import TriOracleSimulation, TopologicalCausalSimulation, gen_orbits
from tcda_uq.estimators import aipw_estimator, aipw_scores, cross_fit
from tcda_uq.metrics import (
    Band,
    interval_wise_error,
    mean_width,
    pointwise_coverage,
    simultaneous_coverage,
)
from tcda_uq.silhouette import silhouette_from_pointcloud
from tcda_uq.uq.conformal import ITTEConformal, make_weight_fn
from tcda_uq.utils import numerical_integration


def test_core_imports():
    import tcda_uq
    import tcda_uq.uq.asymptotic  # noqa: F401
    import tcda_uq.uq.conformal  # noqa: F401

    assert tcda_uq.__version__


def test_simulation_returns_three_oracles():
    sim = TriOracleSimulation(n_hom_dim=2, resolution=50, seed=0)
    sample = sim.sample(100, rng=1)

    phi, A, X = sample.observed
    assert phi.shape == (100, 2, 50)
    assert A.shape == (100,)
    assert X.shape[0] == 100
    assert sample.oracle_itte.shape == (100, 2, 50)
    assert sample.oracle_ctate.shape == (100, 2, 50)
    assert sample.oracle_tate.shape == (2, 50)


def test_cross_fit_scores_mean_equals_aipw():
    sim = TriOracleSimulation(seed=0, resolution=35)
    sample = sim.sample(120, rng=2)
    result = cross_fit(sample.observed, sim.tseq, n_basis=5, n_splits=2, random_state=0)
    assert np.array_equal(np.sort(result.order), np.arange(120))
    for d in range(sim.n_hom_dim):
        assert np.allclose(result.scores[d].mean(axis=0), result.aipw[d])


def test_aipw_recovers_true_tate_reasonably():
    sim = TriOracleSimulation(n_basis=5, noise_scale=0.3, resolution=35, seed=0)
    sample = sim.sample(350, rng=3)
    result = cross_fit(sample.observed, sim.tseq, n_basis=5, n_splits=2, random_state=0)
    for d in range(sim.n_hom_dim):
        err = numerical_integration(np.abs(result.aipw[d] - sample.oracle_tate[d]), sim.tseq)
        mag = numerical_integration(np.abs(sample.oracle_tate[d]), sim.tseq)
        assert err / mag < 0.25


def test_aipw_mean_matches_manual():
    sim = TriOracleSimulation(seed=0, resolution=30)
    sample = sim.sample(100, rng=4)
    phi, A, X = sample.observed

    from tcda_uq.estimators.nuisance import (
        fit_functional_regression,
        fit_propensity,
        predict_functional_regression,
    )

    reg = fit_functional_regression(sample.observed, sim.tseq, n_basis=4)
    mu_hats = predict_functional_regression(reg, X, sim.tseq)
    pi = fit_propensity(X, A).predict_proba(X)[:, 1]
    manual = aipw_estimator(pi, mu_hats, sample.observed)
    scores = aipw_scores(pi, mu_hats, sample.observed)
    for d in range(sim.n_hom_dim):
        assert np.allclose(scores[d].mean(axis=0), manual[d])


def test_metrics():
    tseq = np.linspace(0, 1, 30)
    truth = np.sin(2 * np.pi * tseq)
    band = Band(tseq, truth - 0.5, truth + 0.5, center=truth)
    assert mean_width(band.lower, band.upper) == pytest.approx(1.0)
    assert interval_wise_error(band.lower, band.upper, truth) == 0.0
    assert simultaneous_coverage(band.lower, band.upper, truth) == 1.0
    bad = truth.copy()
    bad[10] += 10
    assert simultaneous_coverage(band.lower, band.upper, bad) == 0.0
    assert pointwise_coverage(band.lower, band.upper, np.vstack([truth, bad]))[10] == pytest.approx(0.5)


def test_silhouette_from_pointcloud():
    rng = np.random.default_rng(0)
    pts = rng.random((40, 2))
    sil = silhouette_from_pointcloud(pts, interval=(0.0, 0.2), r=3, resolution=40)
    assert sil.shape == (2, 40)
    assert np.all(np.isfinite(sil))
    assert np.all(sil >= 0)


def test_orbit_generator_shapes():
    X, y = gen_orbits(rhos=(3.5, 4.0), num_pts=30, num_orbits_each=4, rng=0)
    assert X.shape == (8, 30, 2)
    assert set(np.unique(y)) == {0, 1}
    assert X.min() >= 0
    assert X.max() <= 1


def test_itte_conformal_produces_ordered_bounds():
    sim = TriOracleSimulation(n_hom_dim=1, resolution=25, n_basis=5, seed=0)
    train = sim.sample(180, rng=5)
    test = sim.sample(8, rng=6)
    model = ITTEConformal.fit(
        train.observed,
        sim.tseq,
        n_basis=5,
        weight_fn=make_weight_fn("overlap"),
        random_state=0,
    )
    lo, hi, center = model.band_bounds(test.X, alpha=0.1, d=0)
    assert lo.shape == hi.shape == center.shape == (8, 25)
    assert np.all(hi >= lo)


def test_level_b_mc_oracles_have_expected_shapes():
    sim = TopologicalCausalSimulation(
        n_cov=3,
        homology_dims=(0, 1),
        resolution=12,
        pts_per_loop=8,
        background=2,
        max_loops=2,
        seed=0,
    )
    X = np.array([[0.0, 0.1, -0.2], [1.0, -0.5, 0.3]])
    assert sim.true_ctate_mc(X, n_mc=1, rng=1).shape == (2, 2, 12)
    assert sim.true_tate_mc(n_x=3, n_mc=1, rng=2).shape == (2, 12)
