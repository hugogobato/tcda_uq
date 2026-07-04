"""Deterministic unit / regression tests for the UQ building blocks.

Fast and seed-free where possible (pure numpy); no repeated cross-fitting. The
statistical coverage-property tests (which drive the simulation harness over many
replicates) live in ``test_coverage_properties.py`` and are marked ``slow``.
"""

import importlib

import numpy as np
import pytest

from tcda_uq.estimators import EPS_PI, aipw_estimator, aipw_scores, plugin_estimator
from tcda_uq.estimators.aipw import _clip_pi
from tcda_uq.uq.asymptotic.covariance import (
    eif_correlation,
    eif_covariance,
    eif_pointwise_sd,
    eif_pointwise_variance,
)
from tcda_uq.uq.conformal.functional_cp import (
    grid_discretization_slack,
    modulation,
    split_conformal_radius,
    sup_norm_score,
)
from tcda_uq.uq.conformal.weighted_cp import (
    propensity_weights,
    weighted_conformal_radius,
)
# The package __init__ re-exports functions named `stabilized_weights` /
# `silhouette_bridge`-family, which shadow the submodule attributes; fetch the
# real modules from sys.modules via importlib.
SB = importlib.import_module("tcda_uq.uq.conformal.silhouette_bridge")
SW = importlib.import_module("tcda_uq.uq.conformal.stabilized_weights")


# --------------------------------------------------------------------- covariance
def test_eif_covariance_symmetric_psd_diag_matches_variance():
    rng = np.random.default_rng(0)
    phi = rng.standard_normal((300, 12))
    cov = eif_covariance(phi)
    assert cov.shape == (12, 12)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-8           # PSD
    var = eif_pointwise_variance(phi)
    assert np.allclose(np.diag(cov), var)                  # diagonal == pointwise var
    assert np.allclose(eif_pointwise_sd(phi), np.sqrt(np.maximum(var, 1e-12)))
    corr = eif_correlation(phi)
    assert np.allclose(np.diag(corr), 1.0, atol=1e-6)      # unit diagonal


# ----------------------------------------------------------------- functional CP
def test_split_conformal_radius_is_finite_sample_quantile():
    scores = np.arange(1.0, 11.0)                          # n = 10
    # k = ceil((n+1)(1-alpha)); returns the k-th smallest, else +inf if k > n
    assert split_conformal_radius(scores, 0.1) == 10.0     # k = ceil(9.9)  = 10
    assert split_conformal_radius(scores, 0.2) == 9.0      # k = ceil(8.8)  = 9
    assert split_conformal_radius(scores, 0.05) == np.inf  # k = ceil(10.45)= 11 > n


def test_modulation_kinds_and_sup_norm_score():
    R = np.array([[1.0, -1.0, 0.5], [-1.0, 1.0, -0.5], [0.5, -0.5, 0.25]])
    assert np.allclose(modulation(R, "constant"), 1.0)
    assert np.allclose(modulation(R, "pointwise-sd"), np.maximum(R.std(axis=0), 1e-8))
    # lipschitz floors s(t) away from zero even where the residual spread vanishes
    R_flat = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert np.all(modulation(R_flat, "lipschitz") > 0)
    with pytest.raises(ValueError):
        modulation(R, "nope")
    # score = max_t |r|/s = max(2/1, 4/2, 1/1) = 2
    assert sup_norm_score(np.array([[2.0, -4.0, 1.0]]), np.array([1.0, 2.0, 1.0])) == 2.0


def test_grid_discretization_slack():
    tseq = np.linspace(0.0, 1.0, 11)                       # spacing 0.1
    s = np.array([0.5, 1.0, 2.0])                          # min 0.5
    assert grid_discretization_slack(tseq, s) == pytest.approx(1.0 * 0.1 / 2 / 0.5)
    assert grid_discretization_slack(0.2, s, lipschitz_const=2.0) == pytest.approx(
        2.0 * 0.2 / 2 / 0.5
    )


# ------------------------------------------------------------------- weighted CP
def test_propensity_weights_and_arm_validation():
    pi = np.array([0.2, 0.5, 0.8])
    assert np.allclose(propensity_weights(pi, 1), 1.0 / _clip_pi(pi))
    assert np.allclose(propensity_weights(pi, 0), 1.0 / (1.0 - _clip_pi(pi)))
    with pytest.raises(ValueError):
        propensity_weights(pi, 2)


def test_weighted_radius_reduces_to_split_radius_under_uniform_weights():
    rng = np.random.default_rng(1)
    scores = rng.random(50)
    for alpha in (0.05, 0.1, 0.2):
        w = weighted_conformal_radius(scores, alpha, np.ones(50), 1.0)
        assert w == split_conformal_radius(scores, alpha)


def test_weighted_radius_inf_atom_fires_for_dominant_test_weight():
    scores = np.arange(1.0, 11.0)
    assert np.isinf(weighted_conformal_radius(scores, 0.1, np.ones(10), 1e6))


# ------------------------------------------------------- positivity-stabilization
def test_overlap_and_matching_weights_are_bounded():
    rng = np.random.default_rng(2)
    pi = rng.uniform(0.001, 0.999, size=500)
    for arm in (0, 1):
        for w in (SW.overlap_weights(pi, arm), SW.matching_weights(pi, arm)):
            assert np.all(w > 0) and np.all(w <= 1.0 + 1e-12)
    # overlap: treated -> 1-pi, control -> pi (on the clipped propensity)
    assert np.allclose(SW.overlap_weights(pi, 1), 1.0 - _clip_pi(pi))
    assert np.allclose(SW.overlap_weights(pi, 0), _clip_pi(pi))


def test_tilted_endpoints_recover_naive_and_overlap():
    pi = np.array([0.1, 0.3, 0.6, 0.9])
    assert np.allclose(SW.tilted_weights(pi, 1, gamma=0.0), SW.naive_weights(pi, 1))
    assert np.allclose(SW.tilted_weights(pi, 1, gamma=1.0), SW.overlap_weights(pi, 1))
    with pytest.raises(ValueError):
        SW.tilted_weights(pi, 1, gamma=1.5)


def test_weight_upper_bounds():
    assert SW.weight_upper_bound("overlap") == 1.0
    assert SW.weight_upper_bound("matching") == 1.0
    assert SW.weight_upper_bound("tilted", gamma=1.0) == 1.0
    assert SW.weight_upper_bound("tilted", gamma=0.5) == pytest.approx(EPS_PI ** (-0.5))
    assert SW.weight_upper_bound("clip", eps=0.1) == pytest.approx(10.0)
    assert SW.weight_upper_bound("shrink", cap=7.0) == 7.0
    # naive is unbounded except for the EPS_PI clip -> the largest bound of all
    assert SW.weight_upper_bound("naive") == pytest.approx(1.0 / EPS_PI)


def test_shrink_saturates_below_cap():
    w = SW.shrink_weights(np.array([1e-6]), 1, cap=10.0)   # tiny pi -> huge naive weight
    assert 9.0 < w[0] < 10.0


def test_stabilizer_registry_dispatch_and_binding():
    assert set(SW.STABILIZERS) == set(SW.EXACT_TARGET_METHODS) | set(SW.APPROX_TARGET_METHODS)
    for m in SW.STABILIZERS:
        assert isinstance(SW.target_description(m), str) and SW.target_description(m)
    with pytest.raises(ValueError):
        SW.stabilized_weights(np.array([0.5]), 1, method="nope")
    fn = SW.make_weight_fn("tilted", gamma=0.4)
    assert fn.method == "tilted" and fn.params == {"gamma": 0.4}
    assert np.allclose(fn(np.array([0.3]), 1), SW.tilted_weights(np.array([0.3]), 1, gamma=0.4))


# ------------------------------------------------------------- silhouette bridge
def test_bridge_constant_formula():
    assert SB.bridge_constant(3.0, 1.0, 5.0) == pytest.approx(1 + 2 * 3)    # c drops at r=1
    assert SB.bridge_constant(2.0, 2.0, 3.0) == pytest.approx(1 + 2 * 2 * 2 * 3)  # = 25
    assert SB.bridge_constant(0.0, 2.0, 9.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        SB.bridge_constant(1.0, 0.0, 1.0)


def test_estimate_bridge_params():
    diags = [np.array([[0.0, 1.0], [0.0, 3.0]])]           # lifetimes 1, 3
    L, c = SB.estimate_bridge_params(diags, r=2.0)
    assert c == pytest.approx(3.0)
    assert L == pytest.approx(1.0)                          # max(lt^{-1}) = max(1, 1/3)
    assert SB.estimate_bridge_params([], r=2.0) == (0.0, 0.0)


def test_score_perturbation_and_certificates():
    scores = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
    K, eps = 2.0, 0.1
    delta = SB.score_perturbation_bound(K, eps, np.array([0.5, 1.0]))       # 2*0.1/0.5
    assert delta == pytest.approx(0.4)
    assert np.allclose(
        SB.certified_score_inflation(scores, K, eps, np.array([0.5, 1.0])), scores + delta
    )
    s = np.array([0.25, 1.0])
    assert SB.certified_width_inflation(K, eps, s) == pytest.approx(
        2 * SB.score_perturbation_bound(K, eps, s) * np.mean(s)
    )
    # radius 1.0, delta 0.8 -> only score 1.5 falls in the miss window (1.0, 1.8]
    cert = SB.coverage_certificate(scores, 1.0, K, eps, s)
    assert cert["delta"] == pytest.approx(0.8)
    assert cert["coverage_loss_bound"] == pytest.approx(0.2)
    assert cert["certified_coverage"] == pytest.approx(0.6 - 0.2)
    # an unbounded band cannot miss
    cert_inf = SB.coverage_certificate(scores, np.inf, K, eps, s)
    assert cert_inf["coverage_loss_bound"] == 0.0
    assert cert_inf["certified_coverage"] == 1.0


# --------------------------------------------------------------------- estimators
def test_aipw_scores_mean_equals_estimator_and_plugin_identity():
    rng = np.random.default_rng(3)
    n, res = 80, 8
    phi = rng.standard_normal((n, 1, res))
    A = rng.integers(0, 2, n)
    X = rng.standard_normal((n, 3))
    pi = rng.uniform(0.2, 0.8, n)
    mu_hats = [(rng.standard_normal((n, res)), rng.standard_normal((n, res)))]
    sample = (phi, A, X)
    scores = aipw_scores(pi, mu_hats, sample)
    est = aipw_estimator(pi, mu_hats, sample)
    assert np.allclose(scores[0].mean(axis=0), est[0])
    plug = plugin_estimator(mu_hats)
    assert np.allclose(plug[0], (mu_hats[0][1] - mu_hats[0][0]).mean(axis=0))
