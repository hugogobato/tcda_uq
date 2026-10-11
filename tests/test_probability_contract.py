"""Exact probability contracts and direct algebraic checks for AIPW."""
import numpy as np
import pytest

from tcda_uq.estimators.aipw import aipw_estimator, aipw_scores, ipw_estimator


def example():
    phi = np.array([[[1., 2.]], [[3., 4.]]])
    sample = (phi, np.array([0, 1]), np.zeros((2, 1)))
    means = [(np.zeros((2, 2)), np.ones((2, 2)))]
    return sample, means


@pytest.mark.parametrize("pi", [[0., .5], [.5, 1.], [np.nan, .5], [-.1, .5], [.5, 1.1]])
def test_invalid_probabilities_not_silently_changed(pi):
    sample, means = example()
    with pytest.raises(ValueError):
        aipw_scores(pi, means, sample)


def test_explicit_estimated_probability_regularization():
    sample, means = example()
    regularized = aipw_scores([0., 1.], means, sample, propensity_clip=.1)
    explicit = aipw_scores([.1, .9], means, sample)
    np.testing.assert_allclose(regularized, explicit)


def test_small_known_probabilities_preserve_actual_inverse_weights():
    sample, _ = example()
    pi = np.array([1.e-5, 1. - 1.e-5])
    _, weight = ipw_estimator(pi, sample, return_inv_weight=True)
    np.testing.assert_allclose(weight[:, 0], [-1 / (1 - pi[0]), 1 / pi[1]])
    # Swap treatment so actual large inverse probabilities are also checked.
    sample = (sample[0], 1 - sample[1], sample[2])
    _, weight = ipw_estimator(pi, sample, return_inv_weight=True)
    np.testing.assert_allclose(weight[:, 0], [1 / pi[0], -1 / (1 - pi[1])])


def test_aipw_score_matches_direct_armwise_formula():
    sample, means = example()
    pi = np.array([.2, .8])
    phi, treatment, _ = sample
    mu0, mu1 = means[0]
    expected = mu1 - mu0 + treatment[:, None] / pi[:, None] * (phi[:, 0] - mu1)
    expected -= (1 - treatment)[:, None] / (1 - pi[:, None]) * (phi[:, 0] - mu0)
    np.testing.assert_allclose(aipw_scores(pi, means, sample)[0], expected)
    np.testing.assert_allclose(aipw_estimator(pi, means, sample)[0], expected.mean(0))


def test_treatment_and_probability_shapes_are_checked():
    sample, means = example()
    with pytest.raises(ValueError):
        aipw_scores([.5], means, sample)
    invalid_sample = (sample[0], np.array([0, 2]), sample[2])
    with pytest.raises(ValueError):
        aipw_scores([.5, .5], means, invalid_sample)


@pytest.mark.parametrize("defect", ["different_grid", "floating_order"])
def test_ctate_precomputed_scores_preserve_grid_and_row_alignment(defect):
    from types import SimpleNamespace
    from tcda_uq.estimators import CTATEDRLearner
    sample, _ = example()
    grid = np.array([0., 1.])
    result = SimpleNamespace(
        order=np.array([0, 1]), tseq=grid.copy(),
        scores=[sample[0][:, 0].copy()],
    )
    if defect == "different_grid":
        result.tseq = np.array([0., 2.])
    else:
        result.order = result.order.astype(float)
    with pytest.raises(ValueError):
        CTATEDRLearner(n_basis=1).fit(sample, grid, cross_fit_result=result)
