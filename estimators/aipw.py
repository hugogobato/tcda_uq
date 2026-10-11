"""PI / IPW / AIPW estimators for the functional (silhouette) treatment effect.

The *mean* estimators (:func:`ipw_estimator`, :func:`plugin_estimator`,
:func:`aipw_estimator`) target the marginal functional treatment effect.
:func:`aipw_scores` / :func:`aipw_influence` return the *per-unit* doubly-robust
score process phi_hat_d(t, Z; eta_hat) -- the object used by confidence bands.

Conventions
-----------
``sample`` is a triplet ``(phi, A, X)`` with
  * ``phi``: silhouettes, shape ``[n, n_hom_dim, resolution]``
  * ``A``:   treatment, shape ``[n]``
  * ``X``:   covariates, shape ``[n, d]``
``mu_hats`` is a length-``n_hom_dim`` list of ``(mu0_hat, mu1_hat)`` predictions,
each of shape ``[n, resolution]`` (evaluated on the *estimation* sample).
``pi_hat`` is the estimated propensity, shape ``[n]``.
"""

from __future__ import annotations

import numpy as np

# Conventional explicit regularization level, never applied implicitly here.
EPS_PI = 1e-2


def _clip_pi(pi_hat, *, clip=None):
    """Validate probabilities; apply clipping only when explicitly requested.

    Clipping an estimated nuisance can regularize an estimator. Clipping true
    likelihood ratios changes them and does not preserve exact conformal
    transport guarantees. The historical private name is retained for callers.
    """
    pi = np.asarray(pi_hat, dtype=float)
    if pi.ndim != 1 or not np.isfinite(pi).all():
        raise ValueError("propensities must be a finite one-dimensional array")
    if np.any((pi < 0) | (pi > 1)):
        raise ValueError("propensities must lie in [0, 1]")
    if clip is not None:
        if not np.isfinite(clip) or not 0 < clip < .5:
            raise ValueError("propensity_clip must lie strictly between 0 and .5")
        return np.clip(pi, clip, 1 - clip)
    if np.any((pi <= 0) | (pi >= 1)):
        raise ValueError("propensities must lie strictly in (0, 1); request propensity_clip explicitly for estimated nuisances")
    return pi.copy()


def _validate_sample(sample):
    phi, A, _ = sample
    phi, A = np.asarray(phi, dtype=float), np.asarray(A, dtype=float)
    if phi.ndim != 3 or min(phi.shape) < 1 or not np.isfinite(phi).all():
        raise ValueError("outcomes must be finite with shape [n, degree, grid]")
    if A.shape != (len(phi),) or not np.isfinite(A).all() or not np.isin(A, [0, 1]).all():
        raise ValueError("treatment must be a binary vector with one entry per outcome")
    return phi, A


def _inv_weight(pi_hat, A):
    """IPW sign-weight  A/pi - (1-A)/(1-pi),  shape ``[n, 1]``."""
    A = np.asarray(A, dtype=float)
    return (A / pi_hat - (1 - A) / (1 - pi_hat))[:, np.newaxis]


def ipw_estimator(pi_hat, sample, return_inv_weight=False, *, propensity_clip=None):
    """Mean IPW estimate per homology dim. Returns list of ``[resolution]`` arrays."""
    phi, A = _validate_sample(sample)
    n_hom_dim = phi.shape[-2]
    pi_hat = _clip_pi(pi_hat, clip=propensity_clip)
    if pi_hat.shape != A.shape:
        raise ValueError("one propensity is required per outcome")
    inv_weight = _inv_weight(pi_hat, A)

    ipw = [np.mean(inv_weight * phi[:, d, :], axis=0) for d in range(n_hom_dim)]
    if return_inv_weight:
        return ipw, inv_weight
    return ipw


def plugin_estimator(mu_hats, return_mu=False):
    """Mean plug-in estimate per homology dim. Returns list of ``[resolution]`` arrays."""
    plugin = [np.mean(mu1 - mu0, axis=0) for mu0, mu1 in mu_hats]
    if return_mu:
        mu0_list, mu1_list = zip(*mu_hats)
        return plugin, list(mu0_list), list(mu1_list)
    return plugin


def aipw_estimator(pi_hat, mu_hats, sample, *, propensity_clip=None):
    """Mean AIPW (doubly-robust) estimate per homology dim. List of ``[resolution]``."""
    return [s.mean(axis=0) for s in aipw_scores(pi_hat, mu_hats, sample,
                                               propensity_clip=propensity_clip)]


def aipw_scores(pi_hat, mu_hats, sample, *, propensity_clip=None):
    """Per-unit doubly-robust score process, one ``[n, resolution]`` array per hom dim.

    The mean over units equals :func:`aipw_estimator`. The *centered* version
    (see :func:`aipw_influence`) is the estimated efficient influence function
    process phi_hat_d(t, Z_i; eta_hat) that the multiplier bootstrap (Phase 2.1)
    reweights.
    """
    phi, A = _validate_sample(sample)
    n_hom_dim = phi.shape[-2]
    pi_hat = _clip_pi(pi_hat, clip=propensity_clip)
    if pi_hat.shape != A.shape:
        raise ValueError("one propensity is required per outcome")
    if len(mu_hats) != n_hom_dim:
        raise ValueError("one pair of nuisance means is required per outcome degree")
    inv_weight = _inv_weight(pi_hat, A)  # [n, 1]
    A = np.asarray(A, dtype=float)[:, np.newaxis]

    scores = []
    for d in range(n_hom_dim):
        mu0, mu1 = (np.asarray(mu, dtype=float) for mu in mu_hats[d])
        expected = (len(phi), phi.shape[-1])
        if mu0.shape != expected or mu1.shape != expected or not np.isfinite(mu0).all() or not np.isfinite(mu1).all():
            raise ValueError("nuisance predictions must be finite with shape [n, grid]")
        plugin = mu1 - mu0                                   # [n, res]
        ipw = inv_weight * phi[:, d, :]                      # [n, res]
        model = inv_weight * (A * mu1 + (1 - A) * mu0)       # [n, res]
        scores.append(plugin + ipw - model)                 # [n, res]
    return scores


def aipw_influence(pi_hat, mu_hats, sample, *, propensity_clip=None):
    """Centered EIF process ``scores - mean``, one ``[n, resolution]`` array per hom dim."""
    scores = aipw_scores(pi_hat, mu_hats, sample, propensity_clip=propensity_clip)
    return [s - s.mean(axis=0, keepdims=True) for s in scores]
