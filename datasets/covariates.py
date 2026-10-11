"""Covariate + treatment DGP (ported from top-causal-effect utils/generate.py).

Two-component Gaussian covariates and a logistic-with-interactions treatment
mechanism, shared by the ORBIT and SARS-CoV-2 experiments.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit


TARGETS = ("full", "overlap", "matching", "sqrt")


def _sample_size(n, name="n"):
    if isinstance(n, bool) or int(n) != n or n < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(n)


def target_tilt(propensity, target="full"):
    """Known-propensity covariate tilt h defining dQ_X/dP_X = h/E[h].

    ``full`` uses h=1, ``overlap`` uses pi(1-pi), ``matching`` uses
    min(pi,1-pi), and ``sqrt`` uses sqrt(pi(1-pi)). Values are never clipped.
    Tilted targets exclude zero/one-propensity covariates. The full target
    requires positivity separately wherever arm likelihood ratios are used.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    pi = np.asarray(propensity, dtype=float)
    if not np.isfinite(pi).all() or np.any((pi < 0) | (pi > 1)):
        raise ValueError("propensity must contain finite probabilities in [0, 1]")
    if target == "full":
        return np.ones_like(pi)
    if target == "overlap":
        return pi * (1 - pi)
    if target == "matching":
        return np.minimum(pi, 1 - pi)
    return np.sqrt(pi * (1 - pi))


def _sample_tilted_covariates(draw_covariates, propensity, n, target, rng, max_draws):
    """Rejection-sample independent X before expensive potential outcomes."""
    n = _sample_size(n)
    target_tilt(np.empty(0), target)  # validate even for n=0 or target='full'
    if n == 0 or target == "full":
        return draw_covariates(n, rng)
    max_draws = _sample_size(max_draws, "max_draws")
    if max_draws < n:
        raise ValueError("max_draws must be at least n")
    bound = {"overlap": 0.25, "matching": 0.5, "sqrt": 0.5}[target]
    accepted, n_accepted, drawn = [], 0, 0
    while n_accepted < n and drawn < max_draws:
        batch = min(max(256, 2 * (n - n_accepted)), 10000, max_draws - drawn)
        X = draw_covariates(batch, rng)
        accept_prob = target_tilt(propensity(X), target) / bound
        selected = X[rng.random(batch) < accept_prob]
        accepted.append(selected)
        n_accepted += len(selected)
        drawn += batch
    if n_accepted < n:
        raise RuntimeError(
            f"target '{target}' accepted {n_accepted}/{n} units after {drawn} draws; "
            "increase max_draws or check that the target has positive mass"
        )
    return np.concatenate(accepted, axis=0)[:n]


def gen_covariate(mu1, mu2, Sigma, n, rng=None):
    """I.i.d. equal-mixture Gaussian units with random subgroup counts.

    Returns ``(cov1, cov2, covariate)``. The first two arrays contain subgroup
    members; ``covariate`` retains the original random unit order. Counts are
    binomial, rather than forcing half the sample into each subgroup.
    """
    rng = np.random.default_rng(rng)
    n = _sample_size(n)
    mu1, mu2 = np.asarray(mu1, dtype=float), np.asarray(mu2, dtype=float)
    if mu1.ndim != 1 or mu2.shape != mu1.shape:
        raise ValueError("mu1 and mu2 must be vectors of equal length")
    membership = rng.binomial(1, 0.5, size=n)
    noise = rng.multivariate_normal(np.zeros(len(mu1)), Sigma, size=n)
    X = noise + np.where(membership[:, None] == 0, mu1, mu2)
    return X[membership == 0], X[membership == 1], X


def gen_trt_prob(covariate, beta, rng=None):
    """Logistic treatment with two interaction terms. Returns ``(prob, A)``."""
    rng = np.random.default_rng(rng)
    covariate = np.atleast_2d(np.asarray(covariate, dtype=float))
    y = covariate @ beta
    if covariate.shape[1] >= 3:
        y = y + 0.5 * covariate[:, 1] * covariate[:, 2] - 0.7 * covariate[:, 0] * covariate[:, 2]
    prob = expit(y)
    A = rng.binomial(n=1, p=prob)
    return prob, A
