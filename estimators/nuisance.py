"""Nuisance estimation + cross-fitting for the functional DR estimator.

Fourier function-on-scalar regression via scikit-fda, propensity estimation, and
a reusable cross-fitting driver (:func:`cross_fit`) that also returns the
per-unit EIF process for UQ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import warnings

import numpy as np
import pandas as pd
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import FourierBasis
from skfda.ml.regression import LinearRegression
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold

from .aipw import aipw_scores


def fit_functional_regression(sample, tseq, n_basis):
    """Function-on-scalar regression of silhouettes on X, per arm and homology dim.

    Returns a length-``n_hom_dim`` list of ``(f_reg0, f_reg1)`` fitted skfda
    ``LinearRegression`` models (control, treated). Faithful port.
    """
    phi, A, X, tseq = validate_sample(sample, tseq)
    if tseq.size < 2:
        raise ValueError("functional regression requires a nondegenerate grid")
    if not isinstance(n_basis, (int, np.integer)) or n_basis < 1:
        raise ValueError("n_basis must be a positive integer")
    if set(np.unique(A)) != {0,1}:
        raise ValueError("functional outcome fitting requires both arms in independent training")
    n_hom_dim = phi.shape[-2]

    ind = np.asarray(A).astype(bool)
    X0 = X[~ind]
    X1 = X[ind]

    fb = FourierBasis(tseq[[0, -1]], n_basis)
    estimators = []
    for d in range(n_hom_dim):
        phi0 = phi[~ind, d, :]
        phi0_fb = FDataGrid(phi0, tseq).to_basis(fb)
        f_reg0 = LinearRegression().fit(pd.DataFrame(X0), phi0_fb)

        phi1 = phi[ind, d, :]
        phi1_fb = FDataGrid(phi1, tseq).to_basis(fb)
        f_reg1 = LinearRegression().fit(pd.DataFrame(X1), phi1_fb)

        estimators.append((f_reg0, f_reg1))
    return estimators


def predict_functional_regression(reg, X_eval, tseq):
    """Evaluate fitted function-on-scalar models at ``X_eval`` over ``tseq``.

    Returns ``mu_hats``: length-``n_hom_dim`` list of ``(mu0_hat, mu1_hat)``,
    each ``[n_eval, resolution]``.
    """
    Xdf = pd.DataFrame(X_eval)
    mu_hats = []
    for f_reg0, f_reg1 in reg:
        mu0 = np.asarray(f_reg0.predict(Xdf)(tseq))
        mu1 = np.asarray(f_reg1.predict(Xdf)(tseq))
        if mu0.ndim == 3:  # (n, res, 1) -> (n, res)
            mu0, mu1 = mu0[..., 0], mu1[..., 0]
        mu_hats.append((mu0, mu1))
    return mu_hats


def fit_propensity(X, A, estimator=None):
    """Fit a propensity model pi(x)=P(A=1|X=x). Defaults to a random forest."""
    if set(np.unique(A)) != {0, 1}:
        raise ValueError("propensity fitting requires both treatment arms in training data")
    est = RandomForestClassifier(random_state=0) if estimator is None else clone(estimator)
    return est.fit(X, A)


@dataclass
class NuisanceFit:
    """A single fold's fitted nuisances (outcome regressions + propensity)."""

    mu_reg: list           # list of (f_reg0, f_reg1) per homology dim
    prop_model: object     # fitted sklearn classifier
    tseq: np.ndarray
    propensity_feature_fn: Optional[Callable] = None

    def predict_mu(self, X_eval):
        return predict_functional_regression(self.mu_reg, X_eval, self.tseq)

    def predict_pi(self, X_eval):
        feat = self.propensity_feature_fn
        X = np.asarray(X_eval) if feat is None else feat(np.asarray(X_eval))
        return self.prop_model.predict_proba(X)[:, 1]


@dataclass
class CrossFitResult:
    """Cross-fitted DR estimates + the per-unit EIF process (for Phase 2 UQ)."""

    tseq: np.ndarray
    aipw: list             # mean AIPW per hom dim, each [resolution]
    ipw: list
    plugin: list
    scores: list           # per-unit DR score, each [n, resolution] (mean == aipw)
    pi_hat: np.ndarray     # cross-fitted propensity, [n]
    order: np.ndarray      # index into the ORIGINAL sample for each row of scores
    folds: list = field(default_factory=list)  # list of NuisanceFit, one per fold
    fold_scheme: str = "independent"
    propensity_clip: Optional[float] = None
    propensity_clip_count: int = 0

    def influence(self):
        """Centered EIF process (scores - mean), one [n, resolution] per hom dim."""
        return [s - s.mean(axis=0, keepdims=True) for s in self.scores]

    def tate(self):
        """The cross-fitted TATE point estimate == :attr:`aipw`."""
        return self.aipw


def cross_fit(
    sample,
    tseq,
    n_basis: int = 3,
    propensity_estimator=None,
    n_splits: int = 2,
    stratify: bool = False,
    random_state: Optional[int] = 0,
    propensity_feature_fn: Optional[Callable] = None,
    propensity_clip: Optional[float] = 0.01,
):
    """K-fold cross-fitted AIPW with per-unit DR scores.

    For each fold, nuisances are fit on the complement and used to score the
    held-out units; scores are concatenated across folds into a proper
    cross-fitted EIF process. The K=2 case mirrors the notebook's sample-split.

    Args:
        sample: ``(phi, A, X)``.
        n_basis: Fourier basis size for the outcome regression.
        propensity_estimator: sklearn classifier (default random forest).
        n_splits: number of cross-fitting folds.
        stratify: ``False`` (default) makes the partition independent of observations.
            ``True`` is experimental treatment stratification, outside the stated
            independent-fold sufficient regime.
        propensity_clip: explicit fitted-propensity clipping into
            ``[eps, 1-eps]`` (default .01). ``None`` rejects boundary predictions.
            Clipping controls denominators but does not establish nuisance rates.
        propensity_feature_fn: optional ``X -> features`` map for the propensity
            model (e.g. add interaction terms, as in the ORBIT experiments).
    """
    phi, A, X, tseq = validate_sample(sample, tseq)
    n = phi.shape[0]
    n_hom_dim = phi.shape[-2]
    if not isinstance(n_splits, (int, np.integer)) or not 2 <= n_splits <= n:
        raise ValueError("n_splits must be an integer between 2 and n")
    if propensity_clip is not None and (not np.isfinite(propensity_clip) or not 0 < propensity_clip < .5):
        raise ValueError("propensity_clip must be None or lie strictly between 0 and .5")
    if stratify:
        warnings.warn("Treatment-stratified folds are experimental and are outside the independent-fold theorem", UserWarning, stacklevel=2)

    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        if stratify
        else KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    )
    split_iter = splitter.split(X, A) if stratify else splitter.split(X)

    feat = propensity_feature_fn if propensity_feature_fn is not None else (lambda z: z)

    order_parts, pi_parts = [], []
    scores_parts = [[] for _ in range(n_hom_dim)]   # per-unit DR score (mean == aipw)
    plugin_parts = [[] for _ in range(n_hom_dim)]   # per-unit mu1 - mu0
    ipw_parts = [[] for _ in range(n_hom_dim)]      # per-unit IPW contribution
    folds = []
    clip_count = 0

    for train_idx, test_idx in split_iter:
        if set(np.unique(A[train_idx])) != {0, 1}:
            raise ValueError("An independent training fold lacks a treatment arm; increase the sample or prespecify an independent partition")
        train = (phi[train_idx], A[train_idx], X[train_idx])
        X_test, A_test, phi_test = X[test_idx], A[test_idx], phi[test_idx]

        reg = fit_functional_regression(train, tseq, n_basis=n_basis)
        prop = fit_propensity(feat(X[train_idx]), A[train_idx], propensity_estimator)
        folds.append(NuisanceFit(mu_reg=reg, prop_model=prop, tseq=tseq,
                                 propensity_feature_fn=propensity_feature_fn))

        mu_hats = predict_functional_regression(reg, X_test, tseq)
        pi_hat = prop.predict_proba(feat(X_test))[:, 1]
        if not np.isfinite(pi_hat).all() or np.any((pi_hat < 0) | (pi_hat > 1)):
            raise ValueError("fitted propensities must be finite probabilities")
        if propensity_clip is not None:
            clip_count += int(np.count_nonzero((pi_hat < propensity_clip) | (pi_hat > 1-propensity_clip)))
            pi_hat = np.clip(pi_hat, propensity_clip, 1.0 - propensity_clip)
        elif np.any((pi_hat <= 0) | (pi_hat >= 1)):
            raise ValueError("boundary fitted propensities require an explicit propensity_clip")
        test_sample = (phi_test, A_test, X_test)

        s = aipw_scores(pi_hat, mu_hats, test_sample)
        for d in range(n_hom_dim):
            mu0, mu1 = mu_hats[d]
            scores_parts[d].append(s[d])
            plugin_parts[d].append(mu1 - mu0)
        ipw_units = _ipw_units(pi_hat, test_sample)  # list per hom, [n_test, res]
        for d in range(n_hom_dim):
            ipw_parts[d].append(ipw_units[d])
        order_parts.append(test_idx)
        pi_parts.append(pi_hat)

    order = np.concatenate(order_parts)
    pi_hat_full = np.concatenate(pi_parts)
    scores = [np.concatenate(scores_parts[d], axis=0) for d in range(n_hom_dim)]
    plugin_u = [np.concatenate(plugin_parts[d], axis=0) for d in range(n_hom_dim)]
    ipw_u = [np.concatenate(ipw_parts[d], axis=0) for d in range(n_hom_dim)]

    return CrossFitResult(
        tseq=np.asarray(tseq),
        aipw=[s.mean(axis=0) for s in scores],
        ipw=[u.mean(axis=0) for u in ipw_u],
        plugin=[u.mean(axis=0) for u in plugin_u],
        scores=scores,
        pi_hat=pi_hat_full,
        order=order,
        folds=folds,
        fold_scheme="stratified_experimental" if stratify else "independent",
        propensity_clip=propensity_clip,
        propensity_clip_count=clip_count,
    )


def _ipw_units(pi_hat, sample):
    """Per-unit IPW contributions, list of ``[n, resolution]`` (mean == IPW)."""
    from .aipw import _clip_pi, _inv_weight

    phi, A, _ = sample
    n_hom_dim = phi.shape[-2]
    inv_weight = _inv_weight(_clip_pi(pi_hat), A)
    return [inv_weight * phi[:, d, :] for d in range(n_hom_dim)]


def validate_sample(sample, tseq):
    """Validate finite observed curves, binary treatment, covariates, and grid."""
    phi, A, X = sample
    phi, A, X = np.asarray(phi, float), np.asarray(A), np.asarray(X, float)
    grid = np.asarray(tseq, float)
    if phi.ndim != 3 or phi.shape[1] < 1 or phi.shape[2] < 1 or not np.isfinite(phi).all():
        raise ValueError("phi must be finite with shape [n, n_hom_dim, resolution]")
    if A.shape != (phi.shape[0],) or not np.isin(A, (0, 1)).all():
        raise ValueError("A must be a binary [n] array")
    if X.ndim != 2 or X.shape[0] != phi.shape[0] or not np.isfinite(X).all():
        raise ValueError("X must be a finite [n, p] array")
    if grid.shape != (phi.shape[2],) or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("tseq must be finite, strictly increasing, and match resolution")
    return phi, A, X, grid
