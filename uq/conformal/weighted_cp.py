"""Weighted split-conformal prediction with a retained +infinity test atom.

Exact likelihood ratios give a marginal coverage lower bound for their target.
Learned training-measurable weights give that bound for the estimated-weight law;
coverage for the intended law subtracts its total-variation discrepancy. Neither
outcome-regression accuracy nor independent fitting makes estimated ratios exact.
No doubly robust guarantee or universally finite radius is asserted here.
"""
from __future__ import annotations

import numpy as np
from bisect import bisect_left

from ...metrics import Band
from ...estimators.nuisance import validate_sample
from ..asymptotic._validation import validate_alpha
from .functional_cp import modulation, residual_curves, sup_norm_score


def _probabilities(pi, *, strict=False):
    pi = np.asarray(pi, dtype=float)
    if not np.isfinite(pi).all() or np.any((pi < 0) | (pi > 1)):
        raise ValueError("propensities must be finite probabilities in [0, 1]")
    if strict and np.any((pi <= 0) | (pi >= 1)):
        raise ValueError("full-population inverse weights require strict positivity; use explicit clip/shrink for approximate weights")
    return pi


def propensity_weights(pi_hat, arm: int):
    """Full-population ratio proportional to 1/e_a, without hidden clipping."""
    if arm not in (0, 1):
        raise ValueError("arm must be 0 or 1")
    pi = _probabilities(pi_hat, strict=True)
    return 1.0 / (pi if arm == 1 else 1.0 - pi)


def weighted_conformal_radius(scores, alpha: float, w_calib, w_new):
    """Left quantile of calibration scores plus the target's +infinity atom.

    Nonnegative finite weights are allowed for tilted targets. Empty calibration
    or zero total mass returns infinity. Scaling all weights by a common positive
    constant has no effect. Even bounded weights can yield an infinite radius.
    """
    validate_alpha(alpha)
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(w_calib, dtype=float)
    target_weights = np.asarray(w_new, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("scores must be a finite nonnegative [n] array")
    if weights.shape != scores.shape or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("w_calib must be finite nonnegative weights matching scores")
    if target_weights.ndim > 1 or not np.isfinite(target_weights).all() or np.any(target_weights < 0):
        raise ValueError("w_new must be a finite nonnegative scalar or vector")
    scalar = target_weights.ndim == 0
    wv = np.atleast_1d(target_weights)
    radius = np.full(wv.shape, np.inf)
    if scores.size:
        order = np.argsort(scores, kind="mergesort")
        # Every supplied finite binary float is a dyadic rational. Integer
        # cross-products implement the inclusive CDF comparison exactly, without
        # tolerances, cumulative roundoff, overflow, or omitting the test atom.
        ratios = [float(value).as_integer_ratio() for value in weights[order]]
        denominator = max(q for _, q in ratios)
        cumulative, total = [], 0
        for numerator, q in ratios:
            total += numerator * (denominator // q)
            cumulative.append(total)
        alpha_numerator, alpha_denominator = float(alpha).as_integer_ratio()
        retained = alpha_denominator-alpha_numerator
        for j, wn in enumerate(wv):
            new_numerator, new_denominator = float(wn).as_integer_ratio()
            if total == 0 and new_numerator == 0:
                continue
            numerator = retained * (total*new_denominator + new_numerator*denominator)
            divisor = alpha_denominator*new_denominator
            threshold = (numerator+divisor-1)//divisor
            idx = bisect_left(cumulative, threshold)
            if idx < scores.size:
                radius[j] = scores[order[idx]]
    return float(radius[0]) if scalar else radius


def _covariates(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("covariates must be a finite [m, p] array")
    return X


class ConformalArm:
    """One arm's weighted prediction band with training-frozen modulation.

    `modulation_kind='constant'` fixes s=1. Other kinds require
    `modulation_residuals` from independent training, or an already frozen
    positive `modulation_scale`. Independence of supplied objects is a caller
    obligation. `propensity_fn` can supply true probabilities, evaluated on raw X.
    `weights_known=True` records that the supplied weights are actual target
    ratios; it cannot verify that scientific assertion. Estimated ratios require
    a TV qualification for the intended target. `weighted=False` targets the
    observed arm covariate law unless that law already equals the desired target.
    """

    def __init__(self, nuisance, calib_sample, d: int, a: int, *,
                 modulation_kind: str = "constant", floor: float = 0.0,
                 weighted: bool = True, propensity_feature_fn=None,
                 modulation_residuals=None, modulation_scale=None, weight_fn=None,
                 target: str = "full", propensity_fn=None, weights_known: bool = False):
        if modulation_kind not in ("constant", "pointwise-sd", "lipschitz"):
            raise ValueError("modulation_kind must be constant, pointwise-sd, or lipschitz")
        if not np.isfinite(floor) or floor < 0:
            raise ValueError("floor must be finite and nonnegative")
        phi, A, X, self.tseq = validate_sample(calib_sample, nuisance.tseq)
        if a not in (0, 1) or not isinstance(d, (int, np.integer)) or not 0 <= d < phi.shape[1]:
            raise ValueError("invalid arm or homology dimension")
        if target not in ("full", "overlap", "matching"):
            raise ValueError("target must be 'full', 'overlap', or 'matching'")
        self.nuisance, self.d, self.a, self.weighted = nuisance, d, a, weighted
        self._feat = propensity_feature_fn
        self._propensity_fn = propensity_fn
        if weight_fn is None:
            from .stabilized_weights import make_weight_fn
            weight_fn = make_weight_fn("naive" if target == "full" else target)
        elif target != "full" and getattr(weight_fn, "method", target) != target:
            raise ValueError("target and weight_fn describe different populations")
        self._weight_fn = weight_fn
        method = getattr(weight_fn, "method", "custom")
        self.target_ = target if method in ("naive", "custom") else ("full" if method in ("clip", "shrink") else method)
        self.weights_known_ = bool(weights_known and method not in ("clip", "shrink") and weighted)
        self.coverage_scope_ = ("exact target ratios, marginal lower bound" if self.weights_known_ else
                                "estimated-weight law; intended target requires armwise TV qualification") if weighted else "observed-arm law"
        mask = A == a
        self.calib_index_ = np.flatnonzero(mask)
        Xa, phi_a = X[mask], phi[mask, d, :]
        mu_a = self.center(Xa) if Xa.shape[0] else np.empty((0, phi.shape[-1]))
        R = residual_curves(phi_a, mu_a)
        if modulation_scale is not None:
            self.s = np.array(modulation_scale, dtype=float, copy=True)
        elif modulation_kind == "constant":
            self.s = np.ones(phi.shape[-1])
        elif modulation_residuals is None:
            raise ValueError("nonconstant modulation requires independent training modulation_residuals or a frozen modulation_scale")
        else:
            self.s = modulation(modulation_residuals, modulation_kind, floor=floor)
        if self.s.shape != (phi.shape[-1],) or not np.isfinite(self.s).all() or np.any(self.s <= 0):
            raise ValueError("modulation_scale must be finite, strictly positive, and match resolution")
        self.s.setflags(write=False)
        self.scores = np.atleast_1d(sup_norm_score(R, self.s))
        self.w_calib = self._w_new(Xa) if Xa.shape[0] else np.empty(0)

    def _pi(self, X):
        X = _covariates(X)
        if self._propensity_fn is not None:
            pi = self._propensity_fn(X)
        elif self._feat is not None:
            pi = self.nuisance.prop_model.predict_proba(self._feat(X))[:, 1]
        elif hasattr(self.nuisance, "predict_pi"):
            pi = self.nuisance.predict_pi(X)
        else:
            pi = self.nuisance.prop_model.predict_proba(X)[:,1]
        pi = np.asarray(pi, float)
        if pi.shape != (X.shape[0],):
            raise ValueError("propensity predictions must have shape [m]")
        return _probabilities(pi)

    def _w_new(self, X_new):
        X_new = _covariates(X_new)
        if not self.weighted:
            return np.ones(X_new.shape[0])
        weights = np.asarray(self._weight_fn(self._pi(X_new), self.a), float)
        if weights.shape != (X_new.shape[0],) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("weight_fn must return finite nonnegative [m] weights")
        return weights

    def center(self, X_new):
        X_new = _covariates(X_new)
        c = np.asarray(self.nuisance.predict_mu(X_new)[self.d][self.a], float)
        if c.shape != (X_new.shape[0], self.tseq.size) or not np.isfinite(c).all():
            raise ValueError("outcome predictions must be finite [m, resolution] curves")
        return c

    def radius(self, X_new, alpha: float):
        X_new = _covariates(X_new)
        return np.atleast_1d(weighted_conformal_radius(self.scores, alpha, self.w_calib, self._w_new(X_new)))

    def band_bounds(self, X_new, alpha: float):
        c = self.center(X_new)
        half = self.radius(X_new, alpha)[:, None] * self.s[None, :]
        return c - half, c + half, c

    def band(self, x_new, alpha: float) -> Band:
        lower, upper, center = self.band_bounds(x_new, alpha)
        if lower.shape[0] != 1:
            raise ValueError("band expects one covariate row; use band_bounds for batches")
        return Band(tseq=self.tseq, lower=lower[0], upper=upper[0], center=center[0],
                    level=1.0-alpha, kind="prediction")
