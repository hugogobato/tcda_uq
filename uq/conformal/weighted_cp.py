"""Layer 2 -- weighted causal split conformal prediction (Phase 4.2).

Intervening ``do(A = a)`` shifts the covariate distribution through the
propensity, breaking the exchangeability plain split-CP needs. Because treatment
is **binary**, none of the continuous-treatment machinery (Dirac-delta /
Gaussian-limit / M-bound of Schroeder et al. 2025) is required: we are in the
**weighted split-conformal** regime of Tibshirani et al. (2019) / Lei-Candes
(2021). Reweight the (already scalarized -- Layer 1) calibration scores by the
propensity likelihood ratio between the *interventional* target population
``X ~ p(x)`` and the *observed arm-``a``* calibration population
``X ~ p(x | A = a)``:

    a = 1 (treated arm):   w(x)  proportional to  1 / pi(x)
    a = 0 (control arm):   w(x)  proportional to  1 / (1 - pi(x)).

The conformal quantile is taken w.r.t. the tilted empirical law
``sum_i p_i(x_new) delta_{S_i} + p_{n+1}(x_new) delta_{+inf}`` with
``p_i proportional to w(X_i)``, ``p_{n+1} proportional to w(x_new)``. The
``+inf`` atom is the test point's own (unobserved) score: if its normalized
weight alone exceeds ``alpha`` the honest radius is ``+inf`` (an unbounded band
-- the weak-overlap failure mode Phase 6.5 stabilizes).

**Coverage.** Finite-sample and *marginal over ``X ~ p``* under known weights
(e.g. a randomized experiment, where ``w`` is constant and this reduces to plain
split-CP with exact coverage). Under estimated ``pi_hat`` it is *approximately*
valid and **doubly robust** in the Lei-Candes sense (holds if either ``pi`` or
the outcome model is well estimated).

Because Layer 1 reduced each curve to a scalar ``S_i``, the weighting acts on
scalars exactly as in the standard weighted-CP proof -- the composition lemma of
Phase 4.4.
"""

from __future__ import annotations

import numpy as np

from ...estimators.aipw import _clip_pi
from ...metrics import Band
from .functional_cp import modulation, residual_curves, sup_norm_score


def propensity_weights(pi_hat, arm: int):
    """Likelihood-ratio weights ``w(x)`` for the arm-``a`` -> population tilt.

    ``1/pi_hat`` for the treated arm (``arm=1``), ``1/(1-pi_hat)`` for the
    control arm (``arm=0``). ``pi_hat`` is clipped away from ``{0,1}`` with the
    same ``EPS_PI`` as the AIPW estimator, so weights stay finite (a naive
    unstabilized guard; Phase 6.5 replaces it with density-gated shrinkage).
    """
    pi = _clip_pi(pi_hat)
    if arm == 1:
        return 1.0 / pi
    if arm == 0:
        return 1.0 / (1.0 - pi)
    raise ValueError(f"arm must be 0 or 1, got {arm!r}")


def weighted_conformal_radius(scores, alpha: float, w_calib, w_new):
    """Weighted split-conformal radius, vectorized over test weights ``w_new``.

    Smallest calibration score ``k`` such that the normalized calibration weight
    mass at or below ``k`` reaches ``1-alpha`` under the test-point tilt::

        p_i = w_i / (sum_j w_j + w_new),   atom p_{new} = w_new / (sum_j w_j + w_new)
        k   = min { S : sum_{S_i <= S} p_i >= 1 - alpha }   (or +inf via the atom).

    Args:
        scores: calibration nonconformity scores ``[n]``.
        alpha: 1 - target coverage.
        w_calib: calibration weights ``[n]`` (all ones -> unweighted split-CP).
        w_new: test-point weight(s); scalar or array ``[m]``.

    Returns:
        Scalar radius if ``w_new`` is scalar, else array ``[m]``. Entries are
        ``+inf`` where even placing all calibration mass below the top score
        cannot reach ``1-alpha`` given the atom (weak-overlap / small-``n`` case).
    """
    scores = np.asarray(scores, dtype=float)
    w_calib = np.asarray(w_calib, dtype=float)
    n = scores.shape[0]

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    w_sorted = w_calib[order]
    Wcum = np.cumsum(w_sorted)          # [n], ascending
    Wtot = float(Wcum[-1])

    w_new = np.asarray(w_new, dtype=float)
    scalar = w_new.ndim == 0
    wv = np.atleast_1d(w_new)

    thresh = (1.0 - alpha) * (Wtot + wv)          # [m]
    # first index i with Wcum[i] >= thresh (Wcum ascending) -> that score is the radius
    idx = np.searchsorted(Wcum, thresh, side="left")   # in [0, n]
    radius = np.where(
        idx >= n, np.inf, s_sorted[np.clip(idx, 0, n - 1)]
    )
    return float(radius[0]) if scalar else radius


class ConformalArm:
    """A calibrated split-CP model for one arm ``a`` and homology dim ``d``.

    Holds a fitted outcome regression + propensity (a
    :class:`~tcda_uq.estimators.nuisance.NuisanceFit`), the calibration
    nonconformity scores, the modulation ``s(t)`` and (Layer 2) the calibration
    weights. Produces a **prediction** band for the potential-outcome silhouette
    ``phi^a_d(., x)`` at any new covariate ``x``.

    Args:
        nuisance: fitted :class:`NuisanceFit` (mu regressions + propensity),
            trained on data **disjoint** from ``calib_sample`` (split-CP).
        calib_sample: calibration triplet ``(phi, A, X)`` -- only the arm-``a``
            units are used.
        d: homology dimension.
        a: arm (0 control, 1 treated).
        modulation_kind: ``"pointwise-sd"`` (default), ``"constant"`` or
            ``"lipschitz"`` (see :mod:`.functional_cp`).
        floor: Lipschitz-aware floor for the modulation.
        weighted: apply the propensity likelihood-ratio weights (Layer 2). If
            ``False`` the arm is plain (exchangeable) split-CP -- exact when the
            calibration covariates already match the target (randomized design).
        propensity_feature_fn: optional ``X -> features`` map matching the one the
            propensity model was trained with (e.g. interaction terms).
        modulation_residuals: optional independent residual curves ``[m, res]``
            to compute ``s(t)`` from (keeps the scores exactly exchangeable). If
            ``None``, ``s(t)`` is the symmetric pointwise sd of the calibration
            residuals (standard DFV practice).
        weight_fn: the Layer-2 weight map ``(pi_hat, arm) -> weights``. Defaults to
            :func:`propensity_weights` (the naive ``1/pi`` tilt). Pass a
            positivity-stabilized weighter from
            :mod:`~tcda_uq.uq.conformal.stabilized_weights` (Phase 6.5) to suppress
            the weak-overlap ``+inf`` atom; the guarantee then holds for that
            weighter's (possibly tilted) target population.
    """

    def __init__(
        self,
        nuisance,
        calib_sample,
        d: int,
        a: int,
        *,
        modulation_kind: str = "pointwise-sd",
        floor: float = 0.0,
        weighted: bool = True,
        propensity_feature_fn=None,
        modulation_residuals=None,
        weight_fn=None,
    ):
        phi, A, X = calib_sample
        A = np.asarray(A)
        self.nuisance = nuisance
        self.d = d
        self.a = a
        self.weighted = weighted
        self._weight_fn = weight_fn if weight_fn is not None else propensity_weights
        self.tseq = np.asarray(nuisance.tseq, dtype=float)
        self._feat = propensity_feature_fn if propensity_feature_fn is not None else (lambda z: z)

        mask = A == a
        Xa = X[mask]
        phi_a = phi[mask, d, :]                                   # [n_a, res]
        mu_a = nuisance.predict_mu(Xa)[d][a]                      # [n_a, res]
        R = residual_curves(phi_a, mu_a)                         # [n_a, res]

        mod_R = R if modulation_residuals is None else np.asarray(modulation_residuals, float)
        self.s = modulation(mod_R, modulation_kind, floor=floor)
        self.scores = sup_norm_score(R, self.s)
        self.scores = np.atleast_1d(self.scores)

        if weighted:
            self.w_calib = self._weight_fn(self._pi(Xa), a)
        else:
            self.w_calib = np.ones(self.scores.shape[0])

    # ------------------------------------------------------------------ nuisance
    def _pi(self, X):
        return self.nuisance.prop_model.predict_proba(self._feat(np.atleast_2d(X)))[:, 1]

    def _w_new(self, X_new):
        if not self.weighted:
            return np.ones(X_new.shape[0])
        return self._weight_fn(self._pi(X_new), self.a)

    # ------------------------------------------------------------------- predict
    def center(self, X_new):
        """Predicted potential-outcome silhouette ``mu_hat_a(., x)``: ``[m, res]``."""
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        return self.nuisance.predict_mu(X_new)[self.d][self.a]

    def radius(self, X_new, alpha: float):
        """Weighted conformal radius/radii at ``X_new`` (array ``[m]``)."""
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        return np.atleast_1d(
            weighted_conformal_radius(self.scores, alpha, self.w_calib, self._w_new(X_new))
        )

    def band_bounds(self, X_new, alpha: float):
        """Prediction-band bounds for ``phi^a`` at ``X_new``.

        Returns ``(lower, upper, center)`` each ``[m, resolution]``.
        """
        c = self.center(X_new)                               # [m, res]
        k = self.radius(X_new, alpha)                        # [m]
        half = k[:, None] * self.s[None, :]                  # [m, res]
        return c - half, c + half, c

    def band(self, x_new, alpha: float) -> Band:
        """Prediction :class:`Band` for ``phi^a(., x)`` at a single ``x``."""
        lower, upper, center = self.band_bounds(x_new, alpha)
        return Band(
            tseq=self.tseq,
            lower=lower[0],
            upper=upper[0],
            center=center[0],
            level=1.0 - alpha,
            kind="prediction",
        )
