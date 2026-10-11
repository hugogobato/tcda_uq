"""Covariate-adaptive widths under training-frozen score rules.

Independent global and local scales preserve the weighted split-conformal target
scope when ratios are exact. Learned ratios require TV qualification. The optional
kernel-localized quantile is experimental and has no general conditional-coverage
guarantee. ConformalMetaLearner predicts a fitted pseudo-outcome, not the ITTE.
"""

from __future__ import annotations

import numpy as np

from ...estimators.aipw import aipw_scores
from ...estimators.nuisance import (
    NuisanceFit,
    fit_functional_regression,
    fit_propensity,
    predict_functional_regression,
)
from ...metrics import Band
from .composition import fit_split_nuisances, _indices
from ...estimators.nuisance import validate_sample
from ..asymptotic._validation import validate_alpha
from .weighted_cp import ConformalArm, weighted_conformal_radius
from .functional_cp import modulation, residual_curves, sup_norm_score
from .weighted_cp import propensity_weights


# ===========================================================================
# Local difficulty scale  sigma_hat_a(x)   (mechanism A)
# ===========================================================================
class LocalScale:
    """Fitted local difficulty ``sigma_hat_a(x)`` for arm ``a``, homology dim ``d``.

    Regresses each **training** unit's sup-norm residual magnitude
    ``m_i = max_t |phi^a_{i,d}(t) - mu_hat_a(t, X_i)| / s(t)`` on ``X`` and returns a
    strictly-positive, floored prediction. Dividing the nonconformity score by
    ``sigma_hat_a(X_i)`` is the CQR / normalized-score construction (Lei et al. 2018;
    Romano et al. 2019): it makes the band wider where the outcome is harder to
    predict and tighter where it is easier, while -- crucially -- keeping the
    calibration scores exchangeable (``sigma_hat_a`` is a fixed function, fit on data
    disjoint from calibration), the exact-ratio marginal coverage lower bound is preserved.
    Learned propensity ratios still require total-variation qualification.

    Args:
        estimator: an sklearn regressor (default :class:`RandomForestRegressor`)
            fit on ``(X_train_a, m_train_a)``. Any predictor works; the forest is a
            robust nonparametric default.
        floor_quantile: the predicted scale is floored at this quantile of the
            training magnitudes, so a near-zero prediction cannot blow up a score.
    """

    def __init__(self, estimator=None, floor_quantile: float = 0.1, random_state: int = 0):
        self._estimator = estimator
        self.floor_quantile = float(floor_quantile)
        if not np.isfinite(self.floor_quantile) or not 0 <= self.floor_quantile <= 1:
            raise ValueError("floor_quantile must lie in [0,1]")
        self.random_state = random_state
        self.floor_ = 1e-8
        self.model_ = None
        self.const_ = 1.0

    def fit(self, X, m):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        m = np.asarray(m, dtype=float)
        if X.ndim != 2 or m.ndim != 1 or X.shape[0] != m.size or not m.size or not np.isfinite(X).all() or not np.isfinite(m).all() or np.any(m < 0):
            raise ValueError("local scale requires finite independent training covariates and nonnegative magnitudes")
        self.floor_ = max(float(np.quantile(m, self.floor_quantile)), 1e-8)
        # constant fallback when there is too little data to regress difficulty
        self.const_ = max(float(np.mean(m)), self.floor_)
        if X.shape[0] >= 10 and np.std(m) > 1e-12:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.base import clone

            # seed the forest so the local scale (and thus the band) is reproducible
            est = RandomForestRegressor(n_estimators=100, min_samples_leaf=5,
                                        random_state=self.random_state)
            self.model_ = (clone(self._estimator) if self._estimator is not None else est)
            self.model_.fit(X, m)
        return self

    def predict(self, X):
        """Local scale ``sigma_hat_a(x)`` at rows of ``X`` (array ``[m]``, > 0)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.model_ is None:
            out = np.full(X.shape[0], self.const_)
        else:
            out = np.asarray(self.model_.predict(X), dtype=float)
        if not np.isfinite(out).all():
            raise ValueError("local scale predictions must be finite")
        return np.maximum(out, self.floor_)


def fit_local_scale(train_sample, nuisance, d, a, s, *, estimator=None,
                    floor_quantile: float = 0.1) -> LocalScale:
    """Fit :class:`LocalScale` from the arm-``a`` training residuals (dim ``d``).

    ``s`` is the (global) modulation ``s_a(t)`` used to scalarize the residual curve;
    the same ``s`` is used at band time, so the divisor factorizes as
    ``s_a(t) * sigma_hat_a(x)``.
    """
    phi, A, X = train_sample
    A = np.asarray(A)
    mask = A == a
    Xa = np.atleast_2d(np.asarray(X, dtype=float))[mask]
    phi_a = np.asarray(phi, dtype=float)[mask, d, :]
    mu_a = nuisance.predict_mu(Xa)[d][a]
    R = residual_curves(phi_a, mu_a)
    m = sup_norm_score(R, s)                       # [n_a] training magnitudes
    m = np.atleast_1d(m)
    return LocalScale(estimator=estimator, floor_quantile=floor_quantile).fit(Xa, m)


# ===========================================================================
# Kernel localization  (mechanism B)
# ===========================================================================
def _standardize(X, ref):
    ref = np.atleast_2d(np.asarray(ref, dtype=float))
    sd = ref.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return np.atleast_2d(np.asarray(X, dtype=float)) / sd, sd


def median_bandwidth(X, *, subsample: int = 500, rng=None) -> float:
    """Median-heuristic kernel bandwidth on **standardized** covariates.

    ``h`` = median pairwise Euclidean distance (a robust, scale-free default; the
    covariates are standardized first so no single coordinate dominates). Uses a
    random subsample of rows when ``X`` is large.
    """
    Xs, _ = _standardize(X, X)
    n = Xs.shape[0]
    rng = np.random.default_rng(rng)
    if n > subsample:
        Xs = Xs[rng.choice(n, subsample, replace=False)]
    diff = Xs[:, None, :] - Xs[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(Xs.shape[0], k=1)
    med = float(np.median(dist[iu])) if iu[0].size else 1.0
    return med if med > 1e-8 else 1.0


def kernel_weights(X_calib, x_new, bandwidth, ref):
    """Gaussian covariate-space kernel weights ``K_h(x_new, X_calib)``: ``[m, n]``.

    Row ``t`` holds the localization weights of every calibration unit for test point
    ``x_new[t]``. Covariates are standardized by ``ref`` (the training/calibration
    spread) so the isotropic bandwidth is meaningful. ``bandwidth -> inf`` recovers
    uniform (global, un-localized) weights.
    """
    ref_s, sd = _standardize(ref, ref)
    Xc = np.atleast_2d(np.asarray(X_calib, dtype=float)) / sd
    Xn = np.atleast_2d(np.asarray(x_new, dtype=float)) / sd
    d2 = ((Xn[:, None, :] - Xc[None, :, :]) ** 2).sum(-1)     # [m, n]
    return np.exp(-0.5 * d2 / (bandwidth ** 2))


def localized_weighted_radius(scores, alpha, W_calib, w_new):
    """Weighted split-conformal radius with **per-test-point** calibration weights.

    Generalizes :func:`~tcda_uq.uq.conformal.weighted_cp.weighted_conformal_radius`
    to the localized case, where each test point ``t`` carries its own calibration
    weight vector ``W_calib[t]`` (kernel centered at ``x_new[t]``). For each ``t`` the
    radius is the smallest score ``S`` whose normalized weight mass under the tilt
    ``p_i = W_calib[t,i] / (sum_i W_calib[t,i] + w_new[t])`` reaches ``1 - alpha``
    (else ``+inf`` via the test atom).

    Args:
        scores: calibration scores ``[n]``.
        alpha: 1 - target coverage.
        W_calib: ``[m, n]`` calibration weights, one row per test point.
        w_new: ``[m]`` test-point atom weights.

    Returns:
        ``[m]`` radii (``+inf`` where the atom alone exceeds ``alpha``).
    """
    W = np.atleast_2d(np.asarray(W_calib, dtype=float))
    wn = np.atleast_1d(np.asarray(w_new, dtype=float))
    if W.ndim != 2 or W.shape != (wn.size, np.asarray(scores).size):
        raise ValueError("localized weights must have shape [m, n_scores]")
    return np.array([weighted_conformal_radius(scores, alpha, row, atom)
                     for row, atom in zip(W, wn)])


# ===========================================================================
# Adaptive arm  (composition of mechanisms A + B on one arm)
# ===========================================================================
class AdaptiveConformalArm(ConformalArm):
    """Covariate-adaptive split-CP for one arm ``a``, homology dim ``d`` (Phase 5.1).

    Extends :class:`~tcda_uq.uq.conformal.weighted_cp.ConformalArm` with the two
    Phase-5 adaptivity mechanisms:

      * a fitted :class:`LocalScale` ``sigma_hat_a`` divides the nonconformity scores
        and multiplies the band half-width (mechanism A -- X-dependent width, preserving the marginal
        target scope when weights are actual likelihood ratios);
      * optional kernel localization of the conformal quantile at band time
        (mechanism B -- experimental, no general conditional guarantee).

    Produces a **prediction** band for ``phi^a_d(., x)`` at any ``x``.

    Args:
        nuisance: fitted :class:`NuisanceFit`, trained on data disjoint from
            ``calib_sample``.
        calib_sample: calibration triplet ``(phi, A, X)`` (only arm-``a`` units used).
        d, a: homology dim and arm.
        local_scale: a fitted :class:`LocalScale` (mechanism A). ``None`` disables it
            (falls back to the un-normalized :class:`ConformalArm` behavior).
        localize: enable kernel localization (mechanism B).
        bandwidth: kernel bandwidth on standardized covariates (default: the
            median heuristic on independent training covariates).
        modulation_kind, floor: global modulation ``s_a(t)`` (see functional_cp).
        weighted: apply propensity likelihood-ratio weights (Layer 2).
        propensity_feature_fn: ``X -> features`` map matching the propensity model.
    """

    def __init__(
        self,
        nuisance,
        calib_sample,
        d: int,
        a: int,
        *,
        local_scale: LocalScale | None = None,
        localize: bool = False,
        bandwidth: float | None = None,
        modulation_kind: str = "constant",
        floor: float = 0.0,
        weighted: bool = True,
        propensity_feature_fn=None,
        modulation_residuals=None,
        modulation_scale=None,
        weight_fn=None,
        target="full",
        propensity_fn=None,
        weights_known=False,
        localization_reference=None,
    ):
        super().__init__(nuisance, calib_sample, d, a,
                         modulation_kind=modulation_kind, floor=floor, weighted=weighted,
                         propensity_feature_fn=propensity_feature_fn,
                         modulation_residuals=modulation_residuals,
                         modulation_scale=modulation_scale, weight_fn=weight_fn,
                         target=target, propensity_fn=propensity_fn, weights_known=weights_known)
        self.localize, self.local_scale = localize, local_scale
        self.Xa = np.asarray(calib_sample[2], float)[self.calib_index_]
        self.sigma_calib = np.ones(self.Xa.shape[0]) if local_scale is None or self.Xa.shape[0] == 0 else np.asarray(local_scale.predict(self.Xa), float)
        if self.sigma_calib.shape != self.scores.shape or not np.isfinite(self.sigma_calib).all() or np.any(self.sigma_calib <= 0):
            raise ValueError("local_scale must return finite strictly positive [m] values")
        self.scores = self.scores / self.sigma_calib
        if localize:
            if localization_reference is None:
                raise ValueError("localization needs an independent training localization_reference")
            self.localization_reference = np.asarray(localization_reference, float)
            self.bandwidth = median_bandwidth(self.localization_reference) if bandwidth is None else bandwidth
            if not np.isfinite(self.bandwidth) or self.bandwidth <= 0:
                raise ValueError("bandwidth must be finite and positive")
            self.coverage_scope_ = "experimental localized quantile; no general marginal or conditional guarantee"
        else:
            self.bandwidth = np.inf

    def _sigma_new(self, X_new):
        if self.local_scale is None:
            return np.ones(np.atleast_2d(X_new).shape[0])
        sig = np.asarray(self.local_scale.predict(X_new), float)
        if sig.shape != (np.atleast_2d(X_new).shape[0],) or not np.isfinite(sig).all() or np.any(sig <= 0):
            raise ValueError("local scale must return finite strictly positive [m] predictions")
        return sig

    def radius(self, X_new, alpha: float):
        """Adaptive conformal radius/radii at ``X_new`` (array ``[m]``)."""
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        w_new = self._w_new(X_new)
        if self.localize:
            Kw = kernel_weights(self.Xa, X_new, self.bandwidth, self.localization_reference)   # [m, n_a]
            W_calib = Kw * self.w_calib[None, :]
            return localized_weighted_radius(self.scores, alpha, W_calib, w_new)
        from .weighted_cp import weighted_conformal_radius
        return np.atleast_1d(
            weighted_conformal_radius(self.scores, alpha, self.w_calib, w_new)
        )

    def band_bounds(self, X_new, alpha: float):
        """Adaptive prediction-band bounds for ``phi^a`` at ``X_new``.

        Half-width is ``k_a(x) * s_a(t) * sigma_hat_a(x)`` -- the ``sigma_hat_a(x)``
        factor is what makes the width depend on ``x``. Returns ``(lower, upper,
        center)`` each ``[m, resolution]``.
        """
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        c = self.center(X_new)                               # [m, res]
        k = self.radius(X_new, alpha)                        # [m]
        sig = self._sigma_new(X_new)                         # [m]
        half = (k * sig)[:, None] * self.s[None, :]          # [m, res]
        return c - half, c + half, c

    def band(self, x_new, alpha: float) -> Band:
        """Adaptive prediction :class:`Band` for ``phi^a(., x)`` at a single ``x``."""
        lower, upper, center = self.band_bounds(x_new, alpha)
        return Band(tseq=self.tseq, lower=lower[0], upper=upper[0], center=center[0],
                    level=1.0 - alpha, kind="prediction")


# ===========================================================================
# Composed adaptive ITTE predictor
# ===========================================================================
class AdaptiveITTEConformal:
    """Covariate-adaptive composed ITTE predictor (Phase 5.1).

    The Phase-5 counterpart of :class:`~tcda_uq.uq.conformal.ITTEConformal`: same
    Layer-3 arm composition (interval arithmetic + Bonferroni ``alpha/2``), but each
    arm is an :class:`AdaptiveConformalArm`, so the band **width adapts to ``x``** via
    the local scale ``sigma_hat_a(x)`` (mechanism A) and, optionally, the conformal
    quantile is kernel-localized (mechanism B).

    Usage::

        model = AdaptiveITTEConformal.fit(sample, tseq, propensity_feature_fn=feat)
        band  = model.band(x_new, alpha=0.1, d=0)
        lo, hi, ctr = model.band_bounds(X_test, alpha=0.1, d=0)
    """

    def __init__(self, arms0, arms1, tseq):
        self.arms0 = arms0
        self.arms1 = arms1
        self.tseq = np.asarray(tseq, dtype=float)
        self.n_hom_dim = len(arms0)

    @classmethod
    def fit(
        cls,
        sample,
        tseq,
        *,
        calib_frac: float = 0.5,
        n_basis: int = 5,
        propensity_estimator=None,
        propensity_feature_fn=None,
        modulation_kind: str = "constant",
        floor: float = 0.0,
        weighted: bool = True,
        local_scale: bool = True,
        localize: bool = False,
        bandwidth: float | None = None,
        scale_estimator=None,
        random_state=0,
        nuisance: NuisanceFit | None = None,
        calib_index=None,
        train_index=None,
        target="full",
        weight_fn=None,
        propensity_fn=None,
        weights_known=False,
    ):
        """Fit the covariate-adaptive ITTE conformal predictor.

        Args mirror :meth:`ITTEConformal.fit`, plus:
            local_scale: fit + apply the local difficulty ``sigma_hat_a(x)`` (mechanism A).
            localize: kernel-localize the conformal quantile (mechanism B).
            bandwidth: localization bandwidth (default: median heuristic).
            scale_estimator: sklearn regressor for :class:`LocalScale` (default forest).
            train_index: when supplying a pre-fit ``nuisance``, the indices its local
                scales should be trained from (needed to fit ``sigma_hat_a`` on the
                nuisance-training split); required if ``local_scale`` and ``nuisance``.
        """
        phi, A, X, tseq = validate_sample(sample, tseq)
        if not np.isfinite(calib_frac) or not 0 < calib_frac < 1:
            raise ValueError("calib_frac must be strictly between 0 and 1")
        n = phi.shape[0]
        n_hom_dim = phi.shape[-2]

        if nuisance is None:
            rng = np.random.default_rng(random_state)
            perm = rng.permutation(n)
            n_cal = min(max(int(round(calib_frac * n)), 1), n-1)
            cal_idx = np.sort(perm[:n_cal])
            tr_idx = np.sort(perm[n_cal:])
            nuisance = fit_split_nuisances(
                (phi[tr_idx], A[tr_idx], X[tr_idx]), tseq, n_basis=n_basis,
                propensity_estimator=propensity_estimator,
                propensity_feature_fn=propensity_feature_fn,
            )
        else:
            if calib_index is None:
                raise ValueError("supplying `nuisance` also requires `calib_index`")
            cal_idx = _indices(calib_index, n, "calib_index")
            if (local_scale or localize or modulation_kind != "constant") and train_index is None:
                raise ValueError("`local_scale=True` with a pre-fit `nuisance` needs "
                                 "`train_index` (the nuisance-training units)")
            tr_idx = None if train_index is None else _indices(train_index, n, "train_index")
            if tr_idx is not None and np.intersect1d(cal_idx, tr_idx).size:
                raise ValueError("training and calibration indices must be disjoint")

        calib = (phi[cal_idx], A[cal_idx], X[cal_idx])
        train = None if tr_idx is None else (phi[tr_idx], A[tr_idx], X[tr_idx])

        # Both global modulation and optional local scale are fixed on training.
        scales0, scales1 = [None]*n_hom_dim, [None]*n_hom_dim
        shapes = {}
        for d in range(n_hom_dim):
            for a, store in ((0, scales0), (1, scales1)):
                if train is None:
                    s_shape = np.ones(tseq.size)
                else:
                    mask = train[1] == a
                    Xa = train[2][mask]
                    R = residual_curves(train[0][mask, d, :], nuisance.predict_mu(Xa)[d][a])
                    s_shape = modulation(R, modulation_kind, floor=floor)
                shapes[d, a] = s_shape
                if local_scale:
                    store[d] = fit_local_scale(train, nuisance, d, a, s_shape, estimator=scale_estimator)

        arm_kw = dict(
            modulation_kind=modulation_kind, floor=floor, weighted=weighted,
            propensity_feature_fn=propensity_feature_fn,
            localize=localize, bandwidth=bandwidth, target=target, weight_fn=weight_fn,
            propensity_fn=propensity_fn, weights_known=weights_known,
            localization_reference=None if train is None else train[2],
        )
        arms0 = [AdaptiveConformalArm(nuisance, calib, d, 0, local_scale=scales0[d], modulation_scale=shapes[d, 0], **arm_kw)
                 for d in range(n_hom_dim)]
        arms1 = [AdaptiveConformalArm(nuisance, calib, d, 1, local_scale=scales1[d], modulation_scale=shapes[d, 1], **arm_kw)
                 for d in range(n_hom_dim)]
        model = cls(arms0, arms1, tseq)
        model.nuisance_ = nuisance
        model.calib_index_ = cal_idx
        model.target_ = arms0[0].target_
        model.weights_known_ = arms0[0].weights_known_
        model.coverage_scope_ = arms0[0].coverage_scope_
        return model

    def band_bounds(self, X_new, alpha: float, d: int = 0):
        """Batched adaptive ITTE band bounds for homology dim ``d``.

        Returns ``(lower, upper, center)`` each ``[m, resolution]``.
        """
        validate_alpha(alpha)
        l1, u1, c1 = self.arms1[d].band_bounds(X_new, alpha / 2.0)
        l0, u0, c0 = self.arms0[d].band_bounds(X_new, alpha / 2.0)
        center = c1 - c0
        half = 0.5 * (u1 - l1) + 0.5 * (u0 - l0)
        return center - half, center + half, center

    def band(self, x_new, alpha: float, d: int = 0) -> Band:
        """Adaptive ITTE prediction :class:`Band` for ``delta_d(., x)`` at one ``x``."""
        lower, upper, center = self.band_bounds(x_new, alpha, d)
        return Band(tseq=self.tseq, lower=lower[0], upper=upper[0], center=center[0],
                    level=1.0 - alpha, kind="prediction")

    def bands(self, x_new, alpha: float):
        """One adaptive ITTE band per homology dim at a single ``x``."""
        return [self.band(x_new, alpha, d) for d in range(self.n_hom_dim)]


# ===========================================================================
# Phase 5.2 -- conformal meta-learner (Alaa et al. 2023)
# ===========================================================================
class ConformalMetaLearner:
    """Experimental split-conformal prediction for a fitted pseudo-outcome draw.

    Conditional on training, a frozen nuisance transform maps each observed unit
    to a pseudo-outcome. This object's prediction target is that transformed draw,
    which is distinct from both the mean CTATE and the unobserved individual effect.
    No ITTE or CTATE coverage theorem is claimed. The mean identity requires correct
    nuisance assumptions and does not identify the pseudo-outcome with the ITTE.
    """

    def __init__(self, learner, scores, s, local_scale, tseq, *, weighted, feat):
        self.learner = learner              # fitted CTATEDRLearner (on train)
        self.scores = scores                # list over dim, calibration scores [n_cal]
        self.s = s                          # list over dim, modulation [res]
        self.local_scale = local_scale      # list over dim, LocalScale or None
        self.tseq = np.asarray(tseq, dtype=float)
        self.weighted = weighted
        self._feat = feat
        self.n_hom_dim = len(scores)

    @classmethod
    def fit(
        cls,
        sample,
        tseq,
        *,
        calib_frac: float = 0.5,
        n_basis: int = 5,
        propensity_estimator=None,
        propensity_feature_fn=None,
        feature_fn=None,
        modulation_kind: str = "constant",
        floor: float = 0.0,
        local_scale: bool = True,
        scale_estimator=None,
        random_state=0,
    ):
        """Fit the conformal meta-learner via a train/calibration split.

        Stage 1+2 (nuisances + ``tau_hat``) are fit on the training split; the
        pseudo-outcomes of the held-out calibration units are computed with the
        *training* nuisances (so calibration residuals are exchangeable with a fresh
        test residual) and scored against ``tau_hat``.
        """
        from ...estimators import CTATEDRLearner

        phi, A, X, tseq = validate_sample(sample, tseq)
        if not np.isfinite(calib_frac) or not 0 < calib_frac < 1:
            raise ValueError("calib_frac must lie strictly between 0 and 1")
        n = phi.shape[0]
        n_hom = phi.shape[-2]
        feat = propensity_feature_fn if propensity_feature_fn is not None else (lambda z: z)

        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n)
        n_cal = min(max(int(round(calib_frac * n)), 1), n-1)
        cal_idx = np.sort(perm[:n_cal])
        tr_idx = np.sort(perm[n_cal:])
        train = (phi[tr_idx], A[tr_idx], X[tr_idx])
        Xcal = X[cal_idx]

        # stage 1+2 on train: CTATE DR-learner gives tau_hat(., x)
        learner = CTATEDRLearner(n_basis=n_basis, feature_fn=feature_fn).fit(
            train, tseq, n_splits=2,
            propensity_estimator=propensity_estimator,
            propensity_feature_fn=propensity_feature_fn,
            random_state=int(rng.integers(1 << 31)),
        )
        # train nuisances (fit on the whole train split) to score calibration units
        reg = fit_functional_regression(train, tseq, n_basis=n_basis)
        prop = fit_propensity(feat(train[2]), train[1], propensity_estimator)
        mu_cal = predict_functional_regression(reg, Xcal, tseq)      # list (mu0, mu1)
        pi_cal = np.clip(prop.predict_proba(feat(Xcal))[:, 1], .01, .99)
        psi_cal = aipw_scores(pi_cal, mu_cal, (phi[cal_idx], A[cal_idx], Xcal))  # list [n_cal, res]

        mu_train = predict_functional_regression(reg, train[2], tseq)
        pi_train = np.clip(prop.predict_proba(feat(train[2]))[:, 1], .01, .99)
        pi_cal = np.clip(pi_cal, .01, .99)
        psi_train = aipw_scores(pi_train, mu_train, train)
        tau_train = learner.predict(train[2])
        tau_cal = learner.predict(Xcal)                              # [n_cal, hom, res]

        scores, s_list, scale_list = [], [], []
        for d in range(n_hom):
            R = psi_cal[d] - tau_cal[:, d, :]                        # [n_cal, res]
            R_train = psi_train[d] - tau_train[:, d, :]
            s_shape = modulation(R_train, modulation_kind, floor=floor)
            raw_train = np.atleast_1d(sup_norm_score(R_train, s_shape))
            raw = np.atleast_1d(sup_norm_score(R, s_shape))
            if local_scale:
                ls = LocalScale(estimator=scale_estimator).fit(train[2], raw_train)
                sig = ls.predict(Xcal)
                scale_list.append(ls)
                scores.append(raw / sig)
            else:
                scale_list.append(None)
                scores.append(raw)
            s_list.append(s_shape)

        model = cls(learner, scores, s_list, scale_list, tseq,
                    weighted=False, feat=feat)
        model.calib_index_ = cal_idx
        model.nuisance_ = NuisanceFit(reg, prop, tseq, propensity_feature_fn)
        model.pseudo_outcome_propensity_clip_ = .01
        model.coverage_scope_ = "prediction for the frozen fitted pseudo-outcome transform only"
        return model

    def band_bounds(self, X_new, alpha: float, d: int = 0):
        from .functional_cp import split_conformal_radius

        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        center = self.learner.predict(X_new)[:, d, :]               # [m, res]
        k = split_conformal_radius(self.scores[d], alpha)
        if self.local_scale[d] is None:
            sig = np.ones(X_new.shape[0])
        else:
            sig = self.local_scale[d].predict(X_new)
        half = (k * sig)[:, None] * self.s[d][None, :]
        return center - half, center + half, center

    def band(self, x_new, alpha: float, d: int = 0) -> Band:
        lower, upper, center = self.band_bounds(x_new, alpha, d)
        return Band(tseq=self.tseq, lower=lower[0], upper=upper[0], center=center[0],
                    level=1.0 - alpha, kind="prediction")

    def bands(self, x_new, alpha: float):
        return [self.band(x_new, alpha, d) for d in range(self.n_hom_dim)]


# ===========================================================================
# Phase 5.3 -- CTATE-level prediction band (pairs with the Phase-3 confidence band)
# ===========================================================================
def ctate_prediction_band(model, x, d: int = 0, *, alpha: float = 0.1) -> Band:
    """Legacy adapter to model.band, with that model's marginal target scope.

    This name does not assert conditional coverage at a fixed x, nor prediction
    of the conditional mean. Composed arm models predict individual effects;
    ConformalMetaLearner predicts a fitted pseudo-outcome instead.
    """
    band = model.band(np.atleast_2d(np.asarray(x, dtype=float)), alpha, d)
    band.kind = "prediction"
    return band
