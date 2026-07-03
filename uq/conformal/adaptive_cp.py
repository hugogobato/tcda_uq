"""Covariate-adaptive / conditional conformal for the ITTE (Phase 5).

The Phase-4 composed band :class:`~tcda_uq.uq.conformal.ITTEConformal` is
*marginally* valid: ``P(delta_{n+1,d}(.) in C(X_{n+1}, .)) >= 1 - alpha`` where the
probability averages over ``X``. Its **center** ``mu_hat_a(., x)`` moves with ``x``,
but its **width** ``k_a * s_a(t)`` is essentially constant in ``x`` (the modulation
``s_a(t)`` is global; ``k_a`` only shifts through the weak-overlap ``+inf`` atom).
So the band is *not* adapted to ``x``: it is equally wide in easy and hard regions
of covariate space, and its coverage, while correct on average, can be too high in
low-noise regions and too low in high-noise ones (**conditional** miscoverage).

Phase 5 makes the prediction band **adapt to ``X``** -- the conformal counterpart of
the CTATE (Phase 3). Exact conditional coverage ``P(. | X = x) = 1 - alpha`` at every
``x`` is impossible in finite samples with a distribution-free guarantee (Vovk 2012;
Lei & Wasserman 2014; Foygel Barber et al. 2021), so we target *approximate*
conditional coverage via two orthogonal, composable mechanisms:

  * **(A) Locally-scaled (normalized) scores** -- CQR / mean-absolute-deviation
    normalization (Lei et al. 2018; Romano et al. 2019). Divide each score by a
    fitted local difficulty ``sigma_hat_a(x)``, so the half-width becomes
    ``k_a * s_a(t) * sigma_hat_a(x)`` -- *X-dependent by construction*. Because
    ``sigma_hat_a`` is frozen on the nuisance-training split (disjoint from
    calibration), the scores stay exchangeable and **marginal** validity is exact;
    the win is efficiency and better conditional coverage.

  * **(B) Kernel-localized conformal quantile** -- weight the calibration scores by
    a covariate-space kernel ``K_h(x, X_i)`` (Vovk 2012 "conditional validity";
    Lei & Wasserman 2014; Guan 2023 "localized CP"), so the quantile ``k_a(x)`` is
    driven by calibration units *near* ``x``. This composes multiplicatively with
    the Phase-4.2 propensity likelihood-ratio weights (both are just weights fed to
    :func:`~tcda_uq.uq.conformal.weighted_cp.weighted_conformal_radius`), so causal
    reweighting and localization coexist. Localization buys approximate conditional
    coverage at the cost of *exact* marginal validity (the price of conditioning).

Mechanism (A) is on by default (``local_scale=True``); (B) is optional
(``localize=True``). Arms are still composed at ``alpha/2`` by interval arithmetic
(Phase 4.3), so the output is a simultaneous-in-``t`` **prediction** band whose width
now varies with ``x`` -- the Phase-5 exit criterion.

Phase 5.2 adds :class:`ConformalMetaLearner`: a *conformal meta-learner* (Alaa et al.
2023) that conformalizes the DR-learner pseudo-outcome directly, giving an
adaptive-by-construction CATE-level prediction band; see its docstring for the
honest can-/cannot-claim framing. Phase 5.3's confidence-vs-prediction contrast at a
fixed ``x`` is served by :func:`ctate_prediction_band` paired with
:func:`~tcda_uq.uq.asymptotic.ctate_bands.ctate_confidence_band`.
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
from .composition import fit_split_nuisances
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
    disjoint from calibration), so **marginal coverage stays exact**.

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
        self.random_state = random_state
        self.floor_ = 1e-8
        self.model_ = None
        self.const_ = 1.0

    def fit(self, X, m):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        m = np.asarray(m, dtype=float)
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
    scores = np.asarray(scores, dtype=float)
    W_calib = np.atleast_2d(np.asarray(W_calib, dtype=float))
    w_new = np.atleast_1d(np.asarray(w_new, dtype=float))
    n = scores.shape[0]

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]                          # [n] ascending
    Wc = W_calib[:, order]                            # [m, n]
    Wcum = np.cumsum(Wc, axis=1)                       # [m, n]
    Wtot = Wcum[:, -1]                                 # [m]

    thresh = (1.0 - alpha) * (Wtot + w_new)           # [m]
    # first index j (per row) with Wcum[t, j] >= thresh[t]
    ge = Wcum >= thresh[:, None]                       # [m, n]
    has = ge.any(axis=1)
    idx = np.where(has, ge.argmax(axis=1), n)          # n -> +inf
    radius = np.where(idx >= n, np.inf, s_sorted[np.clip(idx, 0, n - 1)])
    return radius


# ===========================================================================
# Adaptive arm  (composition of mechanisms A + B on one arm)
# ===========================================================================
class AdaptiveConformalArm:
    """Covariate-adaptive split-CP for one arm ``a``, homology dim ``d`` (Phase 5.1).

    Extends :class:`~tcda_uq.uq.conformal.weighted_cp.ConformalArm` with the two
    Phase-5 adaptivity mechanisms:

      * a fitted :class:`LocalScale` ``sigma_hat_a`` divides the nonconformity scores
        and multiplies the band half-width (mechanism A -- X-dependent width, exact
        marginal validity);
      * optional kernel localization of the conformal quantile at band time
        (mechanism B -- approximate conditional coverage).

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
            median heuristic on the calibration covariates).
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
        modulation_kind: str = "pointwise-sd",
        floor: float = 0.0,
        weighted: bool = True,
        propensity_feature_fn=None,
    ):
        phi, A, X = calib_sample
        A = np.asarray(A)
        self.nuisance = nuisance
        self.d = d
        self.a = a
        self.weighted = weighted
        self.localize = localize
        self.local_scale = local_scale
        self.tseq = np.asarray(nuisance.tseq, dtype=float)
        self._feat = propensity_feature_fn if propensity_feature_fn is not None else (lambda z: z)

        mask = A == a
        self.Xa = np.atleast_2d(np.asarray(X, dtype=float))[mask]     # [n_a, p]
        phi_a = np.asarray(phi, dtype=float)[mask, d, :]
        mu_a = nuisance.predict_mu(self.Xa)[d][a]
        R = residual_curves(phi_a, mu_a)                             # [n_a, res]

        self.s = modulation(R, modulation_kind, floor=floor)         # [res] global shape
        raw = np.atleast_1d(sup_norm_score(R, self.s))               # [n_a]
        # mechanism A: normalize by the local difficulty at each calibration unit
        if local_scale is not None:
            self.sigma_calib = local_scale.predict(self.Xa)          # [n_a]
        else:
            self.sigma_calib = np.ones(self.Xa.shape[0])
        self.scores = raw / self.sigma_calib

        if weighted:
            self.w_calib = propensity_weights(self._pi(self.Xa), a)
        else:
            self.w_calib = np.ones(self.scores.shape[0])

        self.bandwidth = (
            bandwidth if bandwidth is not None
            else (median_bandwidth(self.Xa) if localize else np.inf)
        )

    # ------------------------------------------------------------------ nuisance
    def _pi(self, X):
        return self.nuisance.prop_model.predict_proba(self._feat(np.atleast_2d(X)))[:, 1]

    def _w_new(self, X_new):
        if not self.weighted:
            return np.ones(X_new.shape[0])
        return propensity_weights(self._pi(X_new), self.a)

    # ------------------------------------------------------------------- predict
    def center(self, X_new):
        """Predicted potential-outcome silhouette ``mu_hat_a(., x)``: ``[m, res]``."""
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        return self.nuisance.predict_mu(X_new)[self.d][self.a]

    def _sigma_new(self, X_new):
        if self.local_scale is None:
            return np.ones(np.atleast_2d(X_new).shape[0])
        return self.local_scale.predict(X_new)

    def radius(self, X_new, alpha: float):
        """Adaptive conformal radius/radii at ``X_new`` (array ``[m]``)."""
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        w_new = self._w_new(X_new)
        if self.localize:
            Kw = kernel_weights(self.Xa, X_new, self.bandwidth, self.Xa)   # [m, n_a]
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
        modulation_kind: str = "pointwise-sd",
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
        phi, A, X = sample
        phi = np.asarray(phi, dtype=float)
        A = np.asarray(A)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        tseq = np.asarray(tseq, dtype=float)
        n = phi.shape[0]
        n_hom_dim = phi.shape[-2]

        if nuisance is None:
            rng = np.random.default_rng(random_state)
            perm = rng.permutation(n)
            n_cal = int(round(calib_frac * n))
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
            cal_idx = np.asarray(calib_index)
            if local_scale and train_index is None:
                raise ValueError("`local_scale=True` with a pre-fit `nuisance` needs "
                                 "`train_index` (the nuisance-training units)")
            tr_idx = None if train_index is None else np.asarray(train_index)

        calib = (phi[cal_idx], A[cal_idx], X[cal_idx])
        train = None if tr_idx is None else (phi[tr_idx], A[tr_idx], X[tr_idx])

        # fit the per-arm/per-dim local difficulty on the *training* split
        scales0 = [None] * n_hom_dim
        scales1 = [None] * n_hom_dim
        if local_scale:
            for d in range(n_hom_dim):
                # base modulation shape from the training residuals (a fixed function)
                for a, store in ((0, scales0), (1, scales1)):
                    Aa = np.asarray(train[1])
                    m = Aa == a
                    phi_a = train[0][m, d, :]
                    mu_a = nuisance.predict_mu(np.atleast_2d(train[2])[m])[d][a]
                    s_shape = modulation(residual_curves(phi_a, mu_a),
                                         modulation_kind, floor=floor)
                    store[d] = fit_local_scale(train, nuisance, d, a, s_shape,
                                               estimator=scale_estimator)

        arm_kw = dict(
            modulation_kind=modulation_kind, floor=floor, weighted=weighted,
            propensity_feature_fn=propensity_feature_fn,
            localize=localize, bandwidth=bandwidth,
        )
        arms0 = [AdaptiveConformalArm(nuisance, calib, d, 0, local_scale=scales0[d], **arm_kw)
                 for d in range(n_hom_dim)]
        arms1 = [AdaptiveConformalArm(nuisance, calib, d, 1, local_scale=scales1[d], **arm_kw)
                 for d in range(n_hom_dim)]
        model = cls(arms0, arms1, tseq)
        model.nuisance_ = nuisance
        model.calib_index_ = cal_idx
        return model

    def band_bounds(self, X_new, alpha: float, d: int = 0):
        """Batched adaptive ITTE band bounds for homology dim ``d``.

        Returns ``(lower, upper, center)`` each ``[m, resolution]``.
        """
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
    """Conformal meta-learner for the (topological) CATE (Phase 5.2; Alaa et al. 2023).

    Rather than composing two per-arm bands (the Phase-4 route), a *conformal
    meta-learner* conformalizes the **DR-learner pseudo-outcome directly**. The
    pseudo-outcome for the topological effect is the per-unit AIPW/EIF score curve

        psi_{i,d}(t) = mu_hat_1(t, X_i) - mu_hat_0(t, X_i)
                       + A_i/pi (phi_i - mu_hat_1) - (1-A_i)/(1-pi) (phi_i - mu_hat_0),

    whose conditional mean is the CTATE ``tau_d(., x)`` (Kim & Lee eq. 8). Fitting a
    function-on-scalar regression ``tau_hat_d(., x)`` of ``psi`` on ``X`` (the
    :class:`~tcda_uq.estimators.CTATEDRLearner`) and split-conformalizing the residual
    sup-norm score

        S_i = max_t | psi_{i,d}(t) - tau_hat_d(t, X_i) | / (s(t) * sigma_hat(X_i))

    gives a band ``tau_hat_d(., x) +/- k * s(t) * sigma_hat(x)`` -- **adaptive to ``x``
    by construction** (centered at the CATE estimate, width normalized by the local
    scale). One band, no arm Bonferroni.

    **Can-claim / cannot-claim (kept honest).** This band is a finite-sample,
    marginally-valid **prediction** band for a *draw of the pseudo-outcome* ``psi``.
    ``E[psi | X = x] = tau_d(., x)`` (the CTATE), so it is centered correctly and is a
    legitimate meta-learner route to individual-effect inference (Alaa et al. 2023),
    and it is naturally covariate-adaptive. But ``psi`` is **not** the ITTE ``delta_i``
    itself: its spread carries the inverse-propensity inflation of the EIF, not the
    clean aleatoric spread of ``delta``. So its coverage of the *true* ``delta`` is
    only *approximate* and can be conservative under weak overlap -- it is offered as
    an adaptive **alternative/benchmark** to the composed :class:`AdaptiveITTEConformal`
    band, not a replacement for it. (For the mean ``tau`` itself, use the Phase-3
    confidence band -- CP covers a draw, not an expectation.)
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
        modulation_kind: str = "pointwise-sd",
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

        phi, A, X = sample
        phi = np.asarray(phi, dtype=float)
        A = np.asarray(A)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        tseq = np.asarray(tseq, dtype=float)
        n = phi.shape[0]
        n_hom = phi.shape[-2]
        feat = propensity_feature_fn if propensity_feature_fn is not None else (lambda z: z)

        rng = np.random.default_rng(random_state)
        perm = rng.permutation(n)
        n_cal = int(round(calib_frac * n))
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
        pi_cal = prop.predict_proba(feat(Xcal))[:, 1]
        psi_cal = aipw_scores(pi_cal, mu_cal, (phi[cal_idx], A[cal_idx], Xcal))  # list [n_cal, res]

        tau_cal = learner.predict(Xcal)                              # [n_cal, hom, res]

        scores, s_list, scale_list = [], [], []
        for d in range(n_hom):
            R = psi_cal[d] - tau_cal[:, d, :]                        # [n_cal, res]
            s_shape = modulation(R, modulation_kind, floor=floor)
            raw = np.atleast_1d(sup_norm_score(R, s_shape))
            if local_scale:
                ls = LocalScale(estimator=scale_estimator).fit(Xcal, raw)
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
    """CTATE-level **prediction** band for a draw at fixed ``x`` (Phase 5.3).

    A thin adapter that returns the adaptive ITTE band at ``x`` as the *prediction*
    counterpart of :func:`~tcda_uq.uq.asymptotic.ctate_bands.ctate_confidence_band`
    (which bands the *mean* ``tau_d(., x)``). Putting the two side by side at a fixed
    ``x`` makes the confidence-vs-prediction distinction concrete: the confidence band
    shrinks at ``1/sqrt(n)`` onto ``tau``; this prediction band keeps a positive
    aleatoric width and covers an individual ``delta`` drawn at ``x``.

    Args:
        model: a fitted :class:`AdaptiveITTEConformal` or :class:`ConformalMetaLearner`.
        x: covariate value (1-D).
        d: homology dimension.
        alpha: 1 - target coverage.
    """
    band = model.band(np.atleast_2d(np.asarray(x, dtype=float)), alpha, d)
    band.kind = "prediction"
    return band
