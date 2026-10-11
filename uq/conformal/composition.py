"""Armwise prediction intervals combined by Bonferroni and Minkowski difference.

Each arm uses alpha/2; no independence of the two potential-outcome coverage
events is assumed. True arm-to-target ratios give marginal coverage at least
1-alpha. Estimated ratios require subtracting both armwise TV discrepancies for
the intended target. Independent centers and scales affect efficiency, without
making estimated weights doubly robust. Individual widths need not stay positive
in degenerate models. Computations operate on the fixed grid.
"""

from __future__ import annotations

import numpy as np

from ...estimators.nuisance import (
    NuisanceFit,
    fit_functional_regression,
    fit_propensity,
    validate_sample,
)
from ...metrics import Band
from .weighted_cp import ConformalArm


def fit_split_nuisances(
    train_sample,
    tseq,
    *,
    n_basis: int = 5,
    propensity_estimator=None,
    propensity_feature_fn=None,
) -> NuisanceFit:
    """Fit outcome regressions (both arms) + propensity on a training split.

    Fits on data independent of calibration. Weighted calibration and target
    residuals need not be exchangeable; actual ratios handle covariate shift. Returns
    a :class:`NuisanceFit` (its ``predict_mu`` / ``prop_model`` feed
    :class:`~tcda_uq.uq.conformal.weighted_cp.ConformalArm`).
    """
    phi, A, X, tseq = validate_sample(train_sample, tseq)
    if set(np.unique(A)) != {0,1}:
        raise ValueError("training split requires both treatment arms")
    reg = fit_functional_regression(train_sample, tseq, n_basis=n_basis)
    feat = propensity_feature_fn if propensity_feature_fn is not None else (lambda z: z)
    prop = fit_propensity(feat(np.asarray(X)), A, propensity_estimator)
    return NuisanceFit(mu_reg=reg, prop_model=prop, tseq=np.asarray(tseq, dtype=float),
                       propensity_feature_fn=propensity_feature_fn)


class ITTEConformal:
    """Composed conformal predictor of the ITTE ``delta_{i,d}`` (Layers 1-3).

    Usage::

        model = ITTEConformal.fit(sample, tseq, propensity_feature_fn=feat)
        band  = model.band(x_new, alpha=0.1, d=0)          # Band for delta_0(., x_new)
        lo, hi, ctr = model.band_bounds(X_test, alpha=0.1, d=0)   # batched

    ``sample`` is the observed triplet ``(phi, A, X)``. :meth:`fit` splits it into
    a nuisance-training part and a calibration part, fits ``mu_hat_{0,1}`` and
    ``pi_hat`` on the former, and calibrates one
    :class:`~tcda_uq.uq.conformal.weighted_cp.ConformalArm` per arm per homology
    dim on the latter.
    """

    def __init__(self, arms0, arms1, tseq):
        self.arms0 = arms0                       # list over hom dim
        self.arms1 = arms1
        self.tseq = np.asarray(tseq, dtype=float)
        self.n_hom_dim = len(arms0)

    # ---------------------------------------------------------------------- fit
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
        weight_fn=None,
        random_state=0,
        nuisance: NuisanceFit | None = None,
        calib_index=None,
        train_index=None,
        target: str = "full",
        propensity_fn=None,
        weights_known: bool = False,
        modulation_residuals=None,
    ):
        """Fit the composed ITTE conformal predictor.

        Args:
            sample: observed ``(phi, A, X)``.
            calib_frac: fraction of units used for calibration (rest fit nuisances).
            n_basis: Fourier basis size of the outcome regression.
            propensity_estimator: sklearn classifier (default random forest).
            propensity_feature_fn: optional ``X -> features`` for the propensity model.
            modulation_kind / floor: modulation ``s(t)`` (see :mod:`.functional_cp`).
            weighted: apply propensity weights (Layer 2). ``False`` -> plain split-CP
                (exact under a randomized design / matched calibration).
            weight_fn: Layer-2 weight map ``(pi_hat, arm) -> weights`` (default naive
                ``1/pi``). Pass a stabilizer from
                :mod:`~tcda_uq.uq.conformal.stabilized_weights` (Phase 6.5) to bound
                the weights under weak overlap -- the coverage lower bound for that target requires true ratios; learned
                ratios require TV qualification. Bounded weights do not guarantee finite bands.
            random_state: seeds the train/calibration split.
            nuisance: optionally supply a pre-fit :class:`NuisanceFit` (fit on data
                disjoint from the calibration units) instead of splitting here.
            target: full, overlap, or matching covariate population.
            propensity_fn: fixed probability function on raw X (e.g. true propensity).
            weights_known: caller declaration that ratios are actual target ratios.
            train_index: independent units to fit nonconstant scales with external nuisances.
            modulation_residuals: alternatively independent residual arrays indexed [d][a].
            calib_index: optional explicit boolean/int index of calibration units
                (requires ``nuisance``); overrides the internal random split.
        """
        phi, A, X, tseq = validate_sample(sample, tseq)
        if not np.isfinite(calib_frac) or not 0 < calib_frac < 1:
            raise ValueError("calib_frac must be strictly between 0 and 1")
        n = phi.shape[0]
        n_hom_dim = phi.shape[-2]
        if n < 2 and nuisance is None:
            raise ValueError("internal training/calibration split requires at least two observations")

        if nuisance is None:
            rng = np.random.default_rng(random_state)
            perm = rng.permutation(n)
            n_cal = min(max(int(round(calib_frac * n)), 1), n-1)
            cal_idx = np.sort(perm[:n_cal])
            tr_idx = np.sort(perm[n_cal:])
            train = (phi[tr_idx], A[tr_idx], X[tr_idx])
            nuisance = fit_split_nuisances(
                train, tseq, n_basis=n_basis,
                propensity_estimator=propensity_estimator,
                propensity_feature_fn=propensity_feature_fn,
            )
        else:
            if calib_index is None:
                raise ValueError("supplying `nuisance` also requires `calib_index`")
            cal_idx = _indices(calib_index, n, "calib_index")
            tr_idx = None if train_index is None else _indices(train_index, n, "train_index")
            if tr_idx is not None and np.intersect1d(cal_idx, tr_idx).size:
                raise ValueError("training and calibration indices must be disjoint")

        calib = (phi[cal_idx], A[cal_idx], X[cal_idx])

        # Fit every nonconstant scale on independent training before calibration.
        from .functional_cp import residual_curves
        mod = {}
        for d in range(n_hom_dim):
            for a in (0, 1):
                if modulation_kind == "constant":
                    mod[d, a] = None
                elif modulation_residuals is not None:
                    mod[d, a] = modulation_residuals[d][a]
                elif tr_idx is not None:
                    arm_train = tr_idx[A[tr_idx] == a]
                    mu = nuisance.predict_mu(X[arm_train])[d][a]
                    mod[d, a] = residual_curves(phi[arm_train, d, :], mu)
                else:
                    raise ValueError("nonconstant modulation with external nuisance requires independent train_index or modulation_residuals[d][a]")

        arm_kw = dict(
            modulation_kind=modulation_kind,
            floor=floor,
            weighted=weighted,
            propensity_feature_fn=propensity_feature_fn,
            weight_fn=weight_fn, target=target, propensity_fn=propensity_fn,
            weights_known=weights_known,
        )
        arms0 = [ConformalArm(nuisance, calib, d, 0, modulation_residuals=mod[d, 0], **arm_kw) for d in range(n_hom_dim)]
        arms1 = [ConformalArm(nuisance, calib, d, 1, modulation_residuals=mod[d, 1], **arm_kw) for d in range(n_hom_dim)]
        model = cls(arms0, arms1, tseq)
        model.nuisance_ = nuisance
        model.calib_index_ = cal_idx
        model.train_index_ = tr_idx
        model.target_ = arms0[0].target_
        model.weights_known_ = arms0[0].weights_known_
        model.coverage_scope_ = arms0[0].coverage_scope_
        return model

    # ------------------------------------------------------------------- bands
    def band_bounds(self, X_new, alpha: float, d: int = 0):
        """Batched ITTE band bounds for homology dim ``d``.

        Returns ``(lower, upper, center)`` each ``[m, resolution]``. Each arm is
        banded at ``alpha/2``; the composition is centered at the plug-in CTATE
        ``mu_hat_1 - mu_hat_0`` with half-width ``h1 + h0``.
        """
        from ..asymptotic._validation import validate_alpha
        validate_alpha(alpha)
        l1, u1, c1 = self.arms1[d].band_bounds(X_new, alpha / 2.0)
        l0, u0, c0 = self.arms0[d].band_bounds(X_new, alpha / 2.0)
        center = c1 - c0
        half = 0.5 * (u1 - l1) + 0.5 * (u0 - l0)
        return center - half, center + half, center

    def band(self, x_new, alpha: float, d: int = 0) -> Band:
        """ITTE prediction :class:`Band` for ``delta_d(., x)`` at a single ``x``."""
        lower, upper, center = self.band_bounds(x_new, alpha, d)
        if lower.shape[0] != 1:
            raise ValueError("band expects one covariate row; use band_bounds for batches")
        return Band(
            tseq=self.tseq,
            lower=lower[0],
            upper=upper[0],
            center=center[0],
            level=1.0 - alpha,
            kind="prediction",
        )

    def bands(self, x_new, alpha: float):
        """One ITTE band per homology dim at a single ``x``."""
        return [self.band(x_new, alpha, d) for d in range(self.n_hom_dim)]


def _indices(index, n, name):
    index = np.asarray(index)
    if index.size == 0 and index.ndim == 1:
        return np.empty(0, dtype=int)
    if index.dtype == bool:
        if index.shape != (n,):
            raise ValueError(f"{name} boolean mask must have length n")
        return np.flatnonzero(index)
    if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer) or np.any((index < 0) | (index >= n)) or np.unique(index).size != index.size:
        raise ValueError(f"{name} must contain unique valid integer indices")
    return index
