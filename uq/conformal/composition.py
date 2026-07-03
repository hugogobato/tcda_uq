"""Layer 3 -- arm composition into an ITTE prediction band (Phase 4.3).

The individual topological treatment effect ``delta_{i,d}(t) = phi^1_{i,d}(t) -
phi^0_{i,d}(t)`` is a *difference* of two counterfactual silhouettes. Build a
weighted functional-CP band for each arm at level ``alpha/2`` and combine by
**interval arithmetic + Bonferroni** into a simultaneous ``alpha`` band for
``delta``:

    phi^1(t) in [c1(t) - h1(t),  c1(t) + h1(t)]   (treated arm, level 1 - alpha/2)
    phi^0(t) in [c0(t) - h0(t),  c0(t) + h0(t)]   (control arm, level 1 - alpha/2)
    ==>  delta(t) in [ (c1-c0)(t) - (h1+h0)(t),  (c1-c0)(t) + (h1+h0)(t) ].

So the ITTE band is centered at the plug-in CTATE ``c1 - c0 = mu_hat_1 -
mu_hat_0`` with half-width ``h1 + h0``. This is the functional lift of Lemma A.1
of Schroeder et al. (2025).

**Coverage (Phase 4.4 composition lemma).** For a new unit drawn from the target
population, ``P(phi^1 in band_1) >= 1 - alpha/2`` and ``P(phi^0 in band_0) >=
1 - alpha/2`` (Layer 2, each arm marginal-over-``X``), so by the union bound
``P(both) >= 1 - alpha``; on that event ``delta = phi^1 - phi^0`` lies in the
interval-arithmetic band. Hence

    P( delta_{n+1,d}(.) in C(X_{n+1}, .) )  >=  1 - alpha,   simultaneously over t,

finite-sample, distribution-free (marginal over ``X``), doubly robust. **Cost:**
conservative (Bonferroni across arms; the sup-norm already absorbs the union over
``t``). This is a **prediction** band: it carries the aleatoric spread of an
individual draw and its width does **not** vanish as ``n -> infinity`` -- the
essential contrast with the TATE/CTATE *confidence* bands (Phases 2, 3).

**What cannot be claimed:** finite-sample coverage of the *average* ``psi_d`` or
conditional-mean ``tau_d`` -- CP covers a draw, not an expectation. For those,
use the asymptotic bands.
"""

from __future__ import annotations

import numpy as np

from ...estimators.nuisance import (
    NuisanceFit,
    fit_functional_regression,
    fit_propensity,
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

    Split-CP fits the nuisances on data **disjoint** from the calibration set, so
    the calibration residuals are exchangeable with a fresh test residual. Returns
    a :class:`NuisanceFit` (its ``predict_mu`` / ``prop_model`` feed
    :class:`~tcda_uq.uq.conformal.weighted_cp.ConformalArm`).
    """
    phi, A, X = train_sample
    reg = fit_functional_regression(train_sample, tseq, n_basis=n_basis)
    feat = propensity_feature_fn if propensity_feature_fn is not None else (lambda z: z)
    prop = fit_propensity(feat(np.asarray(X)), A, propensity_estimator)
    return NuisanceFit(mu_reg=reg, prop_model=prop, tseq=np.asarray(tseq, dtype=float))


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
        modulation_kind: str = "pointwise-sd",
        floor: float = 0.0,
        weighted: bool = True,
        weight_fn=None,
        random_state=0,
        nuisance: NuisanceFit | None = None,
        calib_index=None,
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
                the weights under weak overlap -- the coverage guarantee then holds
                for that weighter's target population.
            random_state: seeds the train/calibration split.
            nuisance: optionally supply a pre-fit :class:`NuisanceFit` (fit on data
                disjoint from the calibration units) instead of splitting here.
            calib_index: optional explicit boolean/int index of calibration units
                (requires ``nuisance``); overrides the internal random split.
        """
        phi, A, X = sample
        phi = np.asarray(phi, dtype=float)
        A = np.asarray(A)
        X = np.asarray(X, dtype=float)
        tseq = np.asarray(tseq, dtype=float)
        n = phi.shape[0]
        n_hom_dim = phi.shape[-2]

        if nuisance is None:
            rng = np.random.default_rng(random_state)
            perm = rng.permutation(n)
            n_cal = int(round(calib_frac * n))
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
            cal_idx = np.asarray(calib_index)

        calib = (phi[cal_idx], A[cal_idx], X[cal_idx])

        arm_kw = dict(
            modulation_kind=modulation_kind,
            floor=floor,
            weighted=weighted,
            propensity_feature_fn=propensity_feature_fn,
            weight_fn=weight_fn,
        )
        arms0 = [ConformalArm(nuisance, calib, d, 0, **arm_kw) for d in range(n_hom_dim)]
        arms1 = [ConformalArm(nuisance, calib, d, 1, **arm_kw) for d in range(n_hom_dim)]
        model = cls(arms0, arms1, tseq)
        model.nuisance_ = nuisance
        model.calib_index_ = cal_idx
        return model

    # ------------------------------------------------------------------- bands
    def band_bounds(self, X_new, alpha: float, d: int = 0):
        """Batched ITTE band bounds for homology dim ``d``.

        Returns ``(lower, upper, center)`` each ``[m, resolution]``. Each arm is
        banded at ``alpha/2``; the composition is centered at the plug-in CTATE
        ``mu_hat_1 - mu_hat_0`` with half-width ``h1 + h0``.
        """
        l1, u1, c1 = self.arms1[d].band_bounds(X_new, alpha / 2.0)
        l0, u0, c0 = self.arms0[d].band_bounds(X_new, alpha / 2.0)
        center = c1 - c0
        half = 0.5 * (u1 - l1) + 0.5 * (u0 - l0)
        return center - half, center + half, center

    def band(self, x_new, alpha: float, d: int = 0) -> Band:
        """ITTE prediction :class:`Band` for ``delta_d(., x)`` at a single ``x``."""
        lower, upper, center = self.band_bounds(x_new, alpha, d)
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
