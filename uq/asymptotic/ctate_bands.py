"""Gaussian multiplier bands at a prespecified covariate value.

The learner must provide valid smoother weights and residuals. Coverage requires
negligible smoothing, projection, and complete first-stage bias, a suitable
process CLT and conditional multiplier approximation, and the limiting supremum
quantile condition. A correct fixed-dimensional OLS model is one sufficient case.
There is no general guarantee for an arbitrary second-stage regression, nor uniform
coverage over x. The default is unstudentized and uses no numerical variance floor.
"""

from __future__ import annotations

import numpy as np

from ...metrics import Band
from .multiplier_bootstrap import _draw_multipliers
from ._validation import validate_alpha, validate_bootstrap, validate_process


def ctate_confidence_band(
    learner,
    x,
    d: int = 0,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    multiplier: str = "gaussian",
    standardize: bool = False,
    rng=None,
) -> Band:
    """Simultaneous-in-t, pointwise-in-x confidence band for tau_d(., x).

    Args:
        learner: a fitted :class:`~tcda_uq.estimators.CTATEDRLearner`.
        x: the covariate value at which to band tau_d(., x) (1-D, length ``d_cov``).
        d: homology dimension.
        alpha: 1 - target simultaneous-in-t coverage.
        n_boot: number of bootstrap draws.
        multiplier: ``"gaussian"`` (default), ``"rademacher"`` or ``"mammen"``.
            Non-Gaussian choices require separate bootstrap justification.
        standardize: studentize by the pointwise sd (variance-adaptive width).
            ``False`` (default) gives the unstudentized constant-width band.
            ``True`` returns the whole real line if any empirical variance is zero.
        rng: seed or ``np.random.Generator``.

    Returns:
        :class:`~tcda_uq.metrics.Band` with ``kind="confidence"``, centered at
        ``tau_hat_d(., x)``.
    """
    validate_alpha(alpha)
    validate_bootstrap(n_boot, multiplier)
    rng = np.random.default_rng(rng)
    a = np.asarray(learner.weights(x, d=d), dtype=float)                 # [n]  smoother weights
    R = learner.residuals(d)                    # [n, res]  second-stage residuals
    center = learner.predict_dim(np.atleast_2d(x), d)[0]   # [res]

    R = np.asarray(R, dtype=float)
    validate_process(R, center, learner.tseq)
    if getattr(learner, "design_singular_", False):
        center = np.asarray(center, dtype=float)
        return Band(tseq=learner.tseq, lower=np.full_like(center, -np.inf),
                    upper=np.full_like(center, np.inf), center=center,
                    level=1-alpha, kind="confidence")
    if a.shape != (R.shape[0],) or not np.isfinite(a).all():
        raise ValueError("smoother weights must be a finite [n] array")
    contrib = a[:, None] * R                    # [n, res]  a_i r_i(t)
    var = np.einsum("it,it->t", contrib, contrib)          # sum_i (a_i r_i(t))^2
    sd = np.sqrt(var)
    if standardize and np.any(sd == 0):
        return Band(tseq=learner.tseq, lower=np.full_like(center, -np.inf),
                    upper=np.full_like(center, np.inf), center=center,
                    level=1.0 - alpha, kind="confidence")
    scale = sd if standardize else np.ones_like(sd)

    xi = _draw_multipliers(multiplier, (n_boot, a.shape[0]), rng)   # [B, n]
    boot = (xi @ contrib) / scale               # [B, res]
    sup = np.max(np.abs(boot), axis=1)          # [B]
    c = float(np.quantile(sup, 1.0 - alpha))

    half = c * scale
    return Band(
        tseq=learner.tseq,
        lower=center - half,
        upper=center + half,
        center=center,
        level=1.0 - alpha,
        kind="confidence",
    )


def ctate_confidence_bands(learner, x, *, alpha: float = 0.05, **kwargs):
    """Convenience: one CTATE confidence band per homology dim at a fixed ``x``.

    Extra keyword args are forwarded to :func:`ctate_confidence_band`.
    """
    return [
        ctate_confidence_band(learner, x, d, alpha=alpha, **kwargs)
        for d in range(learner.n_hom_dim)
    ]


def ctate_pointwise_sd(learner, x, d: int = 0):
    """Analytic HC0 pointwise sd of ``tau_hat_d(., x)``: ``sqrt(sum_i a_i^2 r_i(t)^2)``.

    The studentizing scale of :func:`ctate_confidence_band`; also usable for a
    Gaussian pointwise (non-simultaneous) interval ``tau_hat +/- z_{1-alpha/2} sd``.
    """
    a = learner.weights(x, d=d)
    R = learner.residuals(d)
    contrib = a[:, None] * R
    return np.sqrt(np.einsum("it,it->t", contrib, contrib))
