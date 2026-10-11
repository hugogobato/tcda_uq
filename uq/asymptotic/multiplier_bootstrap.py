"""Finite-grid Gaussian multiplier inference for a mean effect.

The default uses the unstudentized absolute supremum. Asymptotic coverage
requires uniform asymptotic linearity, oracle and conditional bootstrap process
convergence, and continuity and local strict increase of the limiting supremum
CDF at its quantile. Studentization additionally requires nonvanishing variance;
a zero empirical variance returns a whole real-line band. Finite-grid calculation
does not by itself establish coverage between grid nodes or verify nuisance rates.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...metrics import Band
from ._validation import validate_alpha, validate_bootstrap, validate_process, pointwise_scale

_MULTIPLIERS = ("gaussian", "rademacher", "mammen")


def _draw_multipliers(kind, shape, rng):
    """iid mean-0 variance-1 multipliers ``xi`` of the requested family."""
    kind = kind.lower()
    if kind == "gaussian":
        return rng.standard_normal(shape)
    if kind == "rademacher":
        return rng.choice(np.array([-1.0, 1.0]), size=shape)
    if kind == "mammen":  # skewness-correcting two-point law (golden-ratio)
        s5 = np.sqrt(5.0)
        a, b = (1 - s5) / 2, (1 + s5) / 2
        p = (s5 + 1) / (2 * s5)  # P(xi = a)
        u = rng.random(shape)
        return np.where(u < p, a, b)
    raise ValueError(f"multiplier must be one of {_MULTIPLIERS}, got {kind!r}")


def multiplier_bootstrap_band(
    influence,
    tseq,
    estimate,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    multiplier: str = "gaussian",
    standardize: bool = False,
    variance=None,
    rng=None,
) -> Band:
    """Simultaneous 1-alpha multiplier-bootstrap band for one homology dim.

    Args:
        influence: centered EIF process ``[n, resolution]`` -- one element of
            ``cross_fit(...).influence()``. Its column mean should be ~0.
        tseq: silhouette grid ``[resolution]``.
        estimate: point estimate ``psi_hat_d(t)`` ``[resolution]`` (band center),
            i.e. ``cross_fit(...).aipw[d]``.
        alpha: 1 - target simultaneous coverage.
        n_boot: number of bootstrap draws.
        multiplier: ``"gaussian"`` (default), ``"rademacher"`` or ``"mammen"``.
            Non-Gaussian choices require separate bootstrap justification.
        standardize: studentize by the pointwise sd (variance-adaptive width).
            ``False`` (default) uses the raw absolute supremum and constant width.
            With ``True``, any zero pointwise variance returns the whole real line;
            no numerical variance floor is substituted.
        variance: optional pointwise variance ``sigma_hat^2(t)`` ``[resolution]``;
            if ``None`` it is estimated from ``influence``. Supplying a shared
            variance (e.g. from :func:`~.covariance.eif_pointwise_variance`) keeps
            the studentization identical across bands in a comparison.
        rng: seed or ``np.random.Generator``.

    Returns:
        :class:`~tcda_uq.metrics.Band` with ``kind="confidence"``.
    """
    validate_alpha(alpha)
    validate_bootstrap(n_boot, multiplier)
    rng = np.random.default_rng(rng)
    phi, estimate = validate_process(influence, estimate, tseq)
    n, res = phi.shape
    empirical_sd = pointwise_scale(phi, variance)
    sd = empirical_sd if standardize else np.ones(res)
    if standardize and np.any(sd == 0):
        return Band(tseq=tseq, lower=np.full(res, -np.inf), upper=np.full(res, np.inf),
                    center=estimate, level=1.0 - alpha, kind="confidence")

    # bootstrap sup-statistic:  sup_t | n^{-1/2} sum_i xi_i phi_i(t) | / s(t)
    xi = _draw_multipliers(multiplier, (n_boot, n), rng)      # [B, n]
    boot = (xi @ phi) / np.sqrt(n)                            # [B, res]
    boot /= sd                                                # studentize
    sup = np.max(np.abs(boot), axis=1)                        # [B]
    c = float(np.quantile(sup, 1.0 - alpha))

    half = c * sd / np.sqrt(n)
    return Band(
        tseq=tseq,
        lower=estimate - half,
        upper=estimate + half,
        center=estimate,
        level=1.0 - alpha,
        kind="confidence",
    )


def multiplier_bootstrap_bands(cross_fit_result, *, alpha: float = 0.05, **kwargs):
    """Convenience: one band per homology dim from a :class:`CrossFitResult`.

    Uses ``cross_fit_result.influence()`` and ``.aipw`` as the influence process
    and center. Extra keyword args are forwarded to :func:`multiplier_bootstrap_band`.
    """
    influence = cross_fit_result.influence()
    tseq = cross_fit_result.tseq
    return [
        multiplier_bootstrap_band(
            influence[d], tseq, cross_fit_result.aipw[d], alpha=alpha, **kwargs
        )
        for d in range(len(influence))
    ]


def topological_effect_test(
    influence,
    estimate,
    *,
    n_boot: int = 2000,
    multiplier: str = "gaussian",
    standardize: bool = False,
    variance=None,
    rng=None,
):
    """Gaussian multiplier test of ``H0: psi_d == 0`` (no topological effect).

    Statistic ``T_n = sqrt(n) * sup_t |psi_hat_d(t)| / s(t)`` compared to the
    multiplier-bootstrap null law of ``sup_t |G_hat_n,d(t)| / s(t)``. With
    ``standardize=False`` this is the literal Corollary 5.4 statistic
    ``T_n = sqrt(n) ||psi_hat_AIPW,d||_inf``.

    Returns:
        dict with ``statistic`` ``T_n``, bootstrap ``pvalue`` = mean(sup_boot >= T_n),
        and the critical values ``crit`` at the 90/95/99% levels.
    """
    validate_bootstrap(n_boot, multiplier)
    rng = np.random.default_rng(rng)
    phi, estimate = validate_process(influence, estimate)
    n, res = phi.shape
    empirical_sd = pointwise_scale(phi, variance)
    sd = empirical_sd if standardize else np.ones(res)
    if standardize and np.any(sd == 0):
        return {"statistic": np.nan, "pvalue": 1.0, "crit": {},
                "status": "zero_variance_no_rejection"}

    T_n = float(np.max(np.abs(estimate) / sd) * np.sqrt(n))
    xi = _draw_multipliers(multiplier, (n_boot, n), rng)
    boot = (xi @ phi) / np.sqrt(n) / sd
    sup = np.max(np.abs(boot), axis=1)
    pvalue = float(np.mean(sup >= T_n))
    crit = {lvl: float(np.quantile(sup, lvl)) for lvl in (0.90, 0.95, 0.99)}
    return {"statistic": T_n, "pvalue": pvalue, "crit": crit}
