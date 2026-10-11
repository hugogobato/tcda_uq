"""Finite-grid split-conformal scores and honest quantiles.

A fixed center and strictly positive scale must be frozen on independent training
data before calibration. The safest default is s=1. Learning a symmetric scale
from calibration outcomes does not put the new score under the same frozen rule
and is not supported. `modulation` is a numerical training helper, whose caller
must supply independent residuals. These functions calculate grid coverage;
continuum guarantees require additional extension bounds.
"""

from __future__ import annotations

import numpy as np

from ...metrics import Band
from ..asymptotic._validation import validate_alpha

_MODULATIONS = ("constant", "pointwise-sd", "lipschitz")


def residual_curves(phi_a, mu_hat_a):
    """Residual curves ``r_i(t) = phi_{i}(t) - mu_hat(t, X_i)``: ``[n, res]``.

    ``phi_a`` and ``mu_hat_a`` are both ``[n, resolution]`` (a single homology
    dim, a single arm).
    """
    return np.asarray(phi_a, dtype=float) - np.asarray(mu_hat_a, dtype=float)


def modulation(residuals, kind: str = "constant", *, floor: float = 0.0, eps: float = 1e-8):
    """Modulation function ``s(t)`` from residual curves ``[n, resolution]``.

    Args:
        residuals: residual curves used to shape the band width.
        kind: ``"constant"``, ``"pointwise-sd"`` or ``"lipschitz"`` (see module docstring).
        floor: lower bound on ``s(t)`` (the Lipschitz-aware floor; also applied
            when ``kind == "lipschitz"`` -- there it defaults to a fraction of
            the median sd if left at 0).
        eps: hard numerical floor so scores never divide by ~0.

    Returns:
        ``s(t)`` array ``[resolution]``, strictly positive.
    """
    R = np.asarray(residuals, dtype=float)
    if R.ndim != 2 or R.shape[1] < 1 or not np.isfinite(R).all():
        raise ValueError("residuals must be finite [n, resolution] curves")
    if not np.isfinite(floor) or floor < 0 or not np.isfinite(eps) or eps <= 0:
        raise ValueError("floor must be nonnegative and eps strictly positive")
    kind = kind.lower()
    if kind != "constant" and R.shape[0] == 0:
        raise ValueError("nonconstant modulation needs nonempty independent training residuals")
    if kind == "constant":
        s = np.ones(R.shape[-1])
    elif kind in ("pointwise-sd", "lipschitz"):
        s = R.std(axis=0)
        if kind == "lipschitz" and floor <= 0.0:
            # default Lipschitz floor: keep s(t) from collapsing in flat regions
            floor = 0.1 * float(np.median(s[s > 0])) if np.any(s > 0) else eps
    else:
        raise ValueError(f"modulation kind must be one of {_MODULATIONS}, got {kind!r}")
    return np.maximum(np.maximum(s, floor), eps)


def sup_norm_score(residuals, s):
    """Sup-norm nonconformity scores ``S_i = max_t |r_i(t)| / s(t)``: ``[n]``.

    ``residuals`` is ``[n, resolution]`` (or ``[resolution]`` for a single
    curve); ``s`` is the ``[resolution]`` modulation.
    """
    R = np.atleast_2d(np.asarray(residuals, dtype=float))
    s = np.asarray(s, dtype=float)
    if R.ndim != 2 or not np.isfinite(R).all() or s.shape != (R.shape[1],) or not np.isfinite(s).all() or np.any(s <= 0):
        raise ValueError("residuals must be finite and scale a finite positive [resolution] array")
    out = np.max(np.abs(R) / s, axis=-1)
    return float(out[0]) if out.shape[0] == 1 else out


def split_conformal_radius(scores, alpha: float):
    """Unweighted split-conformal radius ``k_{1-alpha}`` from scalar scores.

    The ``ceil((n+1)(1-alpha))`` smallest score (finite-sample conformal
    quantile). Returns ``+inf`` when ``ceil((n+1)(1-alpha)) > n`` (too few
    calibration points for the requested level -> the honest band is the whole
    line). This is the special case of :func:`~.weighted_cp.weighted_conformal_radius`
    with equal weights.
    """
    validate_alpha(alpha)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("scores must be a finite nonnegative [n] array")
    n = scores.shape[0]
    numerator, denominator = float(alpha).as_integer_ratio()
    k = ((n+1)*(denominator-numerator)+denominator-1)//denominator
    if k > n:
        return np.inf
    return float(np.partition(scores, k - 1)[k - 1])


def functional_cp_band(center, s, radius, tseq, *, alpha=None, kind: str = "prediction") -> Band:
    """Assemble a functional CP band ``center(t) +/- radius * s(t)``.

    Args:
        center: predicted curve ``mu_hat_a(t, x)`` ``[resolution]``.
        s: modulation ``[resolution]``.
        radius: conformal radius ``k`` (scalar; may be ``+inf``).
        tseq: grid ``[resolution]``.
        alpha: for bookkeeping ``level = 1 - alpha`` (optional).
        kind: ``"prediction"`` (default) -- these bands carry aleatoric spread.

    Returns:
        :class:`~tcda_uq.metrics.Band`.
    """
    center = np.asarray(center, dtype=float)
    s = np.asarray(s, dtype=float)
    if alpha is not None:
        validate_alpha(alpha)
    if center.ndim != 1 or not np.isfinite(center).all() or s.shape != center.shape or not np.isfinite(s).all() or np.any(s <= 0) or np.isnan(radius) or radius < 0:
        raise ValueError("center must be finite, scale finite positive, and radius nonnegative (possibly infinity)")
    half = radius * s
    return Band(
        tseq=tseq,
        lower=center - half,
        upper=center + half,
        center=center,
        level=None if alpha is None else 1.0 - alpha,
        kind=kind,
    )


def grid_discretization_slack(scores_grid, s, lipschitz_const: float = 1.0):
    """Grid slack for a residual with a supplied Lipschitz bound.

    The bound is L*delta/(2*min(s)). A continuum guarantee requires a strictly
    positive scale on the entire interval and a verified residual Lipschitz
    modulus. Nonconstant scales need a valid extension bound; values at nodes
    alone do not certify the score between nodes. This helper does not establish
    Lipschitz continuity of a fitted regression.
    """
    g = np.asarray(scores_grid, dtype=float)
    delta = float(g) if g.ndim == 0 else float(np.max(np.diff(g)))
    scale = np.asarray(s, dtype=float)
    if not np.isfinite(scale).all() or np.any(scale <= 0) or not np.isfinite(delta) or delta < 0 or not np.isfinite(lipschitz_const) or lipschitz_const < 0:
        raise ValueError("positive finite scale and nonnegative finite spacing/modulus required")
    return lipschitz_const * delta / 2.0 / float(np.min(scale))
