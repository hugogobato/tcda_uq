"""Native interval-wise testing p-values on a prespecified finite grid.

For valid interval-null p-values, adjusted p-values are the maximum over all
tested intervals containing a point. This controls any false rejection within a
true-null interval, conditional on validity of its interval test. The Gaussian
AIPW bootstrap implemented here requires its own process assumptions. It is not
a permutation test and does not yield a confidence band without inversion.
The legacy envelope helpers are deprecated numerical heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...metrics import Band
from ._validation import validate_alpha, validate_bootstrap, validate_process, pointwise_scale
import warnings
from .multiplier_bootstrap import _draw_multipliers


def _interval_means(square_cumsum):
    """Given prefix sums ``P`` of a per-point array over ``[..., res]``, return the
    per-interval means ``mean_{s in [j,k]} x[s]`` as a dict-free upper-triangular
    representation.

    ``square_cumsum`` has shape ``[..., res + 1]`` (with a leading zero). Returns
    ``means`` of shape ``[..., n_intervals]`` and index arrays ``j, k`` (0-based,
    inclusive) of length ``n_intervals``, ordered by ``(j, k)``.
    """
    res = square_cumsum.shape[-1] - 1
    j_idx, k_idx = np.triu_indices(res)               # all j <= k
    length = (k_idx - j_idx + 1).astype(float)        # |I| in grid points
    total = square_cumsum[..., k_idx + 1] - square_cumsum[..., j_idx]
    return total / length, j_idx, k_idx


@dataclass
class IWTResult:
    """Output of :func:`iwt_pvalues`."""

    tseq: np.ndarray
    p_unadjusted: np.ndarray   # [res]  point-wise error control (eq. 6)
    p_adjusted: np.ndarray     # [res]  interval-wise error control (eq. 7)
    alpha: float
    status: str = "ok"

    def significant(self, adjusted: bool = True):
        """Boolean mask of grid points where the null is rejected at ``alpha``."""
        p = self.p_adjusted if adjusted else self.p_unadjusted
        return p <= self.alpha


def iwt_pvalues(
    influence,
    tseq,
    estimate,
    *,
    ref=0.0,
    alpha: float = 0.05,
    n_boot: int = 2000,
    multiplier: str = "gaussian",
    variance=None,
    rng=None,
) -> IWTResult:
    """Interval-wise unadjusted + adjusted p-value functions for ``H0: psi_d = ref``.

    Args:
        influence: centered EIF process ``[n, res]`` (``cross_fit(...).influence()[d]``).
        tseq: grid ``[res]``.
        estimate: ``psi_hat_d(t)`` ``[res]``.
        ref: null curve (scalar broadcast or ``[res]``); ``0`` = no-effect null.
        alpha: error level for the significance masks.
        n_boot, multiplier, variance, rng: parametric-bootstrap controls
            (shared with the multiplier bootstrap).

    Returns:
        :class:`IWTResult`.
    """
    validate_alpha(alpha)
    validate_bootstrap(n_boot, multiplier)
    rng = np.random.default_rng(rng)
    phi, estimate = validate_process(influence, estimate, tseq)
    n, res = phi.shape
    ref = np.broadcast_to(np.asarray(ref, dtype=float), (res,))
    if not np.isfinite(ref).all():
        raise ValueError("ref must be finite")
    sd = pointwise_scale(phi, variance)
    if np.any(sd == 0):
        # No variance floor: abstain from rejection on the requested grid.
        return IWTResult(np.asarray(tseq, float), np.ones(res), np.ones(res),
                         alpha, status="zero_variance_no_rejection")

    # observed studentised deviation and its square
    d2 = (np.sqrt(n) * (estimate - ref) / sd) ** 2                 # [res]
    d2_cum = np.concatenate([[0.0], np.cumsum(d2)])
    T_obs, j_idx, k_idx = _interval_means(d2_cum)
    exceedances = np.zeros(T_obs.size, dtype=np.int64)
    # Bound each interval-statistic matrix to about 32 MB. Draws still follow
    # the same independent multiplier law and interval family.
    batch = max(1, min(128, 4_000_000 // max(1, T_obs.size)))
    for start in range(0, n_boot, batch):
        size = min(batch, n_boot-start)
        xi = _draw_multipliers(multiplier, (size, n), rng)
        W2 = ((xi @ phi) / np.sqrt(n) / sd) ** 2
        prefix = np.concatenate([np.zeros((size, 1)), np.cumsum(W2, axis=1)], axis=1)
        T_boot, _, _ = _interval_means(prefix)
        exceedances += np.count_nonzero(T_boot >= T_obs[None, :], axis=0)
    p_I = exceedances / n_boot

    # unadjusted p(t) = p of the singleton interval {t};
    # adjusted p_tilde(t) = max over intervals containing t.
    p_unadj = np.empty(res)
    p_adj = np.full(res, -np.inf)
    singleton = j_idx == k_idx
    p_unadj[j_idx[singleton]] = p_I[singleton]
    # scatter-max: p_tilde(t) = max over every interval [j,k] that contains t
    # (np.maximum is order-independent; each interval lifts its whole segment).
    for m in range(p_I.shape[0]):
        seg = p_adj[j_idx[m] : k_idx[m] + 1]
        np.maximum(seg, p_I[m], out=seg)
    return IWTResult(
        tseq=np.asarray(tseq, dtype=float),
        p_unadjusted=p_unadj,
        p_adjusted=p_adj,
        alpha=alpha,
    )


def pini_vantini_band(
    influence,
    tseq,
    estimate,
    *,
    alpha: float = 0.05,
    n_boot: int = 2000,
    multiplier: str = "gaussian",
    variance=None,
    rng=None,
) -> Band:
    """Deprecated heuristic interval, without an asserted coverage level.

    This critical-function envelope is not explicit inversion of interval-wise
    tests. It remains callable for reproducing legacy numerical analyses, and
    returns ``Band(kind=None, level=None)``. Use `iwt_pvalues` for inference.
    """
    warnings.warn("pini_vantini_band is a deprecated heuristic, without confidence-band coverage or IWT inversion; use iwt_pvalues/tate_test", FutureWarning, stacklevel=2)
    validate_alpha(alpha)
    validate_bootstrap(n_boot, multiplier)
    rng = np.random.default_rng(rng)
    phi, estimate = validate_process(influence, estimate, tseq)
    n, res = phi.shape
    sd = pointwise_scale(phi, variance)
    if np.any(sd == 0):
        return Band(tseq=tseq, lower=np.full(res, -np.inf), upper=np.full(res, np.inf),
                    center=estimate, level=None, kind=None)

    xi = _draw_multipliers(multiplier, (n_boot, n), rng)
    W2 = ((xi @ phi) / np.sqrt(n) / sd) ** 2                     # [B, res]
    W2_cum = np.concatenate([np.zeros((n_boot, 1)), np.cumsum(W2, axis=1)], axis=1)
    T_boot, j_idx, k_idx = _interval_means(W2_cum)             # [B, n_int]
    c_I = np.quantile(T_boot, 1.0 - alpha, axis=0)             # [n_int]

    # k_tilde(t)^2 = max over intervals containing t of c_I  (scatter-max)
    k2 = np.zeros(res)
    for m in range(c_I.shape[0]):
        seg = k2[j_idx[m] : k_idx[m] + 1]
        np.maximum(seg, c_I[m], out=seg)
    half = np.sqrt(k2) * sd / np.sqrt(n)
    return Band(
        tseq=tseq,
        lower=estimate - half,
        upper=estimate + half,
        center=estimate,
        level=None,
        kind=None,
    )


def pini_vantini_bands(cross_fit_result, *, alpha: float = 0.05, **kwargs):
    """Deprecated heuristic intervals per degree, without coverage guarantees."""
    influence = cross_fit_result.influence()
    tseq = cross_fit_result.tseq
    return [
        pini_vantini_band(
            influence[d], tseq, cross_fit_result.aipw[d], alpha=alpha, **kwargs
        )
        for d in range(len(influence))
    ]
