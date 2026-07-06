"""Silhouette<->diagram stability bridge inside conformal prediction (Phase 6.1).

The nonconformity scores of the Phase-4 band are computed in **silhouette space**
(cheap, vectorized), but the intrinsic object is the **persistence diagram** ``D``. The
silhouette blurs the number of homological features, and in practice ``D`` itself is only
*estimated* (finite point cloud, chosen filtration), so the silhouette fed to the score
carries a diagram-estimation error. TATE Theorem 5.3 turns that error into an explicit,
computable bound -- and this module turns that theorem into a *load-bearing component of
the conformal analysis* (concept note section 5.1), which, to our knowledge, is a novel
use of a TDA stability bound inside CP.

**Theorem 5.3 (Kim & Lee 2026).** For power-weighted silhouettes ``phi, phi'`` of
diagrams ``D, D'`` with exponent ``r``, under boundedness (A6),

    || phi - phi' ||_inf  <=  ( 1 + 2 L r c^{r-1} ) * W_1(D, D'),

where (Lemma C.3 / Appendix C.5):

  * ``r``  is the power-weight exponent,
  * ``c = max{ b_x - a_x : x in D ∪ D' }`` is the **largest lifetime** across the two
    diagrams,
  * ``L`` bounds ``(b_p - a_p)^(1 - r)`` (the tent-height / weight ratio) over all points.

We call ``K(L, r, c) = 1 + 2 L r c^{r-1}`` the **bridge constant**. It converts a
Wasserstein-``W_1`` perturbation of the diagram into a sup-norm perturbation of the
silhouette, hence (dividing by ``min_t s(t)``) into a perturbation of the sup-norm
nonconformity score.

**Why this matters for coverage.** Suppose the diagrams used to build the scores are
estimated with ``W_1(D_hat, D) <= epsilon`` (a bound one gets from a bottleneck/Wasserstein
stability argument on the filtration, or from a resampling estimate). Then every
silhouette moves by at most ``K * epsilon`` in sup-norm, so every score moves by at most

    Delta = K * epsilon / min_t s(t).

Two operational consequences, both provided here:

  * :func:`certified_score_inflation` -- inflate the calibration scores by ``Delta`` before
    taking the conformal quantile. The resulting band is valid for the *true* silhouette
    even though it was calibrated on *estimated* diagrams (a diagram-robust band), at a
    controlled width cost (:func:`certified_width_inflation`).
  * :func:`coverage_certificate` -- if instead one keeps the ordinary band, its worst-case
    coverage loss from diagram error is bounded through the same ``Delta``. For scores
    computed from estimated diagrams, the vulnerable calibration scores are those just below
    the ordinary radius, in ``(k - Delta, k]``.

All quantities are analytic in ``(L, r, c, epsilon, s)`` -- no extra fitting -- so the
bridge is a cheap add-on that certifies the cheap silhouette-space band against the
expensive diagram-space geometry it stands in for.
"""

from __future__ import annotations

import numpy as np


def bridge_constant(L: float, r: float, c: float) -> float:
    """The Theorem-5.3 bridge constant ``K = 1 + 2 L r c^(r-1)``.

    Converts ``W_1(D, D')`` into an upper bound on ``|| phi - phi' ||_inf``. At ``r = 1``
    the ``c`` term drops (``K = 1 + 2L``), matching the theorem's special case.

    Args:
        L: bound on ``(b_p - a_p)^(1-r)`` over the diagrams' points (A6 constant).
        r: power-weight exponent (> 0).
        c: largest lifetime ``max (b - a)`` across the two diagrams (> 0).
    """
    if r <= 0:
        raise ValueError(f"r must be > 0, got {r!r}")
    if r == 1.0:
        return 1.0 + 2.0 * L * r            # c^0 = 1; kept explicit for clarity
    return 1.0 + 2.0 * L * r * c ** (r - 1.0)


def estimate_bridge_params(diagrams, r: float, *, eps: float = 1e-12):
    """Estimate ``(L, c)`` for :func:`bridge_constant` from a collection of diagrams.

    ``c = max lifetime`` and ``L = max (lifetime)^(1-r)`` over every finite point in every
    diagram (Lemma C.3: the constant depends only on the largest lifetime and, through
    ``L``, on the exponent). Infinite/essential pairs must already be removed (the
    silhouette pipeline drops them).

    Args:
        diagrams: iterable of ``(k_i, 2)`` birth-death arrays (mixed sizes allowed).
        r: power-weight exponent.
        eps: lifetimes below this are ignored (numerical zeros / diagonal points).

    Returns:
        ``(L, c)`` floats. Empty input -> ``(0.0, 0.0)`` (bridge constant then ``1``).
    """
    lifetimes = []
    for D in diagrams:
        D = np.asarray(D, dtype=float)
        if D.size == 0:
            continue
        lt = D[:, 1] - D[:, 0]
        lt = lt[np.isfinite(lt) & (lt > eps)]
        if lt.size:
            lifetimes.append(lt)
    if not lifetimes:
        return 0.0, 0.0
    lt = np.concatenate(lifetimes)
    c = float(lt.max())
    L = float(np.max(lt ** (1.0 - r)))
    return L, c


def score_perturbation_bound(K: float, epsilon: float, s) -> float:
    """Sup-norm score perturbation ``Delta = K * epsilon / min_t s(t)``.

    The most a nonconformity score ``max_t |r(t)| / s(t)`` can move when the underlying
    diagram moves by at most ``epsilon`` in ``W_1`` (silhouette moves by ``K * epsilon``
    in sup-norm, then divided by the smallest modulation value).

    Args:
        K: bridge constant (:func:`bridge_constant`).
        epsilon: ``W_1`` bound on the diagram-estimation error.
        s: modulation ``s(t)`` array (uses its minimum) or a scalar.
    """
    s = np.asarray(s, dtype=float)
    s_min = float(np.min(s)) if s.ndim else float(s)
    return float(K) * float(epsilon) / max(s_min, 1e-12)


def certified_score_inflation(scores, K: float, epsilon: float, s):
    """Add the certified slack ``Delta`` to every calibration score (diagram-robust band).

    Calibrating on ``scores + Delta`` (with ``Delta`` from :func:`score_perturbation_bound`)
    yields a band that covers the **true** silhouette even though the scores were built from
    **estimated** diagrams with ``W_1`` error ``<= epsilon`` -- the exact analogue of the
    Lipschitz grid slack (:func:`~tcda_uq.uq.conformal.functional_cp.grid_discretization_slack`),
    but for diagram estimation rather than grid discretization.
    """
    delta = score_perturbation_bound(K, epsilon, s)
    return np.asarray(scores, dtype=float) + delta


def certified_width_inflation(K: float, epsilon: float, s) -> float:
    """Extra band **width** the diagram-robustness certificate costs (mean over ``t``).

    The half-width grows by ``Delta * s(t)`` at each ``t`` (``Delta`` from
    :func:`score_perturbation_bound`), so the mean *full-width* inflation is
    ``2 * Delta * mean_t s(t) = 2 K epsilon * mean(s) / min(s)``. Reported so the price of
    diagram robustness is explicit.
    """
    s = np.asarray(s, dtype=float)
    delta = score_perturbation_bound(K, epsilon, s)
    return 2.0 * delta * float(np.mean(s))


def coverage_certificate(calib_scores, radius, K: float, epsilon: float, s):
    """Worst-case coverage loss of an *uncorrected* band under diagram error ``<= epsilon``.

    Here ``calib_scores`` are the scores computed from the estimated diagrams. If the band
    uses the ordinary radius ``k`` but the true score can be larger than the estimated score
    by at most ``Delta``, an extra true-silhouette miss can only occur when the estimated
    score lies in ``(k - Delta, k]``. The empirical fraction of calibration scores in that
    window upper-bounds the extra miscoverage, giving a *certificate* that coverage degrades
    by at most this much:

        coverage(true)  >=  coverage(estimated) - #{ k - Delta < S_hat_i <= k } / n.

    Args:
        calib_scores: calibration scores ``[n]`` (as computed on the estimated diagrams).
        radius: the conformal radius ``k`` actually used (scalar).
        K, epsilon, s: as above.

    Returns:
        dict with ``delta`` (score slack), ``coverage_loss_bound`` (the fraction above),
        and ``certified_coverage`` = empirical estimated-score coverage minus the loss
        bound. ``radius`` ``+inf`` -> zero loss (an unbounded band cannot miss).
    """
    scores = np.asarray(calib_scores, dtype=float)
    n = scores.shape[0]
    delta = score_perturbation_bound(K, epsilon, s)
    if not np.isfinite(radius):
        loss = 0.0
    else:
        loss = float(np.mean((scores > radius - delta) & (scores <= radius)))
    emp_cov = float(np.mean(scores <= radius)) if np.isfinite(radius) else 1.0
    return dict(delta=delta, coverage_loss_bound=loss,
                certified_coverage=max(emp_cov - loss, 0.0))
