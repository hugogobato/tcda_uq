"""Normalized-tent stability and numerical score perturbation helpers.

The revised W1 bridge is ||Phi_r(D)-Phi_r(E)|| <= (1+2r) W1(D,E)
for r>=1, without a lower persistence bound. Perturbation helpers require caller-
supplied true error bounds and independent fixed scoring rules. Calibration-score
fractions are empirical diagnostics, never population coverage certificates.
"""

from __future__ import annotations

import numpy as np
import warnings


def bridge_constant(*args, r: float = 1.0, normalization: str = "plain"):
    """Dimension-free normalized-tent W1 stability constant 1+2r, r>=1.

    ``normalization='legacy_gudhi'`` (or the alias ``'gudhi'``) scales the constant by sqrt(2). Legacy positional
    ``(L, r, c)`` calls are accepted with a deprecation warning; L and c play no
    role in the revised theorem. A single positional argument is the exponent.
    """
    if len(args) == 1:
        r = args[0]
    elif len(args) == 3:
        warnings.warn("bridge_constant(L,r,c) is deprecated; use bridge_constant(r=r). The revised bound ignores L and c", FutureWarning, stacklevel=2)
        r = args[1]
    elif args:
        raise TypeError("use bridge_constant(r=...) or deprecated (L,r,c)")
    if not np.isfinite(r) or r < 1:
        raise ValueError("the normalized silhouette stability theorem requires finite r >= 1")
    if normalization not in ("plain", "gudhi", "legacy_gudhi"):
        raise ValueError("normalization must be 'plain' or 'legacy_gudhi' ('gudhi' alias)")
    return float((1+2*r) * (np.sqrt(2) if normalization in ("gudhi", "legacy_gudhi") else 1))


def estimate_bridge_params(diagrams, r: float, *, eps: float = 1e-12):
    """Deprecated lifetime diagnostics (L,c), unnecessary for the revised bridge.

    Empty input returns (0,0); this does not alter the universal 1+2r constant.
    """
    warnings.warn("estimate_bridge_params is a legacy lifetime diagnostic; L,c are not required by the revised 1+2r bound", FutureWarning, stacklevel=2)
    if not np.isfinite(r) or r < 1:
        raise ValueError("r must be finite and at least 1")
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
    if not np.isfinite(s).all() or np.any(s <= 0) or not np.isfinite(K) or K < 0 or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError("finite nonnegative K/epsilon and strictly positive finite scales required")
    return float(K) * float(epsilon) / s_min


def certified_score_inflation(scores, K: float, epsilon: float, s):
    """Numerically inflate scores by a supplied deterministic bridge bound.

    A true-silhouette coverage claim additionally needs a justified score/target
    perturbation comparison and probabilities for the armwise error events.
    This arithmetic helper does not estimate those probabilities.
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
    """Deprecated empirical calibration-score sensitivity, not a coverage bound.

    The returned legacy fields summarize calibration scores near a chosen radius.
    Their empirical fractions cannot bound fresh-unit population miscoverage.
    For such a theorem, supply justified diagram error probabilities and armwise
    score inflation separately; neither follows from this observed fraction.
    """
    warnings.warn("coverage_certificate reports empirical score sensitivity only, not a population coverage certificate; use empirical_score_sensitivity", FutureWarning, stacklevel=2)
    empirical = empirical_score_sensitivity(calib_scores, radius, K, epsilon, s)
    loss = empirical["empirical_affected_fraction"]
    return dict(delta=empirical["delta"], coverage_loss_bound=loss,
                certified_coverage=max(empirical["empirical_inside_fraction"]-loss, 0),
                scope="empirical calibration score sensitivity only")


def empirical_score_sensitivity(calib_scores, radius, K: float, epsilon: float, s):
    """Empirical score fraction affected by a supplied deterministic perturbation."""
    scores = np.asarray(calib_scores, float)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("calib_scores must be a finite nonempty vector")
    delta = score_perturbation_bound(K, epsilon, s)
    if np.isnan(radius) or radius < 0:
        raise ValueError("radius must be nonnegative, possibly infinity")
    fraction = 0.0 if np.isinf(radius) else float(np.mean((scores > radius-delta) & (scores <= radius)))
    return {"delta": delta, "empirical_affected_fraction": fraction,
            "empirical_inside_fraction": float(np.mean(scores <= radius))}
