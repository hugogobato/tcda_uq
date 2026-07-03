"""Positivity-stabilized weighting for causal conformal prediction (Phase 6.5, headline).

**The problem this fixes.** Layer-2 weighted CP (:mod:`.weighted_cp`) reweights the
arm-``a`` calibration scores by the propensity likelihood ratio to target the
*interventional* population ``X ~ p(x)``:

    a = 1 (treated):   w_naive(x)  =  1 / pi(x)
    a = 0 (control):   w_naive(x)  =  1 / (1 - pi(x)).

Where overlap is thin (``pi(x) -> 0`` for the treated arm, ``pi(x) -> 1`` for the
control arm) these weights *blow up*. The weak-overlap failure mode Phase 4.5
localized is exactly this: the weighted-CP ``+inf`` atom fires when the **test
point's** own normalized weight ``p_new = w_new / (sum w_cal + w_new)`` exceeds the
per-arm budget ``beta = alpha/2``, i.e. when ``w_new`` is huge relative to the
calibration weight mass. Phase 4.5 measured this: at overlap scale 4.0, ``inf%``
rises to ~34% and ``ESS/n_cal`` falls to ~0.46 while the *finite* band width barely
moves -- the pathology is **unboundedness, not widening**.

**The stabilization.** The clean fix is to change the *target population* to one on
which the likelihood-ratio weights are **bounded by construction**, following the
positivity-violation literature (Crump et al. 2009 trimming; Li, Morgan & Zaslavsky
2018 overlap weights; Li & Greene 2013 matching weights). Reweighting to a tilted
target ``g(x) proportional to h(pi(x)) * p(x)`` gives the arm-``a`` conformal weight

    w(x)  proportional to  g(x) / p(x | A = a)  =  h(pi(x)) / [a*pi(x) + (1-a)(1-pi(x))].

Choosing ``h(pi) = [pi(1-pi)]^gamma`` (the **overlap tilt**, ``gamma in [0,1]``):

    a = 1 :  w(x)  =  pi(x)^(gamma-1) (1 - pi(x))^gamma
    a = 0 :  w(x)  =  pi(x)^gamma (1 - pi(x))^(gamma-1).

  * ``gamma = 0`` recovers the naive weights (target = full population; unbounded).
  * ``gamma = 1`` is the **overlap-weighted** target ``g proportional to pi(1-pi) p``,
    for which ``w(x) = 1 - pi(x)`` (treated) / ``w(x) = pi(x)`` (control) -- **bounded
    in (0, 1]**. The ``+inf`` atom then essentially never fires: ``w_new <= 1`` while
    ``sum w_cal = O(n_cal * mean weight)``, so ``p_new = O(1/n_cal) << beta``.
  * intermediate ``gamma`` interpolates: a *tunable* trade of how much overlap-region
    emphasis (bounded weights, low ``inf%``) against how close the target stays to the
    full interventional population.

**Coverage guarantee (why this is honest, not a hack).** The overlap/tilted/matching
weights are just a *different, bounded weight function* fed to the identical weighted
split-conformal quantile. So the Tibshirani et al. (2019) / Lei & Candes (2021)
finite-sample guarantee applies verbatim: the band has coverage ``>= 1 - alpha`` and
is **doubly robust**, now *for the tilted target population* rather than the full
population. The estimand shift is explicit and reported (:func:`target_description`) --
you predict the individual effect for a unit drawn from the overlap population, i.e.
where treatment is not near-deterministic. This is the "turn the weakness into the
contribution" framing (concept note section 6): weak overlap becomes a controlled,
well-defined change of estimand instead of an unbounded band.

Two *same-estimand, approximate-coverage* alternatives are also provided for the case
where one insists on the full-population target:

  * ``clip`` -- clip ``pi_hat`` into ``[eps, 1-eps]`` before inverting (bounds the
    weight at ``1/eps``; biased where clipping binds).
  * ``shrink`` -- a smooth self-normalized cap ``w = w_naive / (1 + w_naive / M)`` that
    saturates at ``M`` (density-gated shrinkage toward the bulk; empirical-Bayes-style
    precision damping of the high-weight tail).

All methods share the signature ``f(pi_hat, arm) -> weights`` and so drop straight into
:class:`~tcda_uq.uq.conformal.weighted_cp.ConformalArm` via its ``weight_fn`` hook.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from ...estimators.aipw import _clip_pi

# methods whose coverage guarantee is *exact* (finite-sample, for their own tilted
# target population) vs. *approximate* (aimed at the full population, biased by the
# stabilization). Kept explicit so the manuscript never overstates a guarantee.
EXACT_TARGET_METHODS = ("naive", "overlap", "tilted", "matching")
APPROX_TARGET_METHODS = ("clip", "shrink")
STABILIZERS = EXACT_TARGET_METHODS + APPROX_TARGET_METHODS


def naive_weights(pi_hat, arm: int):
    """Naive inverse-propensity weights (``gamma = 0``; full-population target).

    ``1/pi`` (treated) / ``1/(1-pi)`` (control), with ``pi`` clipped by ``EPS_PI``
    only to stay finite. This is the Phase-4 default and the weak-overlap culprit.
    Identical to :func:`~tcda_uq.uq.conformal.weighted_cp.propensity_weights`.
    """
    pi = _clip_pi(pi_hat)
    if arm == 1:
        return 1.0 / pi
    if arm == 0:
        return 1.0 / (1.0 - pi)
    raise ValueError(f"arm must be 0 or 1, got {arm!r}")


def tilted_weights(pi_hat, arm: int, *, gamma: float = 1.0):
    """Overlap-tilted weights ``w = [pi(1-pi)]^gamma / p(x|A=a)`` (target ``g prop [pi(1-pi)]^gamma p``).

    ``gamma = 0`` -> naive (unbounded); ``gamma = 1`` -> overlap weights (bounded in
    ``(0, 1]``). Intermediate ``gamma`` interpolates the estimand and the weight bound.
    Exact finite-sample coverage for the ``gamma``-tilted target population.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma!r}")
    pi = _clip_pi(pi_hat)
    if arm == 1:
        return pi ** (gamma - 1.0) * (1.0 - pi) ** gamma
    if arm == 0:
        return pi ** gamma * (1.0 - pi) ** (gamma - 1.0)
    raise ValueError(f"arm must be 0 or 1, got {arm!r}")


def overlap_weights(pi_hat, arm: int):
    """Overlap weights (``gamma = 1``): ``w = 1-pi`` (treated) / ``pi`` (control).

    The Li-Morgan-Zaslavsky (2018) overlap population ``g proportional to pi(1-pi) p``.
    Weights are **bounded in (0, 1]** on both arms, so the weak-overlap ``+inf`` atom
    is structurally suppressed. Exact finite-sample coverage for the overlap population.
    """
    return tilted_weights(pi_hat, arm, gamma=1.0)


def matching_weights(pi_hat, arm: int):
    """Matching weights ``w = min(pi, 1-pi) / p(x|A=a)`` (Li & Greene 2013).

    Target ``g proportional to min(pi, 1-pi) p`` -- the conformal analogue of 1:1
    matching. Weights are bounded in ``(0, 1]``: ``min(1, (1-pi)/pi)`` (treated) /
    ``min(1, pi/(1-pi))`` (control). Slightly sharper overlap emphasis than
    ``overlap`` near ``pi = 1/2``. Exact coverage for the matching population.
    """
    pi = _clip_pi(pi_hat)
    h = np.minimum(pi, 1.0 - pi)
    if arm == 1:
        return h / pi
    if arm == 0:
        return h / (1.0 - pi)
    raise ValueError(f"arm must be 0 or 1, got {arm!r}")


def clip_weights(pi_hat, arm: int, *, eps: float = 0.1):
    """Clipped inverse-propensity weights: invert ``clip(pi, eps, 1-eps)``.

    Bounds the weight at ``1/eps`` (treated) resp. ``1/eps`` (control). Targets the
    full interventional population but is **biased** in the region where the clip binds
    (``pi < eps`` treated / ``pi > 1-eps`` control), so coverage is only *approximate*.
    A blunt baseline; the tilted weights are the principled version.
    """
    pi = np.clip(np.asarray(pi_hat, dtype=float), eps, 1.0 - eps)
    if arm == 1:
        return 1.0 / pi
    if arm == 0:
        return 1.0 / (1.0 - pi)
    raise ValueError(f"arm must be 0 or 1, got {arm!r}")


def shrink_weights(pi_hat, arm: int, *, cap: float = 10.0):
    """Smoothly capped naive weights ``w = w_naive / (1 + w_naive / cap)`` (saturates at ``cap``).

    A density-gated / empirical-Bayes-style shrinkage of the high-weight tail toward the
    bulk: identical to the naive weight for small ``w_naive`` and asymptoting to ``cap``
    as ``w_naive -> inf``. Same (full-population) estimand as ``naive`` but *approximate*
    coverage -- the tail damping trades a little bias for a bounded weight (max ``cap``,
    so ``p_new <= cap/(sum w_cal + cap)``, which controls the ``+inf`` atom).
    """
    w = naive_weights(pi_hat, arm)
    return w / (1.0 + w / float(cap))


def stabilized_weights(pi_hat, arm: int, method: str = "overlap", **params):
    """Dispatch to a named stabilizer. ``method in`` :data:`STABILIZERS`.

    Args:
        pi_hat: estimated propensities ``[n]``.
        arm: 0 (control) or 1 (treated).
        method: ``"naive"``, ``"overlap"``, ``"tilted"`` (needs ``gamma``),
            ``"matching"``, ``"clip"`` (``eps``) or ``"shrink"`` (``cap``).
        **params: method-specific keyword args (``gamma`` / ``eps`` / ``cap``).

    Returns:
        Unnormalized weights ``[n]`` (the conformal radius renormalizes them).
    """
    table = {
        "naive": naive_weights,
        "overlap": overlap_weights,
        "tilted": tilted_weights,
        "matching": matching_weights,
        "clip": clip_weights,
        "shrink": shrink_weights,
    }
    if method not in table:
        raise ValueError(f"method must be one of {tuple(table)}, got {method!r}")
    return table[method](pi_hat, arm, **params)


def make_weight_fn(method: str = "overlap", **params):
    """Bind a stabilizer + its params into a ``weight_fn(pi_hat, arm) -> weights``.

    The exact hook :class:`~tcda_uq.uq.conformal.weighted_cp.ConformalArm` /
    :class:`~tcda_uq.uq.conformal.composition.ITTEConformal` expect. ``method="naive"``
    reproduces the Phase-4 band byte-for-byte.
    """
    fn = partial(stabilized_weights, method=method, **params)
    fn.method = method            # tag for bookkeeping / plotting
    fn.params = dict(params)
    return fn


def weight_upper_bound(method: str, **params):
    """Worst-case value of a single weight for ``method`` (``inf`` if unbounded).

    Diagnostic for how aggressively a method suppresses the ``+inf`` atom: bounded
    methods (overlap/tilted with ``gamma>0``/matching/clip/shrink) cannot let one test
    point dominate the calibration mass, so ``p_new`` stays below ``beta`` far longer.
    """
    if method in ("overlap", "matching"):
        return 1.0
    if method == "tilted":
        gamma = float(params.get("gamma", 1.0))
        # w_treated = pi^(g-1)(1-pi)^g is unbounded as pi->0 unless gamma>=1;
        # for gamma in (0,1) the sup is at the EPS_PI-clipped boundary.
        if gamma >= 1.0:
            return 1.0
        from ...estimators.aipw import EPS_PI
        return EPS_PI ** (gamma - 1.0)
    if method == "clip":
        return 1.0 / float(params.get("eps", 0.1))
    if method == "shrink":
        return float(params.get("cap", 10.0))
    if method == "naive":
        from ...estimators.aipw import EPS_PI
        return 1.0 / EPS_PI
    raise ValueError(f"unknown method {method!r}")


def target_description(method: str, **params) -> str:
    """One-line description of the target population a method's coverage refers to.

    Keeps the estimand shift explicit: exact-target methods change *what* the band
    covers (a draw from a tilted population), approximate-target methods keep the full
    population but relax the guarantee.
    """
    if method == "naive":
        return "full interventional population p(x); exact coverage but unbounded weights"
    if method == "overlap":
        return "overlap population g(x) prop pi(x)(1-pi(x)) p(x); exact coverage, bounded weights"
    if method == "tilted":
        g = params.get("gamma", 1.0)
        return (f"tilted population g(x) prop [pi(1-pi)]^{g} p(x); exact coverage "
                f"(gamma=0 -> full pop, gamma=1 -> overlap pop)")
    if method == "matching":
        return "matching population g(x) prop min(pi,1-pi) p(x); exact coverage, bounded weights"
    if method == "clip":
        e = params.get("eps", 0.1)
        return f"full population, approximate: pi clipped to [{e}, {1-e}] (biased where clip binds)"
    if method == "shrink":
        c = params.get("cap", 10.0)
        return f"full population, approximate: naive weight smoothly capped at {c}"
    raise ValueError(f"unknown method {method!r}")
