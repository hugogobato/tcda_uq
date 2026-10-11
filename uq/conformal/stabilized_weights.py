"""Declared target ratios and approximate full-target weight regularization.

With true propensity, full, overlap, matching, and power-tilted formulas are
proportional to actual arm-to-target density ratios. Learned propensities require
TV qualification for the intended law. Clipping and shrinking approximate full
ratios and do not preserve exact full-target coverage. Bounded weights cannot
prevent the +infinity atom in every sample, especially with empty or tiny arms.
"""
from __future__ import annotations

from functools import partial
import numpy as np
from .weighted_cp import _probabilities, propensity_weights

# Formula categories conditional on TRUE propensity; not claims about learned weights.
EXACT_TARGET_METHODS = ("naive", "full", "overlap", "tilted", "matching")
APPROX_TARGET_METHODS = ("clip", "shrink")
STABILIZERS = EXACT_TARGET_METHODS + APPROX_TARGET_METHODS


def _arm(arm):
    if arm not in (0, 1):
        raise ValueError("arm must be 0 or 1")


def naive_weights(pi_hat, arm: int):
    """Full-population ratios with strict positivity and no clipping."""
    return propensity_weights(pi_hat, arm)


def tilted_weights(pi_hat, arm: int, *, gamma: float = 1.0):
    """Ratios for target covariate density proportional to [pi(1-pi)]^gamma p.

    gamma=1 admits endpoint probabilities, with a continuous ratio extension
    outside target support. gamma<1 requires strict positivity and is unbounded.
    """
    _arm(arm)
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be finite and in [0, 1]")
    pi = _probabilities(pi_hat, strict=gamma < 1)
    if gamma == 1:
        return 1.0-pi if arm == 1 else pi
    return pi ** (gamma-1) * (1-pi) ** gamma if arm == 1 else pi ** gamma * (1-pi) ** (gamma-1)


def overlap_weights(pi_hat, arm: int):
    """Ratios for the overlap target pi(1-pi)p, bounded by one."""
    return tilted_weights(pi_hat, arm, gamma=1)


def matching_weights(pi_hat, arm: int):
    """Ratios for the matching target min(pi,1-pi)p, bounded by one.

    At a denominator-zero endpoint the continuous extension is one; such an
    observation cannot occur in that source arm. Opposite endpoints get zero
    weight because they are outside target support.
    """
    _arm(arm)
    pi = _probabilities(pi_hat)
    e = pi if arm == 1 else 1-pi
    return np.divide(np.minimum(pi, 1-pi), e, out=np.ones_like(pi), where=e > 0)


def clip_weights(pi_hat, arm: int, *, eps: float = .1):
    """Explicit clipped inverse ratios, approximate for the full target."""
    _arm(arm)
    if not np.isfinite(eps) or not 0 < eps < .5:
        raise ValueError("eps must be finite and strictly between 0 and .5")
    pi = np.clip(_probabilities(pi_hat), eps, 1-eps)
    return 1.0 / (pi if arm == 1 else 1-pi)


def shrink_weights(pi_hat, arm: int, *, cap: float = 10.):
    """Smooth cap 1/(e_a+1/cap), approximate for the full target."""
    _arm(arm)
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("cap must be finite and positive")
    pi = _probabilities(pi_hat)
    return 1.0 / ((pi if arm == 1 else 1-pi) + 1.0/cap)


def stabilized_weights(pi_hat, arm: int, method: str = "overlap", **params):
    table = {"naive": naive_weights, "full": naive_weights, "overlap": overlap_weights,
             "tilted": tilted_weights, "matching": matching_weights,
             "clip": clip_weights, "shrink": shrink_weights}
    if method not in table:
        raise ValueError(f"method must be one of {tuple(table)}")
    return table[method](pi_hat, arm, **params)


def make_weight_fn(method: str = "overlap", **params):
    """Bind a declared ratio formula to the ConformalArm weight hook."""
    # Eager validation catches invalid parameter domains before calibration.
    stabilized_weights(np.array([.5]), 1, method=method, **params)
    fn = partial(stabilized_weights, method=method, **params)
    fn.method, fn.params = method, dict(params)
    return fn


def weight_upper_bound(method: str, **params):
    """True worst-case formula bound, without implicit numerical clipping."""
    stabilized_weights(np.array([.5]), 1, method=method, **params)
    if method in ("overlap", "matching") or (method == "tilted" and params.get("gamma", 1.) == 1):
        return 1.0
    if method == "clip":
        return 1.0 / params.get("eps", .1)
    if method == "shrink":
        return float(params.get("cap", 10.))
    return np.inf


def target_description(method: str, **params):
    """Intended population and qualification for its ratio formula."""
    stabilized_weights(np.array([.5]), 1, method=method, **params)
    if method in ("full", "naive"):
        target = "full population p(x), unbounded inverse ratios"
    elif method == "overlap":
        target = "overlap population proportional to pi(x)(1-pi(x)) p(x), bounded ratios"
    elif method == "matching":
        target = "matching population proportional to min(pi(x),1-pi(x)) p(x), bounded ratios"
    elif method == "tilted":
        target = f"tilted population proportional to [pi(x)(1-pi(x))]^{params.get('gamma', 1.)} p(x)"
    else:
        return f"full population, approximate {method} ratios; TV discrepancy required, no guaranteed finite radius"
    return target + "; true ratios give a marginal coverage lower bound, learned ratios require TV qualification; no guaranteed finite radius"
