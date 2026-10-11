"""Finite-degree multiplicity for marginal individual prediction.

Bonferroni allocates alpha/(2D) per arm and degree. A joint maximum score
allocates alpha/2 per arm across a prespecified finite degree set with aligned
units and fixed scoring rules. Both inherit the true-ratio target scope or the
learned-ratio TV qualification; no general efficiency ordering is asserted.
Sidak is experimental unless a sufficient dependence condition is justified.
"""

from __future__ import annotations

import numpy as np

from .weighted_cp import weighted_conformal_radius, ConformalArm
from ..asymptotic._validation import validate_alpha
import warnings


def _per_dim_scheme_level(alpha: float, n_dim: int, scheme: str) -> float:
    """Per-dim ITTE level under a degree-multiplicity ``scheme`` (bonferroni/sidak)."""
    if scheme == "bonferroni":
        return alpha / n_dim
    if scheme == "sidak":
        warnings.warn("Sidak allocation requires a separate dependence justification; it is experimental", UserWarning, stacklevel=3)
        return 1.0 - (1.0 - alpha) ** (1.0 / n_dim)
    raise ValueError(f"scheme must be 'bonferroni' or 'sidak', got {scheme!r}")


def bonferroni_itte_bounds(model, X_new, alpha: float, *, scheme: str = "bonferroni",
                           dims=None):
    """Simultaneous-over-degree ITTE bounds by per-dim level correction.

    Bands each requested dim at the corrected per-dim level (:func:`_per_dim_scheme_level`)
    and returns the list of ``(lower, upper, center)``. For ``scheme="bonferroni"``, their
    *intersection over d* holds the whole vector ``(delta_d)_d`` with joint probability
    ``>= 1 - alpha`` by the union bound. For ``scheme="sidak"``, the same statement needs
    the usual Sidak product / dependence condition across degree-miss events.

    Args:
        model: a fitted band with ``band_bounds(X, level, d)`` (``ITTEConformal`` or
            ``AdaptiveITTEConformal``).
        X_new: covariates ``[m, p]``.
        alpha: joint miscoverage budget across all dims.
        scheme: ``"bonferroni"`` or ``"sidak"``.
        dims: iterable of homology dims (default all of ``model.n_hom_dim``).

    Returns:
        dict ``d -> (lower, upper, center)`` each ``[m, resolution]``.
    """
    dims = _dimensions(model, dims, alpha)
    a_d = _per_dim_scheme_level(alpha, len(dims), scheme)
    return {d: model.band_bounds(X_new, a_d, d=d) for d in dims}


def joint_itte_bounds(model, X_new, alpha: float, *, dims=None):
    """Simultaneous-over-``(d, t)`` ITTE bounds via the joint sup-over-degree score.

    Per arm ``a``, the calibration score is maxed over the requested degrees,
    ``S_i^a = max_d (calibration sup-norm score of unit i in dim d)``; its weighted
    conformal quantile ``k_a`` (at ``alpha/2``) then bounds *every* degree at once. The
    dim-``d`` half-width is ``k_a * s_{a,d}(t)`` and the arms compose by interval
    arithmetic, so a fresh unit's whole vector ``(delta_d)_d`` is covered with
    probability ``>= 1 - alpha`` with actual likelihood ratios and frozen scores,
    simultaneously on the fixed grid. Learned weights require TV qualification.

    Requires ``model.arms0``/``arms1`` to be
    :class:`~tcda_uq.uq.conformal.weighted_cp.ConformalArm` (marginal Phase-4 band); the
    covariate-adaptive arms carry an ``x``-dependent scale that does not fold into a single
    per-unit score, so use :func:`bonferroni_itte_bounds` for those.

    Returns:
        dict ``d -> (lower, upper, center)`` each ``[m, resolution]``.
    """
    dims = _dimensions(model, dims, alpha)
    X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
    beta = alpha / 2.0

    out, radii = {}, {}
    for a, arms in enumerate((model.arms0, model.arms1)):
        selected = [arms[d] for d in dims]
        if any(type(arm) is not ConformalArm for arm in selected):
            raise ValueError("joint degree calibration requires plain ConformalArm; use Bonferroni for adaptive/experimental arms")
        first = selected[0]
        for arm in selected[1:]:
            if not np.array_equal(arm.calib_index_, first.calib_index_) or not np.array_equal(arm.w_calib, first.w_calib) or not np.array_equal(arm._w_new(X_new), first._w_new(X_new)):
                raise ValueError("joint scores need identical aligned calibration units and weights across degrees")
        joint_scores = np.stack([arm.scores for arm in selected]).max(axis=0)
        radii[a] = np.atleast_1d(weighted_conformal_radius(joint_scores, beta, first.w_calib, first._w_new(X_new)))
    for d in dims:
        arm1, arm0 = model.arms1[d], model.arms0[d]
        c1, c0 = arm1.center(X_new), arm0.center(X_new)
        half = radii[1][:, None]*arm1.s[None, :] + radii[0][:, None]*arm0.s[None, :]
        center = c1-c0
        out[d] = (center-half, center+half, center)
    return out


def simultaneous_itte_bounds(model, X_new, alpha: float, *, scheme: str = "joint",
                             dims=None):
    """Dispatch to the requested degree-multiplicity ``scheme``.

    ``scheme in {"joint", "bonferroni", "sidak"}`` (see module docstring). ``"joint"``
    uses a joint per-unit maximum without a degree Bonferroni penalty; ``"sidak"`` is
    a comparator unless its dependence condition is part of the theorem being invoked.
    """
    if scheme == "joint":
        return joint_itte_bounds(model, X_new, alpha, dims=dims)
    return bonferroni_itte_bounds(model, X_new, alpha, scheme=scheme, dims=dims)


def _dimensions(model, dims, alpha):
    validate_alpha(alpha)
    dims = list(range(model.n_hom_dim)) if dims is None else list(dims)
    if not dims or len(set(dims)) != len(dims) or any(not isinstance(d, (int, np.integer)) or not 0 <= d < model.n_hom_dim for d in dims):
        raise ValueError("dims must be a nonempty unique subset of valid degrees")
    return dims
