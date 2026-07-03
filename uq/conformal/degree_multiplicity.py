"""Simultaneous ITTE bands across homology degree (Phase 6.4).

The Phase-4 composed band :class:`~tcda_uq.uq.conformal.ITTEConformal` guarantees, for
each homology dim ``d`` *separately*,

    P( delta_{n+1,d}(.) in C_d(X_{n+1}, .) )  >=  1 - alpha,   simultaneously over t.

But a topological effect lives across *several* degrees at once (``H_0`` connected
components, ``H_1`` loops, ``H_2`` voids). A statement about the joint object
``(delta_0, delta_1, delta_2)`` needs an extra multiplicity correction over ``d`` -- the
sup-norm already folds in the continuum ``t``, but the finite degree axis is a separate
union. Three schemes, from bluntest to sharpest:

  * ``"bonferroni"`` -- band every dim at level ``alpha / n_dim`` and intersect. Then
    ``P(any dim missed) <= sum_d alpha/n_dim = alpha`` (union bound). Works with *any*
    per-dim band (marginal or covariate-adaptive), at the cost of a factor ``n_dim`` in
    each dim's level (so each arm at ``alpha/(2 n_dim)``).
  * ``"sidak"`` -- band every dim at ``1 - (1-alpha)^(1/n_dim)``. Sharper than Bonferroni
    under near-independence across degrees; still a per-dim correction.
  * ``"joint"`` -- **fold the degree axis into the sup-norm** exactly as the ``t`` axis
    is folded. Per arm ``a``, define the joint score ``S_i^a = max_d max_t
    |r^a_{i,d}(t)| / s_{a,d}(t)`` (one scalar per unit, maxing over *both* degree and
    ``t``); its weighted-conformal quantile ``k_a`` controls **all degrees at once** with
    no Bonferroni-over-``d`` penalty. Composing the two arms at ``alpha/2`` gives
    simultaneous-over-``(d, t)`` ITTE coverage at level ``alpha``. This is the
    topology-appropriate "sharper multiple-testing scheme": degrees are just another
    finite index the sup-norm absorbs, so a unit that is anomalous in *any* degree is
    anomalous jointly -- no independence assumption, and (empirically, Phase 6) markedly
    tighter than Bonferroni when the degrees are correlated.

The joint scheme requires the per-arm calibration scores to be aligned across degrees
(same arm-``a`` calibration units, same order) -- which is exactly how
:class:`~tcda_uq.uq.conformal.weighted_cp.ConformalArm` builds them -- so it operates on
a fitted :class:`ITTEConformal`'s ``arms0``/``arms1`` directly.
"""

from __future__ import annotations

import numpy as np

from .weighted_cp import weighted_conformal_radius


def _per_dim_scheme_level(alpha: float, n_dim: int, scheme: str) -> float:
    """Per-dim ITTE level under a degree-multiplicity ``scheme`` (bonferroni/sidak)."""
    if scheme == "bonferroni":
        return alpha / n_dim
    if scheme == "sidak":
        return 1.0 - (1.0 - alpha) ** (1.0 / n_dim)
    raise ValueError(f"scheme must be 'bonferroni' or 'sidak', got {scheme!r}")


def bonferroni_itte_bounds(model, X_new, alpha: float, *, scheme: str = "bonferroni",
                           dims=None):
    """Simultaneous-over-degree ITTE bounds by per-dim level correction.

    Bands each requested dim at the corrected per-dim level (:func:`_per_dim_scheme_level`)
    and returns the list of ``(lower, upper, center)`` -- their *intersection over d* holds
    the whole vector ``(delta_d)_d`` with joint probability ``>= 1 - alpha``.

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
    dims = list(range(model.n_hom_dim)) if dims is None else list(dims)
    a_d = _per_dim_scheme_level(alpha, len(dims), scheme)
    return {d: model.band_bounds(X_new, a_d, d=d) for d in dims}


def joint_itte_bounds(model, X_new, alpha: float, *, dims=None):
    """Simultaneous-over-``(d, t)`` ITTE bounds via the joint sup-over-degree score.

    Per arm ``a``, the calibration score is maxed over the requested degrees,
    ``S_i^a = max_d (calibration sup-norm score of unit i in dim d)``; its weighted
    conformal quantile ``k_a`` (at ``alpha/2``) then bounds *every* degree at once. The
    dim-``d`` half-width is ``k_a * s_{a,d}(t)`` and the arms compose by interval
    arithmetic, so a fresh unit's whole vector ``(delta_d)_d`` is covered with
    probability ``>= 1 - alpha``, simultaneously over degree and ``t`` -- with **no**
    Bonferroni-over-``d`` inflation.

    Requires ``model.arms0``/``arms1`` to be
    :class:`~tcda_uq.uq.conformal.weighted_cp.ConformalArm` (marginal Phase-4 band); the
    covariate-adaptive arms carry an ``x``-dependent scale that does not fold into a single
    per-unit score, so use :func:`bonferroni_itte_bounds` for those.

    Returns:
        dict ``d -> (lower, upper, center)`` each ``[m, resolution]``.
    """
    dims = list(range(model.n_hom_dim)) if dims is None else list(dims)
    X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
    beta = alpha / 2.0

    out = {}
    for arms in (model.arms0, model.arms1):
        # joint per-arm calibration score: max over degrees (scores aligned by unit)
        stacked = np.stack([np.asarray(arms[d].scores, dtype=float) for d in dims])  # [D, n_a]
        joint_scores = stacked.max(axis=0)                                           # [n_a]
        w_calib = arms[dims[0]].w_calib                       # same weights for all dims/arm
        w_new = arms[dims[0]]._w_new(X_new)                   # [m]
        k = np.atleast_1d(weighted_conformal_radius(joint_scores, beta, w_calib, w_new))
        for d in dims:
            arms[d]._joint_k = k                              # stash for composition below

    for d in dims:
        arm1, arm0 = model.arms1[d], model.arms0[d]
        c1, c0 = arm1.center(X_new), arm0.center(X_new)       # [m, res]
        half = arm1._joint_k[:, None] * arm1.s[None, :] + arm0._joint_k[:, None] * arm0.s[None, :]
        center = c1 - c0
        out[d] = (center - half, center + half, center)
    return out


def simultaneous_itte_bounds(model, X_new, alpha: float, *, scheme: str = "joint",
                             dims=None):
    """Dispatch to the requested degree-multiplicity ``scheme``.

    ``scheme in {"joint", "bonferroni", "sidak"}`` (see module docstring). ``"joint"``
    is the sharper topology-specific scheme and the recommended default.
    """
    if scheme == "joint":
        return joint_itte_bounds(model, X_new, alpha, dims=dims)
    return bonferroni_itte_bounds(model, X_new, alpha, scheme=scheme, dims=dims)
