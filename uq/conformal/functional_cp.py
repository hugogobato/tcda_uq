"""Layer 1 -- functional split conformal prediction (Phase 4.1).

Finite-sample, distribution-free **prediction** bands for a *curve-valued*
outcome, following Diquigiovanni-Fontana-Vantini (2022). The construction
collapses each calibration curve to a single scalar via a **sup-norm
nonconformity score with a modulation function** ``s(t)``:

    S_i  =  sup_t | phi_{i,d}(t) - mu_hat_a(t, X_i) | / s(t).

Given a fitted regression ``mu_hat_a`` (any predictor) and calibration scores
``{S_i}``, the split-conformal band around a new prediction is

    mu_hat_a(t, x)  +/-  k_{1-beta} * s(t),

with ``k_{1-beta}`` the ``ceil((n_cal+1)(1-beta))`` order statistic of the
scores (the finite-sample conformal quantile). This is the *unweighted*
(exchangeable) core; the causal covariate-shift reweighting is Layer 2
(:mod:`.weighted_cp`) and the arm composition is Layer 3 (:mod:`.composition`).

**Crucial reduction (why Layer 2 drops in with no new theory).** The sup-norm
score maps each *curve* to one *scalar* ``S_i``; weighting then acts on the
scalars exactly as in the standard weighted-CP proof.

**Why the topological outcome is unusually friendly here.** The power-weighted
silhouette is 1-Lipschitz in ``t`` (TATE Lemma 2.1) and excludes infinite-
persistence pairs, so ``||phi||_inf < infinity`` and the residual curves are
equicontinuous: the sup-norm and modulation are numerically tame (no envelope
pathologies), and the sup over the finite grid controls the sup over the whole
interval up to the grid spacing (see :func:`grid_discretization_slack`).

Modulation ``s(t)`` (the width shape; Phase 4.1):
  * ``"constant"``     -- ``s(t) = 1`` (uniform-width band, the plain sup-norm).
  * ``"pointwise-sd"`` -- ``s(t) = sd_i r_{i}(t)`` (local variability; the DFV
    default, shortest bands where the residual spread varies over ``t``).
  * ``"lipschitz"``    -- pointwise-sd floored away from 0. Because the residual
    curves are 1-Lipschitz-driven (Lemma 2.1) a vanishing ``s(t)`` in a flat
    region would blow up the score; the floor keeps the band well posed and is
    the Lipschitz-aware modulation of Phase 4.1.

**Exchangeability note.** The scores ``{S_i}`` are exchangeable when ``s(t)`` is
either (i) a function of a data split independent of the calibration residuals
(pass ``modulation_residuals=``) or (ii) a *symmetric* function of the
calibration residuals (the default here -- ``pointwise-sd`` over the calibration
set). Case (ii) is the standard DFV practice; it self-normalizes the scores and
its finite-sample coverage impact is negligible (validated in Phase 4.5).
"""

from __future__ import annotations

import numpy as np

from ...metrics import Band

_MODULATIONS = ("constant", "pointwise-sd", "lipschitz")


def residual_curves(phi_a, mu_hat_a):
    """Residual curves ``r_i(t) = phi_{i}(t) - mu_hat(t, X_i)``: ``[n, res]``.

    ``phi_a`` and ``mu_hat_a`` are both ``[n, resolution]`` (a single homology
    dim, a single arm).
    """
    return np.asarray(phi_a, dtype=float) - np.asarray(mu_hat_a, dtype=float)


def modulation(residuals, kind: str = "pointwise-sd", *, floor: float = 0.0, eps: float = 1e-8):
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
    kind = kind.lower()
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
    out = np.max(np.abs(R) / s, axis=-1)
    return out if out.shape[0] > 1 else float(out[0])


def split_conformal_radius(scores, alpha: float):
    """Unweighted split-conformal radius ``k_{1-alpha}`` from scalar scores.

    The ``ceil((n+1)(1-alpha))`` smallest score (finite-sample conformal
    quantile). Returns ``+inf`` when ``ceil((n+1)(1-alpha)) > n`` (too few
    calibration points for the requested level -> the honest band is the whole
    line). This is the special case of :func:`~.weighted_cp.weighted_conformal_radius`
    with equal weights.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
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
    """Upper bound on the sup-over-``t`` score missed by the finite-grid sup.

    Because the silhouette is ``lipschitz_const``-Lipschitz in ``t`` (TATE
    Lemma 2.1) and ``mu_hat`` is Lipschitz with the same modulus family, the
    residual ``r(t)`` is Lipschitz, so between two grid nodes at spacing
    ``delta`` the true ``|r|`` exceeds the grid maximum by at most
    ``lipschitz_const * delta / 2``. Dividing by ``min_t s(t)`` converts that to
    a score slack. Adding this slack to the calibration scores gives a band that
    is simultaneous over the *whole* interval, not just the grid -- a
    topology-specific guarantee with no generic-functional-CP analogue.

    Args:
        scores_grid: the grid spacing ``delta`` (scalar) or ``tseq`` to infer it.
        s: modulation ``[resolution]`` (uses its minimum).
        lipschitz_const: Lipschitz modulus of the residual in ``t`` (``2`` if
            both ``phi`` and ``mu_hat`` are 1-Lipschitz).

    Returns:
        Scalar score slack to add to each calibration score.
    """
    g = np.asarray(scores_grid, dtype=float)
    delta = float(g) if g.ndim == 0 else float(np.max(np.diff(g)))
    s_min = float(np.min(np.asarray(s, dtype=float)))
    return lipschitz_const * delta / 2.0 / max(s_min, 1e-12)
