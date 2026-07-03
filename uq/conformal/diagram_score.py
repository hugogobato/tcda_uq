"""Diagram-space (Wasserstein) nonconformity scores (Phase 6.2, optional).

The default Phase-4 band scores in **silhouette space** (:mod:`.functional_cp`): cheap,
but the silhouette blurs the *number* of homological features (concept note section 5.1).
The intrinsic alternative scores in **diagram space**:

    S_i  =  W_q( D_{i,d},  D_hat_{A_i} ),

the ``q``-Wasserstein distance between unit ``i``'s degree-``d`` persistence diagram and a
*predicted* diagram ``D_hat_a`` for its arm. Because a Wasserstein distance is a **scalar**,
the causal reweighting (Layer 2) and the whole split-conformal machinery drop in unchanged
-- only Layer 1 (curve -> scalar) is swapped for (diagram -> scalar). The resulting radius
``k`` defines a **Wasserstein ball** ``{ D : W_q(D, D_hat_a) <= k }`` as a finite-sample
prediction set for the potential-outcome diagram.

Trade-off (why this is *optional*): more faithful to the geometry, but Frechet means of
persistence diagrams are non-unique / unstable and the pairwise Wasserstein solves are
``O((n+m)^3)`` Hungarian assignments -- heavy next to the vectorized silhouette sup-norm.
Provided as a benchmark for the silhouette-space band and to exercise the Theorem-5.3
bridge (:mod:`.silhouette_bridge`) against the true diagram metric.

Self-contained: the ``q``-Wasserstein distance and the Turner et al. (2014) Frechet-mean
barycenter are implemented on top of :func:`scipy.optimize.linear_sum_assignment`, so the
core install needs no optional-transport (`ot`) dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

_INF = 1e12


def _clean(D):
    """Drop essential (infinite) pairs and near-diagonal noise; return ``(k, 2)`` float."""
    D = np.asarray(D, dtype=float)
    if D.size == 0:
        return np.empty((0, 2))
    D = D[np.isfinite(D).all(axis=1)]
    return D[D[:, 1] - D[:, 0] > 1e-12]


def _diag_cost(D, q):
    """L-inf distance to the diagonal, ``(death - birth)/2``, raised to ``q``: ``[k]``."""
    return ((D[:, 1] - D[:, 0]) / 2.0) ** q


def wasserstein_distance(D1, D2, q: float = 1.0):
    """``q``-Wasserstein distance between two persistence diagrams (L-inf ground metric).

    Matches TATE Definition C.2: ground metric ``||p - p'||_inf``, points may be matched to
    the diagonal. Built as a balanced assignment on the augmented cost matrix (points of
    each diagram + diagonal projections of the other's points), solved exactly with the
    Hungarian algorithm.

    Args:
        D1, D2: ``(k, 2)`` birth-death arrays (essential pairs dropped internally).
        q: Wasserstein order (``1`` = ``W_1``, the order in Theorem 5.3).

    Returns:
        ``W_q(D1, D2)`` (float).
    """
    D1, D2 = _clean(D1), _clean(D2)
    n, m = D1.shape[0], D2.shape[0]
    if n == 0 and m == 0:
        return 0.0
    size = n + m
    C = np.full((size, size), _INF)
    # point-to-point (L-inf ground metric, to the q)
    if n and m:
        ground = np.max(np.abs(D1[:, None, :] - D2[None, :, :]), axis=2)   # [n, m]
        C[:n, :m] = ground ** q
    # D1 points -> their own diagonal projection (rows 0..n-1, cols m..m+n-1)
    if n:
        d1 = _diag_cost(D1, q)
        C[:n, m:m + n] = _INF
        C[np.arange(n), m + np.arange(n)] = d1
    # D2 points -> their own diagonal projection (rows n..n+m-1, cols 0..m-1)
    if m:
        d2 = _diag_cost(D2, q)
        C[n:n + m, :m] = _INF
        C[n + np.arange(m), np.arange(m)] = d2
    # diagonal-to-diagonal: free
    C[n:, m:] = 0.0
    r_idx, c_idx = linear_sum_assignment(C)
    total = C[r_idx, c_idx].sum()
    return float(total ** (1.0 / q))


def frechet_mean_diagram(diagrams, q: float = 2.0, *, n_iter: int = 20, tol: float = 1e-6,
                         rng=None):
    """Frechet-mean (barycenter) persistence diagram (Turner et al. 2014).

    Minimizes ``sum_i W_q(mu, D_i)^q`` by alternating (i) optimal matching of every diagram
    to the current mean ``mu`` and (ii) moving each mean point to the average of the points
    matched to it (points matched to the diagonal pull the mean toward the diagonal, and
    persistently-diagonal mean points are pruned). A local optimizer -- diagram barycenters
    are non-convex -- but adequate as a *predicted* diagram for the arm.

    Args:
        diagrams: list of ``(k_i, 2)`` diagrams (arm-``a`` training diagrams).
        q: order (2 is the usual barycenter order; the matching still uses the L-inf ground).
        n_iter, tol: iteration budget / movement tolerance.
        rng: seed for the initial-diagram choice.

    Returns:
        ``(k, 2)`` mean diagram (possibly empty).
    """
    diags = [_clean(D) for D in diagrams]
    diags = [D for D in diags if D.shape[0] > 0]
    if not diags:
        return np.empty((0, 2))
    rng = np.random.default_rng(rng)
    # init at the diagram minimizing total distance among a small random subset (medoid-ish)
    cand = diags if len(diags) <= 8 else [diags[i] for i in rng.choice(len(diags), 8, False)]
    costs = [sum(wasserstein_distance(D, Dj, q) for Dj in diags) for D in cand]
    mu = cand[int(np.argmin(costs))].copy()

    for _ in range(n_iter):
        if mu.shape[0] == 0:
            break
        accum = [list() for _ in range(mu.shape[0])]
        for D in diags:
            k = mu.shape[0]
            m = D.shape[0]
            C = np.full((k + m, k + m), _INF)
            if m:
                ground = np.max(np.abs(mu[:, None, :] - D[None, :, :]), axis=2) ** q
                C[:k, :m] = ground
            # mu points -> diagonal
            C[:k, m:m + k] = _INF
            C[np.arange(k), m + np.arange(k)] = _diag_cost(mu, q)
            # D points -> diagonal
            if m:
                C[k:k + m, :m] = _INF
                C[k + np.arange(m), np.arange(m)] = _diag_cost(D, q)
            C[k:, m:] = 0.0
            r_idx, c_idx = linear_sum_assignment(C)
            for ri, ci in zip(r_idx, c_idx):
                if ri < k:                                   # a mean point
                    if ci < m:
                        accum[ri].append(D[ci])              # matched to a real point
                    else:
                        accum[ri].append(_project(mu[ri]))   # matched to diagonal
        new_mu = []
        for pts in accum:
            if pts:
                new_mu.append(np.mean(np.asarray(pts), axis=0))
        new_mu = np.asarray(new_mu) if new_mu else np.empty((0, 2))
        new_mu = _clean(new_mu)
        move = (np.inf if new_mu.shape != mu.shape
                else float(np.max(np.abs(new_mu - mu))) if new_mu.size else 0.0)
        mu = new_mu
        if move < tol:
            break
    return mu


def _project(p):
    """Project a point onto the diagonal (mean of birth/death on both coords)."""
    mid = 0.5 * (p[0] + p[1])
    return np.array([mid, mid])


def wasserstein_scores(diagrams, ref_diagram, q: float = 1.0):
    """Wasserstein nonconformity scores ``S_i = W_q(D_i, ref)`` against a predicted diagram.

    Args:
        diagrams: list of ``(k_i, 2)`` diagrams (one per unit).
        ref_diagram: the predicted arm diagram ``D_hat_a`` (e.g. a Frechet mean).
        q: Wasserstein order.

    Returns:
        ``[n]`` array of scores.
    """
    return np.array([wasserstein_distance(D, ref_diagram, q) for D in diagrams], dtype=float)


class DiagramConformalArm:
    """Diagram-space split-CP for one arm (Phase 6.2): a Wasserstein-ball prediction set.

    Scores each calibration diagram by its ``W_q`` distance to a predicted (Frechet-mean)
    diagram, then takes the (optionally propensity-weighted) conformal quantile ``k``. The
    prediction set for a fresh potential-outcome diagram is the Wasserstein ball
    ``{ D : W_q(D, D_hat) <= k }``. Because the score is scalar, the Phase-4.2 weighting and
    the weak-overlap ``+inf`` atom behave exactly as in the silhouette-space arm.

    This is a *marginal* predicted diagram (Frechet mean over the arm), not conditional on
    ``X`` -- diagram-valued regression on ``X`` is out of scope; the point is the intrinsic
    diagram-metric benchmark to the silhouette band.
    """

    def __init__(self, ref_diagram, scores, w_calib, q):
        self.ref = ref_diagram
        self.scores = np.atleast_1d(np.asarray(scores, dtype=float))
        self.w_calib = np.asarray(w_calib, dtype=float)
        self.q = q

    @classmethod
    def fit(cls, calib_diagrams, q: float = 1.0, *, ref_diagram=None,
            train_diagrams=None, w_calib=None, frechet_q: float = 2.0, rng=None):
        """Fit from calibration diagrams (+ optional training diagrams for the mean).

        Args:
            calib_diagrams: list of arm-``a`` calibration diagrams.
            q: Wasserstein order for the score.
            ref_diagram: predicted diagram; if ``None`` it is the Frechet mean of
                ``train_diagrams`` (or of ``calib_diagrams`` if those are absent -- note
                that reuses calibration data, so pass a disjoint ``train_diagrams`` for an
                exact split-CP guarantee).
            train_diagrams: disjoint diagrams to build the Frechet mean from.
            w_calib: calibration weights ``[n]`` (default uniform / unweighted).
            frechet_q: order of the Frechet-mean barycenter.
        """
        if ref_diagram is None:
            base = train_diagrams if train_diagrams is not None else calib_diagrams
            ref_diagram = frechet_mean_diagram(base, frechet_q, rng=rng)
        scores = wasserstein_scores(calib_diagrams, ref_diagram, q)
        if w_calib is None:
            w_calib = np.ones(len(scores))
        return cls(ref_diagram, scores, w_calib, q)

    def radius(self, alpha: float, w_new=1.0):
        """Weighted-conformal Wasserstein-ball radius ``k`` at level ``1 - alpha``."""
        from .weighted_cp import weighted_conformal_radius
        return weighted_conformal_radius(self.scores, alpha, self.w_calib, w_new)

    def covers(self, diagram, alpha: float, w_new=1.0) -> bool:
        """Whether a test diagram lies in the ``1 - alpha`` Wasserstein ball."""
        k = self.radius(alpha, w_new)
        if not np.isfinite(k):
            return True
        return wasserstein_distance(diagram, self.ref, self.q) <= k
