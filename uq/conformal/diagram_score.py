"""Diagram-space (Wasserstein) nonconformity scores (Phase 6.2, optional).

The default Phase-4 band scores in **silhouette space** (:mod:`.functional_cp`): cheap,
but the silhouette blurs the *number* of homological features (concept note section 5.1).
The intrinsic alternative scores in **diagram space**:

    S_i  =  W_q( D_{i,d},  D_hat_{A_i} ),

the ``q``-Wasserstein distance between unit ``i``'s degree-``d`` persistence diagram and a
*predicted* diagram ``D_hat_a`` for its arm. Because a Wasserstein distance is a **scalar**,
the causal reweighting (Layer 2) and the whole split-conformal machinery drop in unchanged
-- only Layer 1 (curve -> scalar) is swapped for (diagram -> scalar). The resulting radius
``k`` defines a **Wasserstein ball** ``{ D : W_q(D, D_hat_a) <= k }`` as a marginal prediction set when the score is frozen on independent training
and weights are actual arm-to-target ratios. Learned weights require TV qualification.

Trade-off (why this is *optional*): more faithful to the geometry, but diagram-template optimization
is nonconvex and the supplied arithmetic update has no optimality guarantee and the pairwise Wasserstein solves are
``O((n+m)^3)`` Hungarian assignments -- heavy next to the vectorized silhouette sup-norm.
Provided as a benchmark for the silhouette-space band and to exercise the Theorem-5.3
bridge (:mod:`.silhouette_bridge`) against the true diagram metric.

Self-contained: the ``q``-Wasserstein distance and a legacy heuristic diagram-template
update are implemented on top of :func:`scipy.optimize.linear_sum_assignment`, so the
core install needs no optional-transport (`ot`) dependency.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _clean(D):
    """Keep every strictly positive finite bar; discard essential +infinity deaths."""
    D = np.asarray(D, dtype=float)
    if D.size == 0:
        return np.empty((0, 2))
    if D.ndim != 2 or D.shape[1] != 2 or np.isnan(D).any() or not np.isfinite(D[:, 0]).all() or np.isneginf(D[:, 1]).any():
        raise ValueError("diagrams must have shape [k,2], finite births, and finite or +infinity deaths")
    D = D[np.isfinite(D[:, 1])]
    if np.any(D[:,1] < D[:,0]):
        raise ValueError("diagram deaths must be at least their births")
    return D[D[:, 1] > D[:, 0]]


def _assignment_costs(D1, D2, q):
    """Scaled legal distances and powered costs, with truly forbidden edges."""
    n, m = len(D1), len(D2)
    scale = max(float(np.max(np.abs(D1))) if n else 0.,
                float(np.max(np.abs(D2))) if m else 0.)
    if scale == 0:
        scale = 1.
    distances = np.full((n+m,n+m), np.inf)
    if n and m:
        with np.errstate(over="ignore"):
            raw = np.abs(D1[:,None,:]-D2[None,:,:])
        # Direct subtraction preserves small distances at a large common offset.
        # Opposite-sign overflow is safely calculated after normalizing coordinates.
        norm = np.divide(raw, scale)
        overflow = ~np.isfinite(raw)
        if overflow.any():
            fallback = np.abs(D1[:,None,:]/scale-D2[None,:,:]/scale)
            norm[overflow] = fallback[overflow]
        if np.any((raw > 0) & (norm == 0)):
            raise ValueError("diagram distance underflow after scaling; numerical range is too wide")
        distances[:n,:m] = norm.max(axis=2)
    for D, rows, columns in ((D1,np.arange(n),m+np.arange(n)),
                             (D2,n+np.arange(m),np.arange(m))):
        if not len(D):
            continue
        with np.errstate(over="ignore"):
            lifetime = D[:,1]-D[:,0]
        finite = np.isfinite(lifetime)
        diag = np.empty(len(D))
        diag[finite] = lifetime[finite]/scale/2
        diag[~finite] = (D[~finite,1]/scale-D[~finite,0]/scale)/2
        if np.any(diag == 0):
            raise ValueError("positive diagonal distance underflow after scaling")
        distances[rows,columns] = diag
    distances[n:,m:] = 0.
    legal = np.isfinite(distances)
    cost_scale = float(distances[legal].max()) if legal.any() else 1.
    if cost_scale == 0:
        cost_scale = 1.
    costs = np.full_like(distances,np.inf)
    normalized = distances[legal]/cost_scale
    if np.any((distances[legal] > 0) & (normalized == 0)):
        raise ValueError("normalized assignment costs underflow; numerical range is too wide")
    costs[legal] = normalized**q
    if np.any((normalized > 0)&(costs[legal] == 0)):
        raise ValueError("powered Wasserstein costs underflow; q or numerical range is too large")
    return costs, distances, scale


def wasserstein_distance(D1, D2, q: float = 1.0):
    """q-Wasserstein metric with L-infinity ground distance and diagonal matching.

    Essential +infinity deaths are discarded as a fixed preprocessing rule;
    every strictly positive finite lifetime is retained. Scaled costs prevent
    powering overflow. If the numerical range cannot represent the powered
    objective or final distance, fail explicitly instead of returning a smaller
    distance. Forbidden assignment edges use infinity, not a finite penalty.
    """
    if not np.isfinite(q) or q < 1:
        raise ValueError("Wasserstein order q must be finite and at least 1")
    D1, D2 = _clean(D1), _clean(D2)
    if not len(D1) and not len(D2):
        return 0.
    costs, distances, scale = _assignment_costs(D1,D2,q)
    rows, columns = linear_sum_assignment(costs)
    selected = distances[rows,columns]
    peak = float(selected.max())
    if peak == 0:
        return 0.
    norm = float(np.sum((selected/peak)**q)**(1./q))
    value = np.longdouble(scale)*np.longdouble(peak)*np.longdouble(norm)
    if value > np.finfo(float).max:
        raise OverflowError("Wasserstein distance exceeds the finite float range")
    if 0 < value < np.nextafter(0.,1.):
        raise ValueError("Wasserstein distance is below the positive float range")
    return float(value)


def frechet_mean_diagram(diagrams, q: float = 2.0, *, n_iter: int = 20, tol: float = 1e-6,
                         rng=None):
    """Legacy heuristic template learned by matching and coordinate averaging.

    Arithmetic coordinate averages under L-infinity matching costs are not
    generally exact Frechet minimizers, and this routine does not instantiate
    Turner's optimizer guarantees. The legacy name is retained for compatibility.
    Split-conformal validity needs only a measurable template frozen on independent
    training, not an optimal barycenter. This heuristic can return an empty template.
    """
    if not np.isfinite(q) or q < 1:
        raise ValueError("q must be finite and at least 1")
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
            C, _, _ = _assignment_costs(mu,D,q)
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
                new_mu.append(np.sum(np.asarray(pts)/len(pts),axis=0))
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
    mid = p[0]/2 + p[1]/2
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
                independent ``train_diagrams``. Reusing calibration to fit the center is rejected.
            train_diagrams: disjoint diagrams to build the Frechet mean from.
            w_calib: calibration weights ``[n]`` (default uniform / unweighted).
            frechet_q: order of the Frechet-mean barycenter.
        """
        if ref_diagram is None:
            if train_diagrams is None:
                raise ValueError("ref_diagram or independent train_diagrams is required; calibration cannot fit its own center")
            base = train_diagrams
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
