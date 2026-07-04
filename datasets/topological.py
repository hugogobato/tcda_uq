"""Topological (Level-B) causal DGP: point clouds with X,A-controlled homology (Phase 6 / 7A.2 seed).

The tri-oracle harness (:mod:`.simulation`) generates silhouette *curves* directly, which is
perfect for testing the UQ math but **bypasses the topology map** ``Y -> D -> phi``. Several
Phase-6 claims are specifically *about* that map:

  * the power exponent ``r`` (Phase 6.3) only exists inside the silhouette of a real diagram;
  * the Theorem-5.3 bridge (Phase 6.1) relates a *diagram* ``W_1`` perturbation to a silhouette
    sup-norm perturbation;
  * the diagram-space Wasserstein score (Phase 6.2) needs real diagrams.

This module supplies the smallest honest **topological** DGP with a causal structure: each unit's
potential outcome is a **point cloud** whose number of loops (``H_1`` features) is driven by
``(X, A)``, so the treatment genuinely changes the topology. Running the actual alpha-complex
persistence -> silhouette pipeline on both potential outcomes gives per-unit oracle
``(phi^0, phi^1)`` and hence oracle ITTE / CTATE / TATE at the *topological* level -- the same
tri-oracle contract as :class:`~tcda_uq.datasets.TriOracleSimulation`, but with the topology in
the loop. It is the seed of the Phase-7A Level-B UQ-stress DGP.

Construction (per unit ``i``, arm ``a``):
  * covariates ``X_i in R^p`` from a two-component Gaussian mixture (as in the tri-oracle);
  * loop count ``L^a_i = clip( round( base + a * effect + slope * (X_i . w) ), 0, max_loops )`` --
    the treatment adds ``effect`` loops on average, modulated by a covariate direction ``w`` (this
    is what makes the CTATE depend on ``X``);
  * point cloud: ``L^a_i`` well-separated circles of radius ``radius`` with ``pts_per_loop``
    points each plus isotropic Gaussian jitter ``noise`` (plus a few ``background`` points). The
    stochastic sampling is the **aleatoric** spread the ITTE prediction band must cover.

Persistence diagrams are computed once per (unit, arm) and cached on the sample, so an experiment
can re-silhouette at many ``r`` values (Phase 6.3) or score in diagram space (Phase 6.2) without
recomputing homology.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..silhouette import compute_silhouette, diagrams_from_pointcloud

_MU1 = np.array([1.0, 0.6, -0.7, 2.2, -1.0])
_MU2 = np.array([0.4, -0.4, -0.6, 3.3, 3.0])
_BETA = np.array([-0.5, -0.1, 0.6, 0.1, 0.1])


@dataclass
class TopologicalSample:
    """One realised topological sample with cached diagrams and the tri-oracle truths."""

    tseq: np.ndarray
    X: np.ndarray                  # [n, p]
    A: np.ndarray                  # [n]
    propensity: np.ndarray         # [n]
    diagrams: list                 # diagrams[i][a] = list over hom dim of (k, 2) arrays
    potential_outcomes: np.ndarray  # [n, 2, n_hom, res] silhouettes (phi^0, phi^1)
    oracle_itte: np.ndarray        # [n, n_hom, res]  delta_i = phi^1 - phi^0
    homology_dims: tuple
    interval: tuple
    r: float

    @property
    def observed(self):
        """Observed triplet ``(phi, A, X)`` -- the factual silhouettes."""
        phi = self.potential_outcomes[np.arange(len(self.A)), self.A]
        return (phi, self.A, self.X)

    @property
    def observed_diagrams(self):
        """Per-unit factual diagrams ``[i] -> list over hom dim`` (arm ``A_i``)."""
        return [self.diagrams[i][self.A[i]] for i in range(len(self.A))]


class TopologicalCausalSimulation:
    """Point-cloud causal DGP with ``(X, A)``-controlled loops (Level-B topological oracle).

    Args:
        n_cov: covariate dimension.
        homology_dims: homology dims to summarize (default ``(0, 1)``).
        resolution: silhouette grid size.
        interval: silhouette domain ``[t_min, t_max]`` (radius units).
        r: power-weight exponent used for the *stored* silhouettes (re-silhouette at other
            ``r`` from the cached diagrams via :meth:`silhouettes_at_r`).
        base_loops: control-arm baseline loop count (before the ``X`` tilt).
        loop_effect: extra loops the treatment adds on average (the topological effect).
        loop_slope: strength of the ``X``-dependence of the loop count (drives the CTATE).
        max_loops: cap on the loop count.
        pts_per_loop: points sampled per circle.
        radius: circle radius (and inter-circle spacing scale).
        noise: isotropic Gaussian jitter on the sampled points.
        background: number of uniform background points (adds ``H_0`` noise).
        prop_scale: propensity-logit scale (>1 => weaker overlap; matches the tri-oracle knob).
        radius_tail_df: **diagram tail-heaviness knob** (Phase 7A.3, topological). ``None``
            (default) keeps every loop at the fixed ``radius`` -> light, concentrated
            persistence. When set (a Student-t degrees-of-freedom ``> 2``), each loop's radius
            is inflated by ``1 + radius_tail_scale * |t|`` with ``t`` a unit-variance
            ``df``-Student-t draw, so a few loops are far more persistent than the rest: the
            persistence diagram (hence the silhouette) grows a **heavy tail**. Smaller ``df`` =>
            heavier tail. Off by default so existing samples are byte-for-byte unchanged.
        radius_tail_scale: amplitude of the heavy-tail radius inflation (only used when
            ``radius_tail_df`` is set).
        beta: propensity coefficients.
        seed: seeds the fixed model directions (``w`` for the loop tilt, ``beta`` default).
    """

    def __init__(
        self,
        n_cov: int = 3,
        homology_dims=(0, 1),
        resolution: int = 60,
        interval=(0.0, 1.0),
        r: float = 3.0,
        base_loops: int = 2,
        loop_effect: int = 2,
        loop_slope: float = 1.2,
        max_loops: int = 6,
        pts_per_loop: int = 40,
        radius: float = 1.0,
        noise: float = 0.08,
        background: int = 6,
        prop_scale: float = 1.0,
        radius_tail_df: float | None = None,
        radius_tail_scale: float = 0.6,
        mu1=None,
        mu2=None,
        beta=None,
        sigma2: float = 0.5,
        seed: int = 0,
    ):
        self.n_cov = n_cov
        self.homology_dims = tuple(homology_dims)
        self.n_hom_dim = len(self.homology_dims)
        self.resolution = resolution
        self.interval = interval
        self.r = r
        self.base_loops = base_loops
        self.loop_effect = loop_effect
        self.loop_slope = loop_slope
        self.max_loops = max_loops
        self.pts_per_loop = pts_per_loop
        self.radius = radius
        self.noise = noise
        self.background = background
        self.prop_scale = prop_scale
        if radius_tail_df is not None and radius_tail_df <= 2:
            raise ValueError("radius_tail_df must be > 2 so the loop radii have finite variance")
        self.radius_tail_df = radius_tail_df
        self.radius_tail_scale = float(radius_tail_scale)

        self.mu1 = np.asarray(mu1 if mu1 is not None else _MU1[:n_cov], dtype=float)
        self.mu2 = np.asarray(mu2 if mu2 is not None else _MU2[:n_cov], dtype=float)
        self.Sigma = np.eye(n_cov) * sigma2
        self.beta = np.asarray(beta if beta is not None else _BETA[:n_cov], dtype=float)
        self.tseq = np.linspace(interval[0], interval[1], resolution)

        init = np.random.default_rng(seed)
        self.w_loops = init.normal(size=n_cov)
        self.w_loops /= np.linalg.norm(self.w_loops)
        self.EX = 0.5 * (self.mu1 + self.mu2)
        self._loop_center = float(self.EX @ self.w_loops)

    # ------------------------------------------------------------------ structure
    def _propensity(self, X):
        y = X @ self.beta
        if self.n_cov >= 3:
            y = y + 0.5 * X[:, 1] * X[:, 2] - 0.7 * X[:, 0] * X[:, 2]
        return 1.0 / (1.0 + np.exp(-self.prop_scale * y))

    def _n_loops(self, X, a):
        """Loop count ``L^a(X)`` per unit (integer, clipped to ``[0, max_loops]``)."""
        tilt = self.loop_slope * (X @ self.w_loops - self._loop_center)
        raw = self.base_loops + a * self.loop_effect + tilt
        return np.clip(np.rint(raw), 0, self.max_loops).astype(int)

    def _loop_radii(self, n_loops, rng):
        """Per-loop radii. Fixed at ``radius`` unless the heavy-tail knob is on.

        When ``radius_tail_df`` is ``None`` (default) no random numbers are drawn, so the
        overall RNG stream -- and hence every previously generated sample -- is unchanged.
        """
        if self.radius_tail_df is None or n_loops == 0:
            return np.full(int(n_loops), self.radius)
        t = rng.standard_t(self.radius_tail_df, size=int(n_loops))
        t = t * np.sqrt((self.radius_tail_df - 2.0) / self.radius_tail_df)   # unit variance
        return self.radius * (1.0 + self.radius_tail_scale * np.abs(t))

    def _point_cloud(self, n_loops, rng):
        """Sample ``n_loops`` jittered circles (+ background) as a 2-D point cloud."""
        n_loops = int(n_loops)
        radii = self._loop_radii(n_loops, rng)
        pts = []
        # place circle centers on a coarse grid so loops stay separable even when the
        # heavy-tail knob inflates some radii (spacing tracks the largest circle).
        max_mult = float(radii.max() / self.radius) if n_loops else 1.0
        spacing = 3.0 * self.radius * max_mult
        for k in range(n_loops):
            cx, cy = (k % 3) * spacing, (k // 3) * spacing
            theta = rng.uniform(0, 2 * np.pi, self.pts_per_loop)
            circle = np.column_stack([cx + radii[k] * np.cos(theta),
                                      cy + radii[k] * np.sin(theta)])
            pts.append(circle + rng.normal(scale=self.noise, size=circle.shape))
        if self.background > 0:
            span = spacing * 3
            pts.append(rng.uniform(-span, span, size=(self.background, 2)))
        if not pts:                      # zero loops, zero background -> a tiny blob
            return rng.normal(scale=self.noise, size=(3, 2))
        return np.vstack(pts)

    def _diagrams(self, n_loops, rng):
        return diagrams_from_pointcloud(self._point_cloud(n_loops, rng),
                                        homology_dims=self.homology_dims)

    def _silhouette(self, diags, r=None):
        return compute_silhouette(diags, interval=self.interval,
                                  r=self.r if r is None else r,
                                  resolution=self.resolution)

    def _sample_covariates(self, n, rng):
        """Draw covariates from the fixed two-component mixture."""
        n1 = n // 2
        X1 = rng.multivariate_normal(self.mu1, self.Sigma, size=n1)
        X2 = rng.multivariate_normal(self.mu2, self.Sigma, size=n - n1)
        return np.vstack([X1, X2])

    # ------------------------------------------------------------------ Monte Carlo oracles
    def sample_potential_outcome(self, X, a: int, rng=None, r=None):
        """Sample fresh topological potential-outcome silhouettes at fixed ``X``.

        The loop count is deterministic given ``(X, a)``; the point-cloud sampling,
        persistence diagram, and silhouette are re-drawn. This is the primitive used
        by the Level-B Monte Carlo truth helpers in Phase 7A.2.

        Returns:
            Array ``[len(X), n_hom, resolution]``.
        """
        if a not in (0, 1):
            raise ValueError("a must be 0 or 1")
        rng = np.random.default_rng(rng)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        loops = self._n_loops(X, a)
        out = np.empty((X.shape[0], self.n_hom_dim, self.resolution))
        for i, n_loops in enumerate(loops):
            out[i] = self._silhouette(self._diagrams(n_loops, rng), r=r)
        return out

    def true_ctate_mc(self, X, n_mc: int = 32, rng=None, r=None):
        """Monte Carlo oracle for ``E[phi^1 - phi^0 | X]`` in the topological DGP.

        Unlike :class:`TriOracleSimulation`, the Level-B DGP runs through random
        point clouds and persistence diagrams, so the conditional mean is not in
        closed form. This method averages fresh potential outcomes at fixed ``X``.
        It is intended for validation and coverage studies, not for fitting.
        """
        if n_mc < 1:
            raise ValueError("n_mc must be >= 1")
        rng = np.random.default_rng(rng)
        X = np.atleast_2d(np.asarray(X, dtype=float))
        acc = np.zeros((X.shape[0], self.n_hom_dim, self.resolution))
        seeds = rng.integers(0, 1 << 31, size=n_mc)
        for seed in seeds:
            phi1 = self.sample_potential_outcome(X, 1, rng=int(seed), r=r)
            phi0 = self.sample_potential_outcome(X, 0, rng=int(seed) + 1, r=r)
            acc += phi1 - phi0
        return acc / n_mc

    def true_tate_mc(self, n_x: int = 512, n_mc: int = 8, rng=None, r=None):
        """Monte Carlo oracle for the marginal topological effect ``E_X tau(X)``.

        Args:
            n_x: number of covariate draws used for the outer expectation.
            n_mc: number of fresh point-cloud replicates per covariate value.

        Returns:
            Array ``[n_hom, resolution]``.
        """
        if n_x < 1:
            raise ValueError("n_x must be >= 1")
        rng = np.random.default_rng(rng)
        X = self._sample_covariates(n_x, rng)
        return self.true_ctate_mc(X, n_mc=n_mc, rng=rng, r=r).mean(axis=0)

    # ------------------------------------------------------------------ sampling
    def sample(self, n, rng=None) -> TopologicalSample:
        rng = np.random.default_rng(rng)
        X = self._sample_covariates(n, rng)
        pi = self._propensity(X)
        A = rng.binomial(1, pi)

        L0 = self._n_loops(X, 0)
        L1 = self._n_loops(X, 1)

        diagrams, po = [], np.empty((n, 2, self.n_hom_dim, self.resolution))
        for i in range(n):
            d0 = self._diagrams(L0[i], rng)
            d1 = self._diagrams(L1[i], rng)
            diagrams.append([d0, d1])
            po[i, 0] = self._silhouette(d0)
            po[i, 1] = self._silhouette(d1)

        oracle_itte = po[:, 1] - po[:, 0]
        return TopologicalSample(
            tseq=self.tseq, X=X, A=A, propensity=pi, diagrams=diagrams,
            potential_outcomes=po, oracle_itte=oracle_itte,
            homology_dims=self.homology_dims, interval=self.interval, r=self.r,
        )

    def silhouettes_at_r(self, sample: TopologicalSample, r: float):
        """Re-silhouette a sample's cached diagrams at exponent ``r`` (Phase 6.3, no re-homology).

        Returns ``potential_outcomes`` ``[n, 2, n_hom, res]`` and ``oracle_itte``
        ``[n, n_hom, res]`` at the requested ``r`` -- the only thing that changes when ``r``
        moves is the power weighting of the *same* diagrams.
        """
        n = len(sample.A)
        po = np.empty((n, 2, self.n_hom_dim, self.resolution))
        for i in range(n):
            po[i, 0] = self._silhouette(sample.diagrams[i][0], r=r)
            po[i, 1] = self._silhouette(sample.diagrams[i][1], r=r)
        return po, po[:, 1] - po[:, 0]
