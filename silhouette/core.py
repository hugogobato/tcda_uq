"""Plain normalized persistence tents, with an explicit legacy scale option.

The default is the manuscript's ``sum(lifetime**r * tent) / sum(lifetime**r)``.
GUDHI multiplies this quantity by sqrt(2); the correction is made here once.
The forward W1 stability theorem applies for r >= 1. A silhouette is not an
injective representation of a diagram, so it gives no reverse distance bound.
"""

from __future__ import annotations

import numpy as np
from gudhi.representations import Silhouette


def _validate_exponent(r, allow_experimental_r=False):
    r = float(r)
    if not np.isfinite(r) or r < 0:
        raise ValueError("r must be finite and nonnegative")
    if r < 1 and not allow_experimental_r:
        raise ValueError(
            "r must be >= 1 for the supported stability theorem; "
            "set allow_experimental_r=True for a legacy experimental exponent"
        )
    return r


def _finite_diagram(diag):
    """Remove essential and zero-persistence bars; retain multiplicities."""
    diag = np.asarray(diag, dtype=float)
    if diag.size == 0:
        if diag.ndim > 2 or (diag.ndim == 2 and diag.shape[1] != 2):
            raise ValueError("each diagram must have shape (k, 2)")
        return np.empty((0, 2))
    if diag.ndim != 2 or diag.shape[1] != 2:
        raise ValueError("each diagram must have shape (k, 2)")
    if np.isnan(diag).any():
        raise ValueError("persistence diagrams must not contain NaN")
    # The public convention drops essential bars in every requested degree.
    diag = diag[np.isfinite(diag).all(axis=1)]
    if np.any(diag[:, 1] < diag[:, 0]):
        raise ValueError("death must be at least birth in every finite bar")
    return diag[diag[:, 1] > diag[:, 0]]


def power_weight(point, r: float = 3.0, *, allow_experimental_r=False) -> float:
    """Power weighting of a persistence point: ``|death - birth|**r``."""
    r = _validate_exponent(r, allow_experimental_r)
    birth, death = point
    return np.abs(death - birth) ** r


def compute_silhouette(
    diags, interval=(0.0, 0.2), r: float = 3.0, resolution: int = 100,
    *, normalization="plain", allow_experimental_r=False,
):
    """Power-weighted silhouette of a list of persistence diagrams.

    Args:
        diags: list of ``(k, 2)`` arrays, one persistence diagram per homology dim.
        interval: sample range ``[t_min, t_max]`` of the silhouette.
        r: power-weight exponent, >= 1 by default.
        resolution: number of grid points.
        normalization: ``"plain"`` (default) or ``"legacy_gudhi"`` (sqrt(2)
            times the plain tents). Do not divide the default output again.
        allow_experimental_r: opt into r in [0, 1), outside the stated theorem.

    Returns:
        ``(n_hom_dim, resolution)`` array of silhouette values.
    """
    r = _validate_exponent(r, allow_experimental_r)
    interval = np.asarray(interval, dtype=float)
    if interval.shape != (2,) or not np.isfinite(interval).all() or interval[1] <= interval[0]:
        raise ValueError("interval must contain two finite, increasing endpoints")
    if isinstance(resolution, bool) or int(resolution) != resolution or resolution < 2:
        raise ValueError("resolution must be an integer >= 2")
    resolution = int(resolution)
    if normalization not in ("plain", "legacy_gudhi"):
        raise ValueError("normalization must be 'plain' or 'legacy_gudhi'")
    diagrams = [_finite_diagram(diag) for diag in diags]
    out = np.zeros((len(diagrams), resolution))
    for i, diag in enumerate(diagrams):
        if not len(diag):
            continue
        lifetimes = diag[:, 1] - diag[:, 0]
        scale = float(lifetimes.max())
        if not np.isfinite(scale):
            raise ValueError("finite bar lifetimes must be representable as finite floats")
        # Dividing all lifetimes by their maximum cancels in normalized weights.
        # It avoids 0/0 from underflow for tiny diagrams without an epsilon floor.
        silhouette = Silhouette(
            weight=lambda x: ((x[1] - x[0]) / scale) ** r,
            resolution=resolution,
            sample_range=interval.tolist(),
            keep_endpoints=True,
        )
        out[i] = silhouette.fit_transform([diag])[0]
    if normalization == "plain":
        out /= np.sqrt(2.0)
    return out


def diagrams_from_pointcloud(points, homology_dims=(0, 1)):
    """Alpha-complex persistence diagrams of a point cloud (radii, essential dropped).

    The intermediate the silhouette is built from, exposed so diagram-space
    scores and stability diagnostics can access the diagram directly. Alpha
    filtration values are squared radii, converted to radii (``sqrt``).

    Args:
        points: ``(m, d)`` point cloud.
        homology_dims: homology dimensions to keep (default H0, H1).

    Returns:
        list of ``(k, 2)`` birth-death arrays, one per requested homology dim.
    """
    import gudhi as gd

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 1 or not np.isfinite(points).all():
        raise ValueError("points must be a finite (m, d) array with d >= 1")
    if len(points) == 0:
        return [np.empty((0, 2)) for _ in homology_dims]
    alpha = gd.AlphaComplex(points=points)
    st = alpha.create_simplex_tree()
    st.compute_persistence()

    diags = []
    for hom_dim in homology_dims:
        diag = st.persistence_intervals_in_dimension(hom_dim)
        if diag.size == 0:
            diags.append(np.empty((0, 2)))
            continue
        diag = _finite_diagram(diag)
        # alpha filtration values are squared radii -> use radius
        diag = np.sqrt(np.maximum(diag, 0.0))
        diags.append(diag)
    return diags


def silhouette_from_pointcloud(
    points,
    interval=(0.0, 0.2),
    r: float = 3.0,
    resolution: int = 100,
    homology_dims=(0, 1),
    *,
    normalization="plain",
    allow_experimental_r=False,
):
    """Alpha-complex silhouette of a point cloud (ORBIT modality).

    Builds an alpha complex, extracts persistence, converts squared alpha
    radii to radii (``sqrt``), and returns the power-weighted silhouette.

    Args:
        points: ``(m, d)`` point cloud.
        homology_dims: homology dimensions to keep (default H0, H1).
    """
    diags = diagrams_from_pointcloud(points, homology_dims=homology_dims)
    return compute_silhouette(
        diags, interval=interval, r=r, resolution=resolution,
        normalization=normalization, allow_experimental_r=allow_experimental_r,
    )


def silhouette_from_image(
    image,
    interval=(0.0, 1.0),
    r: float = 1.0,
    resolution: int = 100,
    homology_dims=(0,),
    max_filtration=None,
    *,
    normalization="plain",
    allow_experimental_r=False,
):
    """Cubical (sub-level / lower-star) silhouette of a grayscale image (SARS-CoV-2 CT).

    Args:
        image: 2-D array of pixel intensities (e.g. normalised to ``[0, 1]``).
        homology_dims: homology dimensions to keep (default H0 only).
        max_filtration: if given, essential death times in every degree are
            replaced by this finite cutoff. This changes the prespecified map.
            Otherwise essential bars are excluded in every degree.
        r: exponent >= 1, with default 1. Legacy r < 1 requires the explicit
            experimental opt-in and has no stated W1 stability guarantee.
    """
    import gudhi as gd

    image = np.asarray(image, dtype=float)
    if image.ndim != 2 or not image.size or not np.isfinite(image).all():
        raise ValueError("image must be a nonempty finite 2-D array")
    if max_filtration is not None and (
        not np.isfinite(max_filtration) or max_filtration < image.max()
    ):
        raise ValueError("max_filtration must be finite and at least the maximum intensity")
    cub = gd.CubicalComplex(vertices=image)
    cub.compute_persistence()

    diags = []
    for hom_dim in homology_dims:
        diag = cub.persistence_intervals_in_dimension(hom_dim)
        inf_mask = np.isinf(diag).any(axis=1) if diag.size else np.zeros(0, dtype=bool)
        if max_filtration is None:
            diag = diag[~inf_mask]  # drop essential class
        else:
            diag = diag.copy()
            diag[np.isinf(diag)] = max_filtration
        diags.append(diag)
    return compute_silhouette(
        diags, interval=interval, r=r, resolution=resolution,
        normalization=normalization, allow_experimental_r=allow_experimental_r,
    )
