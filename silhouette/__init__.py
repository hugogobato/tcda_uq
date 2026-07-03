"""Persistence -> silhouette functionals.

Three entry points:
  * :func:`compute_silhouette`          -- from precomputed persistence diagrams
  * :func:`silhouette_from_pointcloud`  -- Alpha-complex filtration (ORBIT)
  * :func:`silhouette_from_image`       -- cubical (lower-star) filtration (SARS-CoV-2 CT)

plus :func:`diagrams_from_pointcloud`, the raw persistence diagrams used by
diagram-space scores and topology-stability diagnostics.
"""

from .core import (
    power_weight,
    compute_silhouette,
    diagrams_from_pointcloud,
    silhouette_from_pointcloud,
    silhouette_from_image,
)

__all__ = [
    "power_weight",
    "compute_silhouette",
    "diagrams_from_pointcloud",
    "silhouette_from_pointcloud",
    "silhouette_from_image",
]
