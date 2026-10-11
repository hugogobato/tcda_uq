"""Input checks shared by the finite-grid inference implementations."""

import numpy as np


def validate_alpha(alpha):
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and strictly between 0 and 1")


def validate_bootstrap(n_boot, multiplier):
    if isinstance(n_boot, (bool, np.bool_)) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    if multiplier not in ("gaussian", "rademacher", "mammen"):
        raise ValueError("multiplier must be 'gaussian', 'rademacher', or 'mammen'")


def validate_process(influence, estimate, tseq=None):
    phi = np.asarray(influence, dtype=float)
    center = np.asarray(estimate, dtype=float)
    if phi.ndim != 2 or min(phi.shape) < 1 or not np.isfinite(phi).all():
        raise ValueError("influence must be a finite nonempty [n, resolution] array")
    if center.shape != (phi.shape[1],) or not np.isfinite(center).all():
        raise ValueError("estimate must be a finite [resolution] array")
    if tseq is not None:
        grid = np.asarray(tseq, dtype=float)
        if grid.shape != center.shape or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
            raise ValueError("tseq must be finite, strictly increasing, and match resolution")
    return phi - phi.mean(axis=0, keepdims=True), center


def pointwise_scale(phi, variance=None):
    var = np.mean(phi ** 2, axis=0) if variance is None else np.asarray(variance, dtype=float)
    if var.shape != (phi.shape[1],) or not np.isfinite(var).all() or np.any(var < 0):
        raise ValueError("variance must be finite, nonnegative, and match resolution")
    return np.sqrt(var)
