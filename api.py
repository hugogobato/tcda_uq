"""Unified, non-breaking convenience layer over the estimand-specific UQ methods.

This module introduces **no new methodology**. It is a thin adapter that wraps
the existing per-method functions and classes behind one small, uniform surface,
so the three estimands can be *banded* and *evaluated* the same way::

    from tcda_uq import tate_band, ctate_band, itte_band, evaluate_coverage

    band = tate_band(cross_fit_result, method="mbb", d=1, alpha=0.10)   # confidence
    band = ctate_band(ctate_learner, x, d=1, alpha=0.10)                # confidence
    band = itte_band(itte_model, x, d=1, alpha=0.10)                    # prediction
    cov  = evaluate_coverage(band, truth)                               # -> CoverageResult

Each function returns a :class:`tcda_uq.metrics.Band` (or a list of them, one per
homology dimension, when ``d`` is left ``None``). The underlying reference API is
unchanged: :func:`~tcda_uq.uq.asymptotic.multiplier_bootstrap_band`,
:func:`~tcda_uq.uq.asymptotic.ctate_confidence_band`,
:class:`~tcda_uq.uq.conformal.ITTEConformal`, ... all keep their exact signatures.

Method aliases (case-insensitive):

  * TATE (:func:`tate_band`) --
    ``"multiplier" | "mbb" | "bootstrap"`` (multiplier bootstrap, TATE Cor 5.4),
    ``"pini_vantini" | "pv" | "iwt"`` (interval-wise band),
    ``"liebl_reimherr" | "lr" | "ffscb"`` (fast-and-fair band).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .metrics import Band
from .metrics.coverage import covers_simultaneous, mean_width

__all__ = [
    "tate_band",
    "ctate_band",
    "itte_band",
    "evaluate_coverage",
    "CoverageResult",
    "TATE_METHODS",
]

# canonical TATE method name -> accepted aliases
TATE_METHODS = {
    "multiplier_bootstrap": ("multiplier_bootstrap", "multiplier", "mbb", "bootstrap"),
    "pini_vantini": ("pini_vantini", "pv", "iwt"),
    "liebl_reimherr": ("liebl_reimherr", "lr", "ffscb"),
}
_TATE_ALIAS = {a: canon for canon, aliases in TATE_METHODS.items() for a in aliases}


def _resolve_tate(method: str):
    """Map a TATE method name/alias to its ``*_bands`` convenience function."""
    canon = _TATE_ALIAS.get(str(method).lower())
    if canon is None:
        raise ValueError(
            f"unknown TATE method {method!r}; choose one of "
            f"{sorted(_TATE_ALIAS)}"
        )
    # imported lazily so `import tcda_uq` stays cheap
    from .uq.asymptotic import (
        multiplier_bootstrap_bands,
        pini_vantini_bands,
        liebl_reimherr_bands,
    )

    return {
        "multiplier_bootstrap": multiplier_bootstrap_bands,
        "pini_vantini": pini_vantini_bands,
        "liebl_reimherr": liebl_reimherr_bands,
    }[canon]


def tate_band(
    cross_fit_result,
    *,
    method: str = "mbb",
    d: Optional[int] = None,
    alpha: float = 0.05,
    **kwargs,
) -> Union[Band, list]:
    """Confidence band(s) for the TATE ``psi_d(t)`` from a cross-fit result.

    Args:
        cross_fit_result: a :class:`~tcda_uq.estimators.CrossFitResult`
            (from :func:`~tcda_uq.estimators.cross_fit`) -- supplies the EIF
            process and the AIPW point estimate.
        method: which asymptotic band -- see :data:`TATE_METHODS` for aliases.
        d: homology dimension to return. ``None`` returns a list with one band
            per homology dimension.
        alpha: 1 - target simultaneous coverage.
        **kwargs: forwarded to the underlying band function (e.g. ``n_boot``,
            ``multiplier``, ``standardize`` for the bootstrap; ``backend`` /
            ``distribution`` for Liebl-Reimherr).

    Returns:
        A :class:`~tcda_uq.metrics.Band` (``kind="confidence"``) if ``d`` is given,
        else a list of them.
    """
    fn = _resolve_tate(method)
    bands = fn(cross_fit_result, alpha=alpha, **kwargs)
    return bands if d is None else bands[d]


def ctate_band(
    learner,
    x,
    *,
    method: str = "confidence",
    d: Optional[int] = None,
    alpha: float = 0.05,
    **kwargs,
) -> Union[Band, list]:
    """Pointwise-in-``x`` confidence band(s) for the CTATE ``tau_d(t, x)``.

    Args:
        learner: a fitted :class:`~tcda_uq.estimators.CTATEDRLearner`.
        x: covariate value at which to band ``tau_d(., x)`` (1-D).
        method: ``"confidence"`` (the only asymptotic CTATE band; covers the
            conditional *mean* curve). For the *prediction* counterpart of the
            CTATE (a draw at ``x``) use a covariate-adaptive conformal model with
            :func:`itte_band` / ``ctate_prediction_band``.
        d: homology dimension (``None`` -> list over dimensions).
        alpha: 1 - target simultaneous-in-t coverage.
        **kwargs: forwarded to
            :func:`~tcda_uq.uq.asymptotic.ctate_confidence_band`.

    Returns:
        A :class:`~tcda_uq.metrics.Band` (``kind="confidence"``) or a list of them.
    """
    m = str(method).lower()
    if m not in ("confidence", "conf", "asymptotic", "mean"):
        raise ValueError(
            "ctate_band only produces the asymptotic *confidence* band "
            f"(method='confidence'); got method={method!r}. The prediction "
            "counterpart at x is covariate-adaptive conformal -- fit an "
            "AdaptiveITTEConformal and call itte_band(...) / ctate_prediction_band(...)."
        )
    from .uq.asymptotic import ctate_confidence_bands

    bands = ctate_confidence_bands(learner, x, alpha=alpha, **kwargs)
    return bands if d is None else bands[d]


def itte_band(
    model,
    x,
    *,
    d: Optional[int] = None,
    alpha: float = 0.1,
) -> Union[Band, list]:
    """Conformal prediction band(s) for the ITTE ``delta_d(., x)`` at a single ``x``.

    Args:
        model: a fitted :class:`~tcda_uq.uq.conformal.ITTEConformal` or
            :class:`~tcda_uq.uq.conformal.AdaptiveITTEConformal`.
        x: covariate value (1-D) at which to band ``delta_d(., x)``.
        d: homology dimension (``None`` -> list over dimensions via ``model.bands``).
        alpha: 1 - target simultaneous coverage.

    Returns:
        A :class:`~tcda_uq.metrics.Band` (``kind="prediction"``) or a list of them.
    """
    if d is None:
        return model.bands(x, alpha)
    return model.band(x, alpha, d)


@dataclass
class CoverageResult:
    """Result of :func:`evaluate_coverage`: empirical coverage + mean width.

    Attributes:
        coverage: simultaneous coverage fraction (all-``t``) over the target(s).
        mean_width: mean pointwise band width (``+inf`` if any bound is unbounded).
        n_targets: number of target curves evaluated.
        level: the band's nominal ``1 - alpha`` (if known), for a gap read-out.
        kind: ``"confidence"`` / ``"prediction"`` (carried from the band).
    """

    coverage: float
    mean_width: float
    n_targets: int
    level: Optional[float] = None
    kind: Optional[str] = None

    @property
    def gap(self) -> Optional[float]:
        """Empirical coverage minus nominal level (``None`` if level unknown)."""
        return None if self.level is None else self.coverage - self.level

    def __repr__(self):
        lvl = "?" if self.level is None else f"{self.level:.3f}"
        gap = "" if self.gap is None else f" gap={self.gap:+.3f}"
        return (
            f"CoverageResult(coverage={self.coverage:.3f} @ level={lvl}{gap}, "
            f"mean_width={self.mean_width:.4g}, n={self.n_targets}"
            f"{'' if self.kind is None else f', {self.kind}'})"
        )


def _unpack_band(band):
    """Return ``(lower, upper, level, kind)`` from a Band or an ``(lo, hi[, ctr])`` tuple."""
    if isinstance(band, Band):
        return band.lower, band.upper, band.level, band.kind
    seq = tuple(band)
    if len(seq) < 2:
        raise TypeError(
            "band must be a tcda_uq.metrics.Band or an (lower, upper[, center]) tuple"
        )
    lower = np.asarray(seq[0], dtype=float)
    upper = np.asarray(seq[1], dtype=float)
    return lower, upper, None, None


def evaluate_coverage(band, truth, *, level: Optional[float] = None) -> CoverageResult:
    """Simultaneous coverage + mean width of a band against ground-truth curve(s).

    Works uniformly for every band this library produces (confidence or
    prediction), because all of them expose ``lower`` / ``upper`` over ``tseq``.

    Args:
        band: a :class:`~tcda_uq.metrics.Band`, or a raw ``(lower, upper)`` /
            ``(lower, upper, center)`` tuple (e.g. from ``model.band_bounds``).
            ``lower``/``upper`` may be ``[resolution]`` (one band) or
            ``[m, resolution]`` (a batch of per-unit bands).
        truth: target curve(s). ``[resolution]`` (single) or ``[k, resolution]``
            (a batch, e.g. many ITTE draws). Broadcasts against a batched band
            row-by-row when both are 2-D with matching leading dim.
        level: override the nominal ``1 - alpha`` (defaults to the band's own).

    Returns:
        :class:`CoverageResult`.
    """
    lower, upper, lvl, kind = _unpack_band(band)
    truth = np.asarray(truth, dtype=float)
    covered = np.atleast_1d(covers_simultaneous(lower, upper, truth))
    coverage = float(np.mean(covered))
    mw = mean_width(lower, upper)
    return CoverageResult(
        coverage=coverage,
        mean_width=mw,
        n_targets=int(covered.size),
        level=level if level is not None else lvl,
        kind=kind,
    )
