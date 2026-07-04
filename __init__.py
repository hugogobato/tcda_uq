"""tcda_uq - Topological Causal Data Analysis: uncertainty quantification.

Estimands: TATE psi_d(t), CTATE tau_d(t,x), ITTE delta_{i,d}(t).

The estimand-specific methods live in :mod:`tcda_uq.uq.asymptotic` (confidence
bands) and :mod:`tcda_uq.uq.conformal` (prediction bands). A small, uniform
convenience layer is re-exported here (see :mod:`tcda_uq.api`)::

    from tcda_uq import tate_band, ctate_band, itte_band, evaluate_coverage

These are loaded lazily, so ``import tcda_uq`` itself stays cheap.
"""

__version__ = "0.0.1"

# Lazily re-exported from tcda_uq.api (keeps top-level import light).
_API_EXPORTS = {
    "tate_band",
    "ctate_band",
    "itte_band",
    "evaluate_coverage",
    "CoverageResult",
}

__all__ = ["__version__", *sorted(_API_EXPORTS)]


def __getattr__(name):
    if name in _API_EXPORTS:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _API_EXPORTS)
