"""Conditional asymptotic mean-effect inference and native testing.

Gaussian multiplier bands require the revised process and bias assumptions.
IWT returns p-values; legacy IWT envelope helpers remain explicitly deprecated.
Liebl-Reimherr benchmarks require additional smooth-process assumptions.
"""

from .covariance import (
    eif_covariance,
    eif_pointwise_variance,
    eif_pointwise_sd,
    eif_correlation,
)
from .multiplier_bootstrap import (
    multiplier_bootstrap_band,
    multiplier_bootstrap_bands,
    topological_effect_test,
)
from .pini_vantini import (
    iwt_pvalues,
    IWTResult,
    pini_vantini_band,
    pini_vantini_bands,
)
from .liebl_reimherr import (
    liebl_reimherr_band,
    liebl_reimherr_bands,
    cov2tau,
    r_backend_available,
)
from .ctate_bands import (
    ctate_confidence_band,
    ctate_confidence_bands,
    ctate_pointwise_sd,
)

__all__ = [
    "eif_covariance",
    "eif_pointwise_variance",
    "eif_pointwise_sd",
    "eif_correlation",
    "multiplier_bootstrap_band",
    "multiplier_bootstrap_bands",
    "topological_effect_test",
    "iwt_pvalues",
    "IWTResult",
    "pini_vantini_band",
    "pini_vantini_bands",
    "liebl_reimherr_band",
    "liebl_reimherr_bands",
    "cov2tau",
    "r_backend_available",
    "ctate_confidence_band",
    "ctate_confidence_bands",
    "ctate_pointwise_sd",
]
