"""Marginal individual prediction with frozen scores and declared targets.

Exact likelihood ratios give a finite-sample coverage lower bound for their
population. Learned ratios require armwise total-variation qualification for the
intended population. Optional localization and pseudo-outcome predictors have
separate, explicitly experimental scope. Legacy bridge certificates report only
empirical score diagnostics. Bands and scores are computed on a fixed grid.
"""

from .functional_cp import (
    residual_curves,
    modulation,
    sup_norm_score,
    split_conformal_radius,
    functional_cp_band,
    grid_discretization_slack,
)
from .weighted_cp import (
    propensity_weights,
    weighted_conformal_radius,
    ConformalArm,
)
from .composition import (
    fit_split_nuisances,
    ITTEConformal,
)
from .adaptive_cp import (
    LocalScale,
    fit_local_scale,
    median_bandwidth,
    kernel_weights,
    localized_weighted_radius,
    AdaptiveConformalArm,
    AdaptiveITTEConformal,
    ConformalMetaLearner,
    ctate_prediction_band,
)
from .stabilized_weights import (
    STABILIZERS,
    EXACT_TARGET_METHODS,
    APPROX_TARGET_METHODS,
    naive_weights,
    overlap_weights,
    tilted_weights,
    matching_weights,
    clip_weights,
    shrink_weights,
    stabilized_weights,
    make_weight_fn,
    weight_upper_bound,
    target_description,
)
from .degree_multiplicity import (
    bonferroni_itte_bounds,
    joint_itte_bounds,
    simultaneous_itte_bounds,
)
from .silhouette_bridge import (
    bridge_constant,
    estimate_bridge_params,
    score_perturbation_bound,
    certified_score_inflation,
    certified_width_inflation,
    coverage_certificate,
    empirical_score_sensitivity,
)
from .diagram_score import (
    wasserstein_distance,
    frechet_mean_diagram,
    wasserstein_scores,
    DiagramConformalArm,
)

__all__ = [
    "residual_curves",
    "modulation",
    "sup_norm_score",
    "split_conformal_radius",
    "functional_cp_band",
    "grid_discretization_slack",
    "propensity_weights",
    "weighted_conformal_radius",
    "ConformalArm",
    "fit_split_nuisances",
    "ITTEConformal",
    "LocalScale",
    "fit_local_scale",
    "median_bandwidth",
    "kernel_weights",
    "localized_weighted_radius",
    "AdaptiveConformalArm",
    "AdaptiveITTEConformal",
    "ConformalMetaLearner",
    "ctate_prediction_band",
    # Phase 6.5 -- positivity-stabilized weighting
    "STABILIZERS",
    "EXACT_TARGET_METHODS",
    "APPROX_TARGET_METHODS",
    "naive_weights",
    "overlap_weights",
    "tilted_weights",
    "matching_weights",
    "clip_weights",
    "shrink_weights",
    "stabilized_weights",
    "make_weight_fn",
    "weight_upper_bound",
    "target_description",
    # Phase 6.4 -- degree multiplicity
    "bonferroni_itte_bounds",
    "joint_itte_bounds",
    "simultaneous_itte_bounds",
    # Phase 6.1 -- silhouette<->diagram bridge
    "bridge_constant",
    "estimate_bridge_params",
    "score_perturbation_bound",
    "certified_score_inflation",
    "certified_width_inflation",
    "coverage_certificate",
    "empirical_score_sensitivity",
    # Phase 6.2 -- diagram-space score
    "wasserstein_distance",
    "frechet_mean_diagram",
    "wasserstein_scores",
    "DiagramConformalArm",
]
