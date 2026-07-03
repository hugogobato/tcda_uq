"""Conformal prediction bands for the individual effect (C2, Phases 4-6).

Finite-sample, distribution-free **prediction** bands for the individual
topological treatment effect ``delta_{i,d}(.)`` -- the conformal counterpart of
the asymptotic *confidence* bands (:mod:`tcda_uq.uq.asymptotic`). The core
(Phase 4) is three orthogonal layers:

  * :mod:`.functional_cp`  -- Layer 1: functional split-CP (sup-norm score + modulation).
  * :mod:`.weighted_cp`    -- Layer 2: propensity-weighted causal CP (binary treatment).
  * :mod:`.composition`    -- Layer 3: arm composition ``phi^1, phi^0 -> delta`` band.

Phase 5 (covariate-adaptive prediction, the conformal counterpart of CTATE):

  * :mod:`.adaptive_cp`    -- locally-scaled / kernel-localized ITTE bands
    (:class:`AdaptiveITTEConformal`) and the conformal meta-learner
    (:class:`ConformalMetaLearner`).

Phase 6 (topology-specific machinery + the positivity-stabilized headline):

  * :mod:`.stabilized_weights` -- positivity-stabilized weighting (6.5, headline): overlap /
    tilted / matching / clip / shrink weighters that bound the weak-overlap ``+inf`` atom.
  * :mod:`.degree_multiplicity` -- simultaneous ITTE bands over homology degree (6.4).
  * :mod:`.silhouette_bridge`  -- TATE Theorem 5.3 as a load-bearing CP component (6.1).
  * :mod:`.diagram_score`      -- diagram-space (Wasserstein) scores (6.2, optional).

Top-level entry points: :class:`ITTEConformal` (marginal, Phase 4) and
:class:`AdaptiveITTEConformal` (covariate-adaptive, Phase 5) -- fit once, band any ``x``;
pass ``weight_fn=make_weight_fn("overlap")`` for the Phase-6.5 stabilized band.
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
    # Phase 6.2 -- diagram-space score
    "wasserstein_distance",
    "frechet_mean_diagram",
    "wasserstein_scores",
    "DiagramConformalArm",
]
