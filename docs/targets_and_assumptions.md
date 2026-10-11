# Targets, assumptions, and guarantee scope

This page summarizes what each package output concerns. Full theorem statements are maintained separately from this code and reproducibility release. An API call produces a numerical object; it does not verify the stated conditions for a chosen data set or nuisance learner.

## Representation and causal identification

Fix a compact filtration interval, a homology degree `d`, an exponent `r >= 1`, and a measurable rule for removing or truncating essential classes. Each outcome is mapped to a persistence diagram and then to its plain normalized power-weighted tent silhouette. This prespecified map defines the outcome curve used by every estimator and band. Changing the interval, degree, exponent, filtration, or essential-class rule changes the target.

The causal identification statements require consistency, conditional exchangeability given the measured covariates, and positivity on the target population. Strong overlap is additionally required for some mean-inference conditions. A conditional mean at a fixed `x` is identified only for a selected version and under the stated model or continuity assumptions; the package does not assert pointwise identification at every value in a probability-zero set.

The silhouette and effect curves are evaluated on a fixed grid. A gridwise coverage calculation is a calculation on that grid. A continuous-interval guarantee needs a valid gridwise result plus a justified interpolation and Lipschitz inflation argument; a finer grid alone does not establish it.

## Mean confidence bands

TATE `tate_band` and fixed-`x` CTATE `ctate_band` target deterministic mean curves. Their asymptotic interpretation requires process convergence, consistent Gaussian multiplier calibration, suitable variance or raw-supremum quantile conditions, and negligible nuisance and smoothing bias. For the sufficient cross-fitted TATE regime, fold assignments are fixed before observing outcomes or randomized independently of the observations, and nuisance errors satisfy the stated product-rate and boundedness conditions. The default Gaussian, unstudentized multiplier avoids dividing by a variance estimate; it does not remove the other assumptions.

For CTATE, the band concerns the conditional mean curve at the chosen `x`. Its validity requires the chosen conditional-mean version and a valid second-stage approximation at that `x`. A confidence band for the mean is not a prediction band for an individual response or treatment effect.

## Interval-wise tests

`tate_test(..., method="iwt")` returns adjusted and unadjusted p-values for a prespecified grid and a mean-curve null. Its adjusted p-values support the stated interval-wise type-I error control. They are not a simultaneous confidence band, and their width is not a confidence-band width. Reporting confidence coverage would require a separate inversion over candidate mean curves.

## Individual-effect prediction

`ITTEConformal` constructs an individual-effect prediction band by calibrating arm-specific potential-outcome scores and taking their interval difference. Training data, calibration observations, and the new unit are independent under the stated split design. The fitted centers, scales, and estimated weights are fixed from training data before calibration. The two armwise error allocations must sum to at most the requested total error. The resulting finite-sample statement is a lower bound, can be conservative, and is marginal over the new covariates under the declared target; it is not a conditional guarantee at every fixed `x`.

The target must match the test population:

| `target` | Covariate tilt | Appropriate evaluation draws |
|---|---|---|
| `"full"` | Original covariate population | `sample_target(..., target="full")` |
| `"overlap"` | Density proportional to `pi(X) * (1 - pi(X))` | `sample_target(..., target="overlap")` |
| `"matching"` | Density proportional to `min(pi(X), 1 - pi(X))` | `sample_target(..., target="matching")` |

With true likelihood ratios, the finite-sample lower bound applies to that target. If propensity ratios are estimated, the weighted calibration result directly concerns the distribution induced by the normalized fitted ratios. A lower bound for the intended population subtracts the armwise total-variation discrepancies. Ratio consistency in normalized `L1` is sufficient for the discrepancy to vanish. Independent fitting, an AIPW product-rate condition, or clipping alone does not establish exact full-population coverage. Replacing full-population weights with overlap or matching weights changes the target.

The target point contributes an atom at infinity to each weighted conformal quantile. Bounded weights can make infinite radii less frequent, but do not guarantee a finite radius for every finite calibration sample. Coverage, the frequency of unbounded outputs, and widths conditional on finiteness should be reported separately.

## Clinical and observational examples

The CT image tutorial transforms images into topological summaries and then constructs treatment assignment and potential outcomes for a synthetic demonstration. Even when a user supplies real images, the notebook does not observe both potential outcomes for clinical patients, estimate clinical counterfactual coverage, or establish a clinical treatment effect.
