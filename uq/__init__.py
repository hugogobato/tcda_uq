"""Uncertainty quantification.

  * :mod:`.asymptotic` -- confidence bands for mean estimands, with assumptions
    specific to each method; native interval-wise testing returns p-values.
  * :mod:`.conformal`  -- C2: finite-sample *prediction* bands for the individual
    effect (ITTE), with declared target populations and training-fixed scores.

The organising distinction: expectations -> confidence (asymptotic, ~1/sqrt(n));
an individual draw -> prediction (conformal, with width reflecting residual
individual uncertainty, which may vanish for deterministic outcomes).
"""
