"""Seeded statistical regression diagnostics, run with ``pytest -m slow``.

These tests detect large changes in selected simulation designs. Their small
replication counts and loose tolerances do not establish coverage theorems.
Mean-effect tests use the generating full population. Overlap prediction tests
use fresh draws from the overlap target and its known generating propensity.
IWT is checked as a native testing procedure, rather than as a confidence band
or a fraction of rejected grid points. Finite-band frequencies are diagnostics,
not a consequence of bounded overlap weights alone.
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tcda_uq.datasets import TriOracleSimulation
from tcda_uq.estimators import CTATEDRLearner, cross_fit
from tcda_uq.uq.asymptotic import (
    ctate_confidence_band,
    liebl_reimherr_band,
    multiplier_bootstrap_band,
    iwt_pvalues,
)
from tcda_uq.uq.conformal import ITTEConformal

pytestmark = pytest.mark.slow

ALPHA = 0.10
NOMINAL = 1.0 - ALPHA


def _prop():
    return LogisticRegression(max_iter=500)


def test_tate_simultaneous_coverage_diagnostic():
    sim = TriOracleSimulation(n_hom_dim=1, resolution=25, n_basis=5, noise_scale=0.3, seed=0)
    truth = sim.true_tate()[0]
    reps = 15

    mbb_cov, lr_cov, mbb_w = [], [], []
    for r in range(reps):
        s = sim.sample(300, rng=5000 + r)
        fit = cross_fit(s.observed, sim.tseq, n_basis=5, n_splits=2,
                        random_state=r, propensity_estimator=_prop())
        inf, est = fit.influence()[0], fit.aipw[0]
        b_mbb = multiplier_bootstrap_band(inf, sim.tseq, est, alpha=ALPHA, n_boot=300, rng=r)
        b_lr = liebl_reimherr_band(inf, sim.tseq, est, alpha=ALPHA, backend="python")
        mbb_cov.append(bool(b_mbb.covers(truth)))
        lr_cov.append(bool(b_lr.covers(truth)))
        mbb_w.append(b_mbb.mean_width())

    assert np.mean(mbb_cov) >= NOMINAL - 0.12
    assert np.mean(lr_cov) >= NOMINAL - 0.05
    assert np.all(np.isfinite(mbb_w)) and np.mean(mbb_w) > 0


def test_ctate_confidence_band_coverage_at_held_out_x():
    sim = TriOracleSimulation(n_hom_dim=1, resolution=25, n_basis=5, noise_scale=0.3, seed=0)
    x = sim.EX + 0.3
    truth = sim.true_ctate(x[None, :])[0, 0]
    reps = 15

    cov = []
    for r in range(reps):
        s = sim.sample(300, rng=6000 + r)
        learner = CTATEDRLearner(n_basis=5).fit(
            s.observed, sim.tseq, n_splits=2, random_state=r, propensity_estimator=_prop()
        )
        band = ctate_confidence_band(learner, x, d=0, alpha=ALPHA, n_boot=300, rng=r)
        cov.append(bool(band.covers(truth)))
    assert np.mean(cov) >= NOMINAL - 0.15              # observed ~0.92


def test_itte_prediction_diagnostic_at_matched_overlap_target():
    sim = TriOracleSimulation(n_hom_dim=1, resolution=25, n_basis=5, noise_scale=0.3, seed=0)
    train = sim.sample(600, rng=42)
    test = sim.sample_target(400, target="overlap", rng=43)
    model = ITTEConformal.fit(
        train.observed, sim.tseq, n_basis=5, target="overlap",
        random_state=0, propensity_fn=sim.propensity_score, weights_known=True,
    )
    lo, hi, _ = model.band_bounds(test.X, alpha=ALPHA, d=0)
    target = test.oracle_itte[:, 0, :]
    covered = ((target >= lo) & (target <= hi)).all(axis=1)
    finite = np.isfinite(hi - lo).all(axis=1)
    assert model.weights_known_
    assert covered.mean() >= NOMINAL - 0.05
    # This assertion describes this calibration sample, not universal finiteness.
    assert finite.mean() >= 0.95


def test_full_target_can_produce_unbounded_bands_under_weak_overlap():
    sim = TriOracleSimulation(n_hom_dim=1, resolution=25, n_basis=5, noise_scale=0.3,
                              prop_scale=4.0, seed=0)                # weak overlap
    train = sim.sample(600, rng=7)
    test = sim.sample(400, rng=8)
    naive = ITTEConformal.fit(
        train.observed, sim.tseq, n_basis=5, target="full",
        random_state=0, propensity_fn=sim.propensity_score, weights_known=True,
    )
    lo_n, hi_n, _ = naive.band_bounds(test.X, alpha=ALPHA, d=0)
    inf_share_naive = (~np.isfinite(hi_n - lo_n).all(axis=1)).mean()
    assert inf_share_naive > 0
    assert np.all(hi_n >= lo_n)


def test_iwt_true_null_interval_error_diagnostic():
    """Check any false rejection on a prespecified contiguous null interval."""
    rng = np.random.default_rng(8901)
    tseq = np.linspace(0, 1, 8)
    null_interval = slice(0, 4)
    truth = np.r_[np.zeros(4), np.full(4, 0.5)]
    rejected_null, detected_signal = [], []
    for rep in range(40):
        noise = rng.normal(size=(400, 8))
        estimate = truth + noise.mean(axis=0)
        result = iwt_pvalues(noise, tseq, estimate, alpha=ALPHA,
                             n_boot=199, rng=rep)
        rejected_null.append(np.any(result.p_adjusted[null_interval] <= ALPHA))
        detected_signal.append(np.any(result.p_adjusted[4:] <= ALPHA))
    # Forty repetitions are an execution diagnostic with wide Monte Carlo error.
    assert np.mean(rejected_null) <= 0.25
    assert np.mean(detected_signal) >= 0.8
