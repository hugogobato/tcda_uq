"""Statistical coverage-property (regression) tests on the tri-oracle harness.

These are the library's *validity* tests: they draw many seeded replicates from
:class:`~tcda_uq.datasets.TriOracleSimulation` (which exposes the true TATE /
CTATE / ITTE) and check that each band's empirical coverage matches its target,
in the sense appropriate to that band:

  * TATE multiplier-bootstrap / Liebl-Reimherr -> **simultaneous** confidence
    coverage of ``psi_d`` at or above nominal;
  * TATE Pini-Vantini -> **interval-wise** error control (a weaker, narrower band);
  * CTATE confidence band -> simultaneous-in-t coverage of ``tau_d(., x)`` at x;
  * ITTE overlap-stabilized conformal -> **prediction** coverage of a fresh
    ``delta`` draw at or above nominal, with the weak-overlap ``+inf`` atom
    suppressed (finite-band share ~ 1).

They are deterministic (fixed seeds) but drive the full nuisance pipeline, so
each replicate costs a cross-fit; hence they are marked ``slow`` and excluded
from the default ``pytest`` run (``addopts = -m 'not slow'``). Run them with::

    pytest -m slow

Tolerances are generous margins below the observed coverage so the assertions
are stable across BLAS/threading, not tight statistical claims (those live in
the ``experiments/`` production tables).
"""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tcda_uq.datasets import TriOracleSimulation
from tcda_uq.estimators import CTATEDRLearner, cross_fit
from tcda_uq.metrics import interval_wise_error
from tcda_uq.uq.asymptotic import (
    ctate_confidence_band,
    liebl_reimherr_band,
    multiplier_bootstrap_band,
    pini_vantini_band,
)
from tcda_uq.uq.conformal import ITTEConformal, make_weight_fn

pytestmark = pytest.mark.slow

ALPHA = 0.10
NOMINAL = 1.0 - ALPHA


def _prop():
    return LogisticRegression(max_iter=500)


def test_tate_bands_coverage_and_ordering():
    """mbb & LR cover psi_d simultaneously ~>= nominal; PV controls interval-wise error."""
    sim = TriOracleSimulation(n_hom_dim=1, resolution=25, n_basis=5, noise_scale=0.3, seed=0)
    truth = sim.true_tate()[0]
    reps = 15

    mbb_cov, lr_cov, pv_iwe, mbb_w, pv_w = [], [], [], [], []
    for r in range(reps):
        s = sim.sample(300, rng=5000 + r)
        fit = cross_fit(s.observed, sim.tseq, n_basis=5, n_splits=2,
                        random_state=r, propensity_estimator=_prop())
        inf, est = fit.influence()[0], fit.aipw[0]
        b_mbb = multiplier_bootstrap_band(inf, sim.tseq, est, alpha=ALPHA, n_boot=300, rng=r)
        b_lr = liebl_reimherr_band(inf, sim.tseq, est, alpha=ALPHA, backend="python")
        b_pv = pini_vantini_band(inf, sim.tseq, est, alpha=ALPHA, n_boot=300, rng=r)
        mbb_cov.append(bool(b_mbb.covers(truth)))
        lr_cov.append(bool(b_lr.covers(truth)))
        pv_iwe.append(interval_wise_error(b_pv.lower, b_pv.upper, truth))
        mbb_w.append(b_mbb.mean_width())
        pv_w.append(b_pv.mean_width())

    assert np.mean(mbb_cov) >= NOMINAL - 0.12          # observed ~0.88-0.97
    assert np.mean(lr_cov) >= NOMINAL - 0.05           # observed ~1.0 (conservative)
    assert np.mean(pv_iwe) <= ALPHA + 0.05             # interval-wise error controlled
    assert np.mean(pv_w) <= np.mean(mbb_w)             # PV narrower than simultaneous
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


def test_itte_conformal_prediction_coverage_and_finite_bands():
    """Overlap-stabilized ITTE band covers a fresh draw >= nominal and stays finite."""
    sim = TriOracleSimulation(n_hom_dim=1, resolution=25, n_basis=5, noise_scale=0.3, seed=0)
    train = sim.sample(600, rng=42)
    test = sim.sample(400, rng=43)
    model = ITTEConformal.fit(
        train.observed, sim.tseq, n_basis=5, weight_fn=make_weight_fn("overlap"),
        random_state=0, propensity_estimator=_prop(),
    )
    lo, hi, _ = model.band_bounds(test.X, alpha=ALPHA, d=0)
    target = test.oracle_itte[:, 0, :]
    covered = ((target >= lo) & (target <= hi)).all(axis=1)
    finite = np.isfinite(hi - lo).all(axis=1)
    assert covered.mean() >= NOMINAL - 0.02            # observed ~0.997 (conservative)
    assert finite.mean() >= 0.95                       # overlap weights suppress the +inf atom


def test_naive_weighting_can_produce_unbounded_bands_under_weak_overlap():
    """Contrast: naive 1/pi weights let the weak-overlap +inf atom fire (headline motivation)."""
    sim = TriOracleSimulation(n_hom_dim=1, resolution=25, n_basis=5, noise_scale=0.3,
                              prop_scale=4.0, seed=0)                # weak overlap
    train = sim.sample(600, rng=7)
    test = sim.sample(400, rng=8)
    naive = ITTEConformal.fit(
        train.observed, sim.tseq, n_basis=5, weight_fn=make_weight_fn("naive"),
        random_state=0, propensity_estimator=_prop(),
    )
    overlap = ITTEConformal.fit(
        train.observed, sim.tseq, n_basis=5, weight_fn=make_weight_fn("overlap"),
        random_state=0, propensity_estimator=_prop(),
    )
    lo_n, hi_n, _ = naive.band_bounds(test.X, alpha=ALPHA, d=0)
    lo_o, hi_o, _ = overlap.band_bounds(test.X, alpha=ALPHA, d=0)
    inf_share_naive = (~np.isfinite(hi_n - lo_n).all(axis=1)).mean()
    inf_share_overlap = (~np.isfinite(hi_o - lo_o).all(axis=1)).mean()
    # the whole point of Phase 6.5: overlap stabilization removes the unbounded bands
    assert inf_share_overlap <= inf_share_naive
    assert inf_share_overlap <= 0.02
