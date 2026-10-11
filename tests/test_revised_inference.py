"""Deterministic regression checks of the revised inference contract.

These verify calculations and target/score separation, not statistical coverage.
"""
import importlib
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.model_selection import KFold

from tcda_uq import tate_band, tate_test
from tcda_uq.uq.asymptotic import (
    multiplier_bootstrap_band, ctate_confidence_band, iwt_pvalues, IWTResult,
    eif_pointwise_sd, eif_correlation, pini_vantini_band,
)
from tcda_uq.uq.conformal import (
    ConformalArm, ITTEConformal, AdaptiveConformalArm, AdaptiveITTEConformal,
    LocalScale, weighted_conformal_radius, split_conformal_radius,
    make_weight_fn, propensity_weights, overlap_weights, matching_weights,
    weight_upper_bound, joint_itte_bounds, bonferroni_itte_bounds,
    bridge_constant, empirical_score_sensitivity, DiagramConformalArm,
)

GRID = np.array([0., .5, 1.])


class FixedNuisance:
    tseq = GRID

    def predict_mu(self, X):
        n = len(X)
        return [(np.zeros((n,3)), np.ones((n,3))) for _ in range(2)]

    def predict_pi(self, X):
        return np.full(len(X), .25)


def sample(n=24, dims=2):
    A = np.arange(n) % 2
    X = np.arange(n, dtype=float)[:,None]/n
    phi = np.repeat((1 + np.arange(n, dtype=float))[:,None,None], dims, axis=1)
    phi = np.repeat(phi, 3, axis=2)
    return phi, A, X


def cross_result(phi=None):
    phi = np.array([[0.,-1.,-2.],[0.,1.,2.],[0.,-2.,-4.],[0.,2.,4.]]) if phi is None else phi
    return SimpleNamespace(tseq=GRID, aipw=[np.array([.1,.2,.3])], influence=lambda:[phi])


def test_default_gaussian_unstudentized_is_direct_absolute_supremum():
    cr = cross_result()
    band = tate_band(cr, d=0, alpha=.2, n_boot=19, rng=8)
    phi = cr.influence()[0]
    g = np.random.default_rng(8).standard_normal((19,4))
    expected = np.quantile(np.max(np.abs(g@phi/2), axis=1), .8)/2
    np.testing.assert_allclose(band.upper-band.center, expected)
    assert np.isfinite(band.lower).all()  # endpoint variance is zero


def test_studentized_zero_variance_returns_whole_line_without_floor():
    cr = cross_result()
    band = tate_band(cr, d=0, standardize=True, n_boot=3)
    assert np.isneginf(band.lower).all() and np.isposinf(band.upper).all()
    assert eif_pointwise_sd(cr.influence()[0])[0] == 0
    with pytest.raises(ValueError, match="zero variance"):
        eif_correlation(cr.influence()[0])


@pytest.mark.parametrize("method", ["iwt","pv","pini_vantini"])
def test_iwt_separate_native_testing_api(method):
    cr = cross_result(np.array([[-1.,-2.,-3.],[1.,2.,3.]]))
    with pytest.raises(ValueError, match="tate_test"):
        tate_band(cr, method=method, n_boot=9)
    result = tate_test(cr, method=method, d=0, n_boot=9, rng=2)
    assert isinstance(result,IWTResult)
    assert np.all(result.p_adjusted >= result.p_unadjusted)
    assert np.all((result.p_adjusted >= 0)&(result.p_adjusted <= 1))


def test_iwt_zero_variance_abstains_and_legacy_envelope_is_unqualified():
    cr = cross_result()
    result = tate_test(cr, d=0, n_boot=3)
    assert result.status == "zero_variance_no_rejection"
    assert not result.significant().any()
    with pytest.warns(FutureWarning, match="deprecated"):
        legacy = pini_vantini_band(cr.influence()[0],GRID,cr.aipw[0],n_boot=3)
    assert legacy.kind is None and legacy.level is None


def test_ctate_signed_residual_process_uses_absolute_sup_and_no_floor():
    R = np.array([[0.,-1.,2.],[0.,3.,-2.],[0.,-4.,1.]])
    a = np.array([.2,.3,.5])
    learner = SimpleNamespace(tseq=GRID,weights=lambda x,d:a,residuals=lambda d:R,
                              predict_dim=lambda x,d:np.array([[1.,2.,3.]]))
    band = ctate_confidence_band(learner,[0.],n_boot=13,rng=4,alpha=.2)
    draws = np.random.default_rng(4).standard_normal((13,3))
    expected = np.quantile(np.max(np.abs(draws@(a[:,None]*R)),axis=1),.8)
    np.testing.assert_allclose(band.upper-band.center,expected)
    studentized = ctate_confidence_band(learner,[0.],n_boot=3,standardize=True)
    assert np.isposinf(studentized.upper).all()
    learner.design_singular_ = True
    singular = ctate_confidence_band(learner,[0.],n_boot=3)
    assert np.isposinf(singular.upper).all()


def test_empty_zero_mass_and_exact_quantile_boundary_keep_test_atom():
    assert np.isinf(weighted_conformal_radius([], .2, [], 1.))
    assert np.isinf(weighted_conformal_radius([1.], .2, [0.], 0.))
    assert np.isinf(split_conformal_radius([], .2))
    # At alpha=.5, calibration mass exactly reaches the required half mass.
    assert weighted_conformal_radius([2.,1.], .5, [1.,1.], 2.) == 2.
    assert np.isinf(weighted_conformal_radius([2.,1.], .5, [1.,1.], 2.001))
    q = weighted_conformal_radius([3.,1.,2.], .4, [2.,3.,1.], [1.,7.])
    np.testing.assert_array_equal(q,[3.,np.inf])
    np.testing.assert_array_equal(q,weighted_conformal_radius([3.,1.,2.],.4,[2e200,3e200,1e200],[1e200,7e200]))


@pytest.mark.parametrize("bad", [np.nan,-1.,np.inf])
def test_weights_reject_invalid_inputs(bad):
    with pytest.raises(ValueError):
        weighted_conformal_radius([1.],.2,[bad],1.)


@pytest.mark.parametrize("alpha", [0.,1.,np.nan])
def test_levels_reject_outside_mathematical_domain(alpha):
    with pytest.raises(ValueError):
        weighted_conformal_radius([1.],alpha,[1.],1.)
    with pytest.raises(ValueError):
        multiplier_bootstrap_band(np.ones((3,3)),GRID,np.zeros(3),alpha=alpha)


def test_full_ratio_strict_positivity_and_target_specific_endpoint_rules():
    with pytest.raises(ValueError,match="strict positivity"):
        propensity_weights([0.,.5],1)
    np.testing.assert_array_equal(overlap_weights([0.,.5,1.],1),[1.,.5,0.])
    np.testing.assert_array_equal(matching_weights([0.,.5,1.],1),[1.,1.,0.])
    assert weight_upper_bound("naive") == np.inf
    assert weight_upper_bound("tilted",gamma=.5) == np.inf
    assert weight_upper_bound("overlap") == 1.
    assert np.isfinite(make_weight_fn("clip",eps=.1)([0.,1.],1)).all()
    with pytest.raises(ValueError):
        make_weight_fn("clip",eps=.5)
    with pytest.raises(ValueError):
        make_weight_fn("shrink",cap=0)


def test_modulation_default_is_calibration_independent_and_nonconstant_requires_training():
    nu = FixedNuisance()
    cal = sample()
    arm = ConformalArm(nu,cal,0,1,weighted=False)
    altered = (cal[0]*100,cal[1],cal[2])
    other = ConformalArm(nu,altered,0,1,weighted=False)
    np.testing.assert_array_equal(arm.s,other.s)
    np.testing.assert_array_equal(arm.s,np.ones(3))
    with pytest.raises(ValueError,match="independent training"):
        ConformalArm(nu,cal,0,1,weighted=False,modulation_kind="pointwise-sd")
    R_train = np.array([[-1.,-2.,0.],[1.,2.,0.]])
    arm = ConformalArm(nu,cal,0,1,weighted=False,modulation_kind="pointwise-sd",modulation_residuals=R_train)
    other = ConformalArm(nu,altered,0,1,weighted=False,modulation_kind="pointwise-sd",modulation_residuals=R_train)
    np.testing.assert_array_equal(arm.s,other.s)
    assert not arm.s.flags.writeable


def test_empty_arm_has_infinite_radius_and_band():
    phi,A,X = sample(6)
    A[:] = 0
    arm = ConformalArm(FixedNuisance(),(phi,A,X),0,1,weighted=False)
    assert arm.scores.size == 0
    lo,hi,_ = arm.band_bounds([[.3]],.1)
    assert np.isneginf(lo).all() and np.isposinf(hi).all()


def test_target_weights_and_known_ratio_metadata_do_not_depend_on_outcomes():
    cal = sample()
    pi = lambda X: .2+.6*X[:,0]
    arm = ConformalArm(FixedNuisance(),cal,0,1,target="overlap",propensity_fn=pi,weights_known=True)
    np.testing.assert_allclose(arm.w_calib,1-pi(cal[2][cal[1]==1]))
    assert arm.target_ == "overlap" and arm.weights_known_
    fitted = ConformalArm(FixedNuisance(),cal,0,1,target="matching")
    assert not fitted.weights_known_ and "TV" in fitted.coverage_scope_
    clipped = ConformalArm(FixedNuisance(),cal,0,1,weight_fn=make_weight_fn("clip"),weights_known=True)
    assert clipped.target_ == "full" and not clipped.weights_known_


def test_internal_fit_uses_train_residuals_and_composes_alpha_over_arms(monkeypatch):
    mod = importlib.import_module("tcda_uq.uq.conformal.composition")
    monkeypatch.setattr(mod,"fit_split_nuisances",lambda *args,**kw:FixedNuisance())
    cal = sample(40)
    model = ITTEConformal.fit(cal,GRID,modulation_kind="pointwise-sd",weighted=False,random_state=3)
    training = model.train_index_
    for a,arms in ((0,model.arms0),(1,model.arms1)):
        residuals = cal[0][training[cal[1][training]==a],0,:]-a
        np.testing.assert_allclose(arms[0].s,np.maximum(residuals.std(axis=0),1e-8))
    calls = []
    for arm in (model.arms0[0],model.arms1[0]):
        original = arm.band_bounds
        def wrapped(X,alpha,_original=original):
            calls.append(alpha)
            return _original(X,alpha)
        monkeypatch.setattr(arm,"band_bounds",wrapped)
    model.band_bounds([[.5]],.2)
    assert calls == [.1,.1]


def test_joint_degree_quantile_allocates_arm_budget_and_retains_infinity():
    cal = sample(8)
    model = ITTEConformal.fit(cal,GRID,nuisance=FixedNuisance(),calib_index=np.arange(8),weighted=False)
    bounds = joint_itte_bounds(model,[[.3]],.1)
    assert all(np.isposinf(hi).all() for _,hi,_ in bounds.values())
    with pytest.raises(ValueError,match="nonempty"):
        joint_itte_bounds(model,[[.3]],.1,dims=[])
    # Bonferroni across two degrees allocates .1 to each ITTE, then .05 per arm.
    calls=[]
    fake=SimpleNamespace(n_hom_dim=2,band_bounds=lambda X,alpha,d:calls.append((alpha,d)))
    bonferroni_itte_bounds(fake,[[.3]],.2)
    assert calls == [(.1,0),(.1,1)]


def test_adaptive_arm_same_training_shape_and_local_scale_at_calibration_and_prediction():
    local = LocalScale().fit(np.array([[0.],[1.]]),np.array([2.,2.]))
    frozen = np.array([1.,2.,3.])
    arm = AdaptiveConformalArm(FixedNuisance(),sample(),0,1,weighted=False,
                               local_scale=local,modulation_scale=frozen)
    residual = sample()[0][sample()[1]==1,0,:]-1
    expected = np.max(np.abs(residual)/frozen,axis=1)/2
    np.testing.assert_allclose(arm.scores,expected)
    _,upper,c = arm.band_bounds([[.5]],.2)
    np.testing.assert_allclose(upper-c,arm.radius([[.5]],.2)[:,None]*2*frozen)
    with pytest.raises(ValueError,match="independent training"):
        AdaptiveConformalArm(FixedNuisance(),sample(),0,1,weighted=False,localize=True)


def test_bridge_uses_revised_universal_bound_and_empirical_diagnostic_is_named():
    assert bridge_constant(r=2.) == 5.
    assert bridge_constant(r=2.,normalization="gudhi") == pytest.approx(5*np.sqrt(2))
    with pytest.warns(FutureWarning):
        assert bridge_constant(100.,2.,300.) == 5.
    with pytest.raises(ValueError):
        bridge_constant(r=.5)
    result=empirical_score_sensitivity([0.,.5,1.],1.,2.,.1,[1.])
    assert result["empirical_affected_fraction"] == pytest.approx(1/3)
    assert "certified_coverage" not in result
    with pytest.raises(ValueError,match="independent"):
        DiagramConformalArm.fit([np.array([[0.,1.]])])


def test_cross_fit_default_partition_independent_of_treatment(monkeypatch):
    nuisance=importlib.import_module("tcda_uq.estimators.nuisance")
    monkeypatch.setattr(nuisance,"fit_functional_regression",lambda *args,**kw:None)
    prop=SimpleNamespace(predict_proba=lambda X:np.tile([.5,.5],(len(X),1)))
    monkeypatch.setattr(nuisance,"fit_propensity",lambda *args,**kw:prop)
    monkeypatch.setattr(nuisance,"predict_functional_regression",lambda reg,X,tseq:[(np.zeros((len(X),3)),np.ones((len(X),3)))])
    cal=sample(40,dims=1)
    first=nuisance.cross_fit(cal,GRID,n_splits=4,random_state=3)
    second=nuisance.cross_fit((cal[0],1-cal[1],cal[2]),GRID,n_splits=4,random_state=3)
    expected=np.concatenate([test for _,test in KFold(4,shuffle=True,random_state=3).split(cal[2])])
    np.testing.assert_array_equal(first.order,expected)
    np.testing.assert_array_equal(first.order,second.order)
    assert first.fold_scheme == "independent"


def test_meta_learner_modulation_and_local_scale_never_fit_calibration(monkeypatch):
    mod=importlib.import_module("tcda_uq.uq.conformal.adaptive_cp")
    estimators=importlib.import_module("tcda_uq.estimators")
    class DummyLearner:
        def __init__(self,**kw):
            pass
        def fit(self,*args,**kw):
            return self
        def predict(self,X):
            return np.zeros((len(X),1,3))
    monkeypatch.setattr(estimators,"CTATEDRLearner",DummyLearner)
    monkeypatch.setattr(mod,"fit_functional_regression",lambda *a,**kw:None)
    monkeypatch.setattr(mod,"fit_propensity",lambda *a,**kw:SimpleNamespace(predict_proba=lambda X:np.tile([.5,.5],(len(X),1))))
    monkeypatch.setattr(mod,"predict_functional_regression",lambda reg,X,tseq:[(np.zeros((len(X),3)),np.zeros((len(X),3)))])
    cal=sample(16,dims=1)
    first=mod.ConformalMetaLearner.fit(cal,GRID,modulation_kind="pointwise-sd",random_state=4)
    changed=cal[0].copy()
    changed[first.calib_index_] *= 100
    second=mod.ConformalMetaLearner.fit((changed,cal[1],cal[2]),GRID,modulation_kind="pointwise-sd",random_state=4)
    np.testing.assert_array_equal(first.s[0],second.s[0])
    assert first.local_scale[0].const_ == second.local_scale[0].const_
    assert "pseudo-outcome" in first.coverage_scope_


def test_adaptive_fit_reuses_training_global_shape(monkeypatch):
    mod=importlib.import_module("tcda_uq.uq.conformal.adaptive_cp")
    monkeypatch.setattr(mod,"fit_split_nuisances",lambda *args,**kw:FixedNuisance())
    cal=sample(40)
    model=AdaptiveITTEConformal.fit(cal,GRID,modulation_kind="pointwise-sd",local_scale=False,
                                   weighted=False,random_state=5)
    training=np.setdiff1d(np.arange(40),model.calib_index_)
    for a,arms in ((0,model.arms0),(1,model.arms1)):
        R=cal[0][training[cal[1][training]==a],0,:]-a
        np.testing.assert_allclose(arms[0].s,np.maximum(R.std(axis=0),1e-8))


def test_joint_degrees_reject_unaligned_units_and_adaptive_arms():
    cal=sample(24)
    model=ITTEConformal.fit(cal,GRID,nuisance=FixedNuisance(),calib_index=np.arange(24),weighted=False)
    model.arms0[1].calib_index_=model.arms0[1].calib_index_[::-1]
    with pytest.raises(ValueError,match="aligned"):
        joint_itte_bounds(model,[[.3]],.2)
    model.arms0[0]=AdaptiveConformalArm(FixedNuisance(),cal,0,0,weighted=False)
    with pytest.raises(ValueError,match="plain ConformalArm"):
        joint_itte_bounds(model,[[.3]],.2)


def test_public_dimension_validation_and_external_scale_split_validation():
    with pytest.raises(ValueError,match="homology"):
        tate_band(cross_result(),d=-1,n_boot=2)
    with pytest.raises(ValueError,match="disjoint"):
        ITTEConformal.fit(sample(),GRID,nuisance=FixedNuisance(),calib_index=[0,1],
                         train_index=[1,2],modulation_kind="pointwise-sd")


def test_diagram_metric_retains_all_positive_lifetimes():
    from tcda_uq.uq.conformal import wasserstein_distance
    tiny=np.array([[0.,1e-13]])
    assert wasserstein_distance(tiny,[]) == pytest.approx(5e-14,rel=1e-15,abs=0)


def test_diagram_metric_cannot_choose_forbidden_finite_penalty_edges():
    from tcda_uq.uq.conformal import wasserstein_distance
    huge=np.array([[0.,1e14],[0.,1e14]])
    assert wasserstein_distance(huge,[]) == pytest.approx(1e14)
    assert wasserstein_distance(huge,[],q=2) == pytest.approx(np.sqrt(2)*5e13)


def test_diagram_metric_scaled_powers_and_finite_numeric_limits():
    from tcda_uq.uq.conformal import wasserstein_distance
    huge=np.array([[0.,1e200],[0.,1e200]])
    assert wasserstein_distance(huge,[],q=2) == pytest.approx(np.sqrt(2)*5e199)
    assert wasserstein_distance(huge,huge,q=2) == 0.
    with pytest.raises(ValueError,match="underflow"):
        wasserstein_distance([[0.,1.]],[[0.,1e200]],q=2)
    with pytest.raises(OverflowError,match="float range"):
        wasserstein_distance(np.repeat([[-1e308,1e308]],3,axis=0),[])
    assert bridge_constant(r=2.,normalization="legacy_gudhi") == bridge_constant(r=2.,normalization="gudhi")


def test_weighted_left_quantile_includes_exact_boundary_without_mass_tolerance():
    assert weighted_conformal_radius([0.,1.,1.,3.],.25,[0.,2.,3.,1.],2.) == 3.
    assert weighted_conformal_radius([0.,1.,1.,3.],.25,[2.,3.,1.,1.],1.) == 1.
    # Even one representable increment past equality must retain the infinity atom.
    assert np.isinf(weighted_conformal_radius([0.,1.,1.,3.],.25,[0.,2.,3.,1.],np.nextafter(2.,np.inf)))


def test_split_and_weighted_quantiles_share_exact_float_level_semantics():
    # Binary .3 lies just below rational 3/10; rounded floating multiplication
    # must not move the inclusive CDF threshold below the supplied level.
    scores=np.arange(1.,10.)
    assert split_conformal_radius(scores,.3) == 8.
    assert weighted_conformal_radius(scores,.3,np.ones(9),1.) == 8.
