"""Independent numerical witnesses for the revised topology and target laws."""

from dataclasses import replace

import numpy as np
import pytest
from gudhi import CubicalComplex
from gudhi.representations import Silhouette
from scipy.integrate import quad
from scipy.special import expit
from scipy.stats import norm

from tcda_uq.datasets import (
    TopologicalCausalSimulation,
    TriOracleSimulation,
    gen_covariate,
    gen_orbits,
    target_tilt,
)
from tcda_uq.datasets.pacce import (
    compute_phi,
    load_outcome_spec,
    load_pacce_from_frames,
    synthetic_pacce_frames,
)
from tcda_uq.datasets.sarscov2 import images_to_silhouettes, make_sarscov2_causal
from tcda_uq.silhouette import (
    compute_silhouette,
    diagrams_from_pointcloud,
    silhouette_from_image,
    silhouette_from_pointcloud,
)


def plain_tents(diagram, grid, r):
    """Direct defining formula, independent of the public core and GUDHI."""
    diagram = np.asarray(diagram)
    diagram = diagram[np.isfinite(diagram).all(axis=1)]
    diagram = diagram[diagram[:, 1] > diagram[:, 0]]
    if not len(diagram):
        return np.zeros(len(grid))
    lifetime = diagram[:, 1] - diagram[:, 0]
    weight = lifetime ** r
    tents = np.maximum(
        0, np.minimum(grid[:, None] - diagram[:, 0], diagram[:, 1] - grid[:, None])
    )
    return tents @ weight / weight.sum()


@pytest.mark.parametrize("r", [1.0, 1.5, 3.0])
def test_plain_formula_and_single_gudhi_scale_correction(r):
    diagram = np.array([[0, 2], [0, 1], [0.5, 1.8]])
    grid = np.linspace(0, 2, 81)
    expected = plain_tents(diagram, grid, r)
    actual = compute_silhouette([diagram], interval=(0, 2), r=r, resolution=len(grid))[0]
    legacy = compute_silhouette(
        [diagram], interval=(0, 2), r=r, resolution=len(grid), normalization="legacy_gudhi"
    )[0]
    library = Silhouette(
        weight=lambda bar: (bar[1] - bar[0]) ** r,
        resolution=len(grid), sample_range=[0, 2], keep_endpoints=True,
    ).fit_transform([diagram])[0]
    np.testing.assert_allclose(actual, expected, atol=1e-15)
    np.testing.assert_allclose(legacy, library, atol=1e-15)
    np.testing.assert_allclose(legacy, np.sqrt(2) * actual, atol=1e-15)
    assert np.max(np.abs(np.diff(actual)) / np.diff(grid)) <= 1 + 1e-12


def test_empty_essential_zero_and_duplicate_bars():
    diagram = np.array([[0, 2], [0, 1]])
    decorated = np.vstack([diagram, [0, np.inf], [0.4, 0.4]])
    ordinary = compute_silhouette([diagram], interval=(0, 2), resolution=21)
    result = compute_silhouette(
        [decorated, np.empty((0, 2)), np.array([[1, 1], [0, np.inf]])],
        interval=(0, 2), resolution=21,
    )
    np.testing.assert_allclose(result[0], ordinary[0])
    np.testing.assert_array_equal(result[1:], np.zeros((2, 21)))
    assert compute_silhouette([], resolution=5).shape == (0, 5)
    # Repeating the entire multiset leaves normalized tents unchanged. This
    # witnesses noninjectivity and prevents interpreting a band as a reverse
    # upper bound on diagram distance.
    repeated = compute_silhouette([np.repeat(diagram, 2, axis=0)], interval=(0, 2), resolution=21)
    np.testing.assert_allclose(repeated, ordinary)


def test_tiny_persistence_is_normalized_without_an_epsilon_floor():
    diagram = np.array([[0, 2], [0, 1]], dtype=float)
    scale = 1e-120  # every unscaled lifetime**3 underflows to zero
    small = compute_silhouette([scale * diagram], interval=(0, 2 * scale), r=3, resolution=21)
    ordinary = compute_silhouette([diagram], interval=(0, 2), r=3, resolution=21)
    assert np.isfinite(small).all()
    np.testing.assert_allclose(small / scale, ordinary, atol=2e-15)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1, 0.1])
def test_exponent_requires_supported_domain_or_explicit_experiment(bad):
    with pytest.raises(ValueError):
        compute_silhouette([], r=bad)


def test_legacy_exponent_opt_in_has_plain_units_and_no_implicit_image_default():
    diagram = np.array([[0, 2], [0, 1]], dtype=float)
    grid = np.linspace(0, 2, 21)
    result = compute_silhouette(
        [diagram], interval=(0, 2), resolution=21, r=0.1, allow_experimental_r=True
    )
    np.testing.assert_allclose(result[0], plain_tents(diagram, grid, 0.1))
    image = np.array([[0, 1, 0.2], [1, 1, 1], [0.4, 1, 0.6]])
    cubical = CubicalComplex(vertices=image)
    cubical.compute_persistence()
    h0 = cubical.persistence_intervals_in_dimension(0)
    expected = plain_tents(h0, np.linspace(0, 1, 31), 1)
    np.testing.assert_allclose(silhouette_from_image(image, resolution=31)[0], expected, atol=1e-15)
    np.testing.assert_allclose(images_to_silhouettes([image], resolution=31)[0, 0], expected, atol=1e-15)
    assert images_to_silhouettes([], resolution=31).shape == (0, 1, 31)
    with pytest.raises(ValueError):
        images_to_silhouettes([image], r=0.1)
    # Explicit essential truncation changes H0, as well as any higher degree.
    truncated = h0.copy()
    truncated[np.isinf(truncated)] = 1
    np.testing.assert_allclose(
        silhouette_from_image(image, resolution=31, max_filtration=1)[0],
        plain_tents(truncated, np.linspace(0, 1, 31), 1),
        atol=1e-15,
    )


def test_alpha_radius_units_empty_cloud_and_duplicate_points():
    points = np.array([[0, 0], [2, 0], [2, 0]], dtype=float)
    diagrams = diagrams_from_pointcloud(points)
    np.testing.assert_allclose(diagrams[0], [[0, 1]])  # half the edge length, not squared radius
    assert diagrams[1].shape == (0, 2)
    curves = silhouette_from_pointcloud(points, interval=(0, 1), resolution=11)
    np.testing.assert_allclose(curves[0], plain_tents(diagrams[0], np.linspace(0, 1, 11), 3))
    np.testing.assert_array_equal(curves[1], np.zeros(11))
    np.testing.assert_array_equal(
        silhouette_from_pointcloud(np.empty((0, 2)), resolution=11), np.zeros((2, 11))
    )


@pytest.mark.parametrize("r", [1.0, 3.0])
def test_forward_stability_witnesses_include_short_bars_and_diagonal_matches(r):
    grid = np.linspace(-0.1, 2.1, 501)
    for lifetime in (1e-8, 0.02, 2.0):
        diagram = np.array([[0, lifetime]])
        empty_curve = compute_silhouette([np.empty((0, 2))], (-0.1, 2.1), r, len(grid))[0]
        curve = compute_silhouette([diagram], (-0.1, 2.1), r, len(grid))[0]
        # Exact W1 distance to the diagonal is lifetime/2 for one bar.
        assert np.max(np.abs(curve - empty_curve)) <= (1 + 2 * r) * lifetime / 2 + 1e-14
    rng = np.random.default_rng(20261011)
    for _ in range(12):
        births = rng.uniform(0, 1, size=8)
        lengths = rng.uniform(0.001, 0.9, size=8)
        diagram = np.column_stack([births, births + lengths])
        shifts = rng.uniform(-0.03, 0.03, size=8)
        other = diagram + shifts[:, None]
        curves = compute_silhouette([diagram, other], (-0.1, 2.1), r, len(grid))
        matching_cost = np.abs(shifts).sum()
        assert np.max(np.abs(curves[0] - curves[1])) <= (1 + 2 * r) * matching_cost + 1e-13


def test_covariate_memberships_are_iid_and_not_fixed_half_or_group_sorted():
    expected_membership = np.random.default_rng(7).binomial(1, 0.5, size=20)
    group0, group1, X = gen_covariate([-1], [2], [[0]], 20, rng=7)
    np.testing.assert_array_equal(X[:, 0], np.where(expected_membership == 0, -1, 2))
    assert len(group0) == int((expected_membership == 0).sum())
    assert len(group1) == int((expected_membership == 1).sum())
    counts = [len(gen_covariate([-1], [2], [[0]], 20, rng=seed)[0]) for seed in range(8)]
    assert len(set(counts)) > 1
    assert any(count != 10 for count in counts)
    for cls in (TriOracleSimulation, TopologicalCausalSimulation):
        sim = cls(n_cov=1, mu1=[-1], mu2=[2], beta=[1.4], sigma2=0)
        np.testing.assert_array_equal(sim.sample_target_covariates(20, rng=7), X)


def target_moments(target):
    """Quadrature of a known 1D mixture, independent of rejection sampling."""
    def density(x):
        return 0.5 * norm.pdf(x, -1, np.sqrt(0.5)) + 0.5 * norm.pdf(x, 2, np.sqrt(0.5))

    def tilt(x):
        pi = expit(1.4 * x)
        if target == "full":
            return 1
        if target == "overlap":
            return pi * (1 - pi)
        if target == "matching":
            return min(pi, 1 - pi)
        return np.sqrt(pi * (1 - pi))

    # Split at zero for the matching tilt's corner. The finite limits lose
    # negligible Gaussian mass, far below the test's Monte Carlo tolerances.
    integrals = [sum(quad(lambda x: x ** k * tilt(x) * density(x), lo, hi,
                          epsabs=1e-11)[0] for lo, hi in ((-12, 0), (0, 12)))
                 for k in range(5)]
    return np.asarray(integrals) / integrals[0]


@pytest.mark.parametrize("target", ["full", "overlap", "matching", "sqrt"])
def test_target_sampler_moments_against_independent_quadrature(target):
    sim = TriOracleSimulation(n_cov=1, mu1=[-1], mu2=[2], beta=[1.4], sigma2=0.5)
    X = sim.sample_target_covariates(30000, target=target, rng=1234)[:, 0]
    moment = target_moments(target)
    for power in (1, 2):
        standard_error = np.sqrt((moment[2 * power] - moment[power] ** 2) / len(X))
        assert abs(np.mean(X ** power) - moment[power]) < 6 * standard_error + 0.002
    if target != "full":
        # The tilted means differ substantially from the full mean 0.5; this
        # check would fail if a full-population test set were mislabeled.
        assert abs(moment[1] - 0.5) > 0.1


def test_target_units_preserve_conditional_law_and_do_not_mislabel_mean_truth():
    sim = TriOracleSimulation(
        n_cov=1, n_hom_dim=1, mu1=[-1], mu2=[2], beta=[1.4], noise_scale=0,
        n_basis=1, resolution=8,
    )
    sim.Gamma[0][0][:] = 0
    sim.Gamma[1][0][:] = [[0, 1]]  # tau(X)=X at every grid point
    np.testing.assert_allclose(sim.true_tate(), np.full((1, 8), 0.5))
    for i, target in enumerate(("full", "overlap", "matching", "sqrt")):
        sample = sim.sample_target(40, target=target, rng=100 + i)
        assert sample.target == target
        np.testing.assert_allclose(sample.oracle_itte, sample.oracle_ctate)
        np.testing.assert_allclose(sample.oracle_ctate[:, 0, 0], sample.X[:, 0])
        np.testing.assert_array_equal(sample.propensity, expit(1.4 * sample.X[:, 0]))
        if target == "full":
            np.testing.assert_array_equal(sample.oracle_tate, sim.true_tate())
        else:
            assert sample.oracle_tate is None
    expected = target_moments("overlap")[1]
    reference = sim.true_target_tate_mc(target="overlap", n_x=30000, rng=982)
    np.testing.assert_allclose(reference, expected, atol=0.04)
    with pytest.raises(ValueError):
        sim.sample_target(2, target="not-a-target")
    with pytest.raises(RuntimeError, match="positive mass"):
        impossible = TriOracleSimulation(n_cov=1, mu1=[1000], mu2=[1000], beta=[1000], sigma2=0)
        impossible.sample_target(2, target="overlap", rng=1, max_draws=10)


def test_generating_propensity_and_tilts_are_never_hiddenly_clipped():
    pi = np.array([0, 1e-12, 0.2, 0.5, 1 - 1e-12, 1])
    np.testing.assert_array_equal(target_tilt(pi, "full"), np.ones_like(pi))
    np.testing.assert_array_equal(target_tilt(pi, "overlap"), pi * (1 - pi))
    np.testing.assert_array_equal(target_tilt(pi, "matching"), np.minimum(pi, 1 - pi))
    np.testing.assert_array_equal(target_tilt(pi, "sqrt"), np.sqrt(pi * (1 - pi)))
    for bad in ([np.nan], [-0.1], [1.1]):
        with pytest.raises(ValueError):
            target_tilt(bad, "overlap")
    for cls in (TriOracleSimulation, TopologicalCausalSimulation):
        sim = cls(n_cov=1, mu1=[-1], mu2=[2], beta=[1])
        X = np.array([[-25], [25]])
        np.testing.assert_array_equal(sim.propensity_score(X), expit(X[:, 0]))
        assert sim.propensity_score(X)[0] < 1e-10


def test_level_b_cached_truth_grid_and_target_metadata_use_plain_units():
    sim = TopologicalCausalSimulation(
        resolution=13, homology_dims=(0, 1), pts_per_loop=8,
        background=2, max_loops=1, seed=8,
    )
    sample = sim.sample_target(5, target="overlap", rng=42)
    assert sample.target == "overlap"
    assert sample.normalization == "plain"
    assert sample.potential_outcomes.shape == (5, 2, 2, 13)
    np.testing.assert_array_equal(sample.tseq, np.linspace(*sim.interval, 13))
    for i in range(5):
        for a in (0, 1):
            for d in range(2):
                np.testing.assert_allclose(
                    sample.potential_outcomes[i, a, d],
                    plain_tents(sample.diagrams[i][a][d], sample.tseq, 3), atol=1e-14,
                )
    np.testing.assert_array_equal(sample.oracle_itte,
                                  sample.potential_outcomes[:, 1] - sample.potential_outcomes[:, 0])
    plain_at_one, _ = sim.silhouettes_at_r(sample, 1)
    legacy_at_one, _ = sim.silhouettes_at_r(replace(sample, normalization="legacy_gudhi"), 1)
    np.testing.assert_allclose(legacy_at_one, np.sqrt(2) * plain_at_one)
    with pytest.raises(ValueError):
        sim.silhouettes_at_r(sample, 0.1)
    assert sim.true_tate_mc(n_x=3, n_mc=1, rng=9, target="matching").shape == (2, 13)


def test_orbit_map_updates_y_using_the_new_x():
    # A witness for the established sequential map; migration preserves it.
    rng = np.random.default_rng(17)
    x, y = rng.random(1)[0], rng.random(1)[0]
    expected_x = (x + 3.5 * y * (1 - y)) % 1
    expected_y = (y + 3.5 * expected_x * (1 - expected_x)) % 1
    orbit, _ = gen_orbits(rhos=(3.5,), num_pts=1, num_orbits_each=1, rng=17)
    np.testing.assert_allclose(orbit[0, 0], [expected_x, expected_y])


def test_empirical_sars_pairing_is_iid_and_distinguishes_sample_population_means():
    infected = np.zeros((2, 1, 4))
    noninfected = np.ones((2, 1, 4))
    expected = np.random.default_rng(7).binomial(1, 0.75, size=20)
    result = make_sarscov2_causal(infected, noninfected, n=20, p=0.75, rng=7)
    np.testing.assert_array_equal(result["potential_outcomes"][:, 1, 0, 0], expected)
    np.testing.assert_allclose(result["population_effect"], 0.75)
    np.testing.assert_allclose(result["true_effect"], expected.mean())


def test_pacce_records_exploratory_scope_and_honors_fixed_geometry_scales():
    demo, lesion = synthetic_pacce_frames(n=12, seed=1)
    data = load_pacce_from_frames(demo, lesion)
    assert "descriptive/associational" in data.extras["analysis_scope"]
    assert "not established" in data.extras["causal_identification"]
    assert data.extras["standardization_source"] == "selected_analysis_cohort"
    spec = load_outcome_spec()
    spec["standardization"] = {"mean": [0.1, 2.0], "sd": [0.3, 0.8]}
    fixed = load_pacce_from_frames(demo, lesion, spec=spec)
    assert fixed.spec["standardization"] == spec["standardization"]
    assert fixed.extras["standardization_source"] == "externally_fixed"
    spec["standardization"]["sd"] = [0, 0.8]
    with pytest.raises(ValueError, match="positive sds"):
        compute_phi({1: np.array([[0, 0], [1, 1]])}, spec)
    # All subjects can fail treatment-affected visit eligibility without making
    # an empty topology array or manifest crash.
    baseline_only = lesion[lesion["DOSREFDY"] <= 0]
    empty = load_pacce_from_frames(demo, baseline_only)
    assert empty.phi.shape == (0, 2, 100)
    assert len(empty.manifest) == len(demo)
