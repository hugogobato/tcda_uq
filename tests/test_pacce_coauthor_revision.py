"""Portable PACCE reproduction checks using synthetic data exclusively."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "reproduction/pacce_coauthor_revision.py"
SPEC = importlib.util.spec_from_file_location("pacce_revision_runner", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_profiles_use_age_in_years_and_the_prespecified_six_queries():
    names, X = runner.profiles(np.ones(12, dtype=bool))
    assert len(names) == len(set(names)) == 6
    np.testing.assert_array_equal(X[:, 1], np.full(6, 61.))
    np.testing.assert_array_equal(X[:, 0], [0., 0., 0., 1., 1., 1.])
    np.testing.assert_array_equal(X[:, 4], [2., 4., 6., 2., 4., 6.])
    np.testing.assert_array_equal(X[:, -2:], [[0, 0], [1, 0], [0, 1]] * 2)


def test_external_output_boundary_resolves_symlinks(tmp_path):
    with pytest.raises(ValueError, match="outside this checkout"):
        runner.outside_checkout(runner.CHECKOUT / "reproduction/results/new-output")
    alias = tmp_path / "checkout-alias"
    alias.symlink_to(runner.CHECKOUT, target_is_directory=True)
    with pytest.raises(ValueError, match="outside this checkout"):
        runner.outside_checkout(alias / "new-output")
    assert runner.outside_checkout(tmp_path / "aggregates") == tmp_path / "aggregates"


def test_frozen_extraction_authenticates_code_only_and_rejects_tampering(tmp_path, monkeypatch):
    with runner.frozen_package() as root:
        extracted = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
        assert extracted == set(runner.FROZEN_SHA256)
        assert all(name.endswith(".py") for name in extracted)
        assert not any("/tests/" in name or "/scripts/" in name for name in extracted)
    corrupted = tmp_path / "corrupted.zip"
    corrupted.write_bytes(b"not the authenticated historical archive")
    monkeypatch.setattr(runner, "SOURCE_ARCHIVE", str(corrupted))
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        with runner.frozen_package():
            pytest.fail("tampered code must never be extracted")


def test_member_checksums_are_enforced(monkeypatch):
    changed = dict(runner.FROZEN_SHA256)
    changed["tcda_uq/uq/conformal/weighted_cp.py"] = "0" * 64
    monkeypatch.setattr(runner, "FROZEN_SHA256", changed)
    with pytest.raises(ValueError, match="member checksum mismatch"):
        with runner.frozen_package():
            pytest.fail("a scientific member with a wrong digest must be rejected")


def test_constant_scores_are_calibration_independent_in_both_frozen_arms():
    # A fresh isolated interpreter avoids the current tcda_uq package already
    # imported by other tests. Both fits use exactly the runner's arm builder.
    program = r'''
import importlib.util
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
spec = importlib.util.spec_from_file_location("runner", sys.argv[2])
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
class Propensity:
    def predict_proba(self, X):
        return np.tile([.4, .6], (len(X), 1))
class Nuisance:
    tseq = np.array([0., .4, .83])
    prop_model = Propensity()
    def predict_mu(self, X):
        return [(np.zeros((len(X), 3)), np.full((len(X), 3), .25))]
nu = Nuisance()
A = np.arange(40) % 2
X = np.linspace(0., 1., 40)[:, None]
phi = np.arange(120, dtype=float).reshape(40, 1, 3) / 120
changed = phi * 100 + 7
for arm in (0, 1):
    fitted = runner.constant_arm(nu, (phi, A, X), arm)
    altered = runner.constant_arm(nu, (changed, A, X), arm)
    np.testing.assert_array_equal(fitted.s, np.ones(3))
    np.testing.assert_array_equal(fitted.s, altered.s)
    expected = np.max(np.abs(phi[A == arm, 0] - .25 * arm), axis=1)
    np.testing.assert_allclose(fitted.scores, expected)
    assert not np.array_equal(fitted.scores, altered.scores)
    radius = fitted.radius([[.3]], .05)[0]
    assert radius == np.max(expected)  # 20 points + test atom, equal arm weights
'''
    with runner.frozen_package() as root:
        result = subprocess.run([sys.executable, "-I", "-B", "-c", program, str(root), str(SCRIPT)],
                                capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_width_arithmetic_and_support_intersection():
    class Arm:
        def __init__(self, radius, scale):
            self._radius = radius
            self.s = np.asarray(scale)

        def radius(self, X, alpha):
            assert alpha == .05
            return np.full(len(X), self._radius)

    class Model:
        arms0 = [Arm(2., [1., 2., 3.])]
        arms1 = [Arm(3., [2., 2., 2.])]

        def band_bounds(self, X, alpha):
            assert alpha == .1
            half = 2 * self.arms0[0].s + 3 * self.arms1[0].s
            center = np.zeros((len(X), 3))
            return center - half, center + half, center

    t = np.array([0., .4, .83])
    row = runner.width_summaries(Model(), [[0.]], ["synthetic_profile"], t)[0]
    assert row["mean_full_width"] == pytest.approx(2 * (2 * 2 + 3 * 2))
    assert row["width_from_arm_radii"] == pytest.approx(row["mean_full_width"])
    assert row["full_support_envelope_mean_width"] == pytest.approx(np.mean(2 * np.sqrt(2) * t))
    assert row["support_intersected_mean_full_width"] == pytest.approx(row["full_support_envelope_mean_width"])
    assert row["support_intersection_empty_nodes"] == 0
    assert row["support_intersection_zero_included_all"]


def test_help_does_not_require_loading_historical_code():
    result = subprocess.run([sys.executable, "-I", str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--sas-dir" in result.stdout and "--include-fold-sensitivity" in result.stdout


def test_serializer_rejects_record_fields_before_writing(tmp_path):
    tables = {filename: [] for filename in runner.OUTPUT_COLUMNS}
    columns = runner.OUTPUT_COLUMNS["ctate_age61_profiles.csv"]
    tables["ctate_age61_profiles.csv"] = [{**dict.fromkeys(columns, 0), "SUBJID": "synthetic"}]
    out_dir = tmp_path / "rejected"
    with pytest.raises(ValueError, match="Unexpected aggregate columns"):
        runner._write_outputs(out_dir, tables, {})
    assert not out_dir.exists()


def test_synthetic_cli_writes_only_allowlisted_aggregate_outputs(tmp_path):
    out_dir = tmp_path / "fixture-aggregates"
    result = subprocess.run([
        sys.executable, "-I", str(SCRIPT), "--fixture", "--out-dir", "fixture-aggregates",
        "--ctate-boot", "9", "--fold-boot", "11", "--include-fold-sensitivity",
    ], cwd=tmp_path, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    assert {path.name for path in out_dir.iterdir()} == {*runner.OUTPUT_COLUMNS, "method_provenance.json"}
    tables = {}
    for filename, columns in runner.OUTPUT_COLUMNS.items():
        with (out_dir / filename).open() as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == list(columns)
            tables[filename] = list(reader)
    assert len(tables["ctate_age61_profiles.csv"]) == 6
    assert len(tables["prediction_width_summary.csv"]) == 6
    assert len(tables["arm_score_diagnostics.csv"]) == 6
    assert len(tables["fold_diagnostics.csv"]) == 6
    for row in tables["prediction_width_summary.csv"]:
        assert float(row["age_years"]) == 61.
        assert float(row["arm0_scale_mean"]) == float(row["arm1_scale_mean"]) == 1.
        assert float(row["mean_full_width"]) == pytest.approx(float(row["width_from_arm_radii"]))
        assert float(row["per_arm_alpha"]) == .05
    provenance = json.loads((out_dir / "method_provenance.json").read_text())
    assert provenance["data_kind"] == "synthetic fixture"
    assert provenance["profile_age_years"] == 61.
    assert provenance["prediction"]["outcome_fourier_functions"] == 5
    assert provenance["ctate"]["second_stage_fourier_functions"] == 4
    assert provenance["ctate"]["bootstrap_draws"] == 9
    assert provenance["fold_sensitivity"]["bootstrap_draws"] == 11
    assert str(tmp_path) not in json.dumps(provenance)
    forbidden = {"SUBJID", "subject_id", "patient_id", "input_sha256", "sas_dir", "train_index", "calib_index",
                 "phi", "lower", "upper", "center", "scores", "residuals", "pi_hat"}

    def check_keys(obj):
        if isinstance(obj, dict):
            assert not forbidden.intersection(obj)
            for value in obj.values():
                check_keys(value)
        elif isinstance(obj, list):
            for value in obj:
                check_keys(value)

    check_keys(provenance)
    for rows in tables.values():
        for row in rows:
            check_keys(row)


def test_controlled_data_cannot_change_the_frozen_bootstrap_counts(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), "--sas-dir", str(tmp_path),
                             "--out-dir", str(tmp_path / "out"), "--ctate-boot", "9"],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "requires 1999 CTATE and 4999 fold" in result.stderr
    assert not (tmp_path / "out").exists()
