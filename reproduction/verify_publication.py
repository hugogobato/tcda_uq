"""Verify additional public result artifacts and their clinical release scope.

Uses only the standard library; no clinical inputs or scientific reruns.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FORBIDDEN_FIELDS = {"subjid", "usubjid", "subject_id", "patient_id", "pilot_ids",
                    "input_sha256", "sas_dir", "out"}
SPLIT_CSV = {
    "adjusted_selected_contrast.csv": {"t", "estimate", "lower", "upper"},
    "fixed_profile_diagnostics.csv": {"backbone", "burden", "exact_matching", "bin_matching",
                                      "peak_abs_estimate", "width", "zero_excluded_fraction"},
    "selection_weight_sensitivity.csv": {"gamma", "n_eligible", "n_released_analysis",
                                         "peak_abs_weighted_contrast"},
}
MAIN_CSV = {
    "tate_estimates.csv": {"tseq", "psi_hat", "lower", "upper"},
    "sldfun_tate_estimates.csv": {"tseq", "psi_hat", "lower", "upper"},
    "ctate_curves.csv": {"profile", "t", "center", "lower", "upper"},
    "tate_test.csv": {"pvalue", "reject", "statistic", "mean_width", "n"},
    "sldfun_tate_test.csv": {"pvalue", "reject", "mean_width"},
    "ctate_profile_summary.csv": {"profile", "burden", "chemo_iri", "n_matching", "peak_abs_tau",
                                  "t_at_peak", "mean_halfwidth", "excursion_frac", "sup_stat",
                                  "sup_pvalue", "bonferroni_p"},
    "conventional_benchmark.csv": {"benchmark", "arm", "n", "value", "estimate", "denominator"},
    "baseline_balance.csv": {"covariate", "mean_arm1", "mean_arm0", "sd_arm1", "sd_arm0", "smd", "n_arm1", "n_arm0"},
    "domain_subject_counts.csv": {"domain", "rows", "n_subjects"},
    "placebo_checks.csv": {"tag", "pvalue", "reject"},
    "sensitivity_forest.csv": {"variant", "n_eligible", "pvalue", "reject"},
    "missingness_theory_checks.csv": {"check", "gamma", "peak_abs_effect", "peak_t", "sup_abs", "n_weighted"},
    "topology_preprocessing_sensitivity.csv": {"variant", "n_subjects", "t_max", "h1_freq", "h1_strong_freq",
                                              "h1_totpers_med", "h0_totpers_med", "h0_l1_med",
                                              "corr_h0l1_blsld", "corr_h0l1_pctchg"},
}
REVISION_CSV = {
    "ctate_profile_summary.csv": {"profile", "age_years", "folds", "n", "n_matching",
        "peak_abs_tau", "t_at_peak", "mean_halfwidth", "mean_full_width", "excursion_frac",
        "sup_stat", "sup_pvalue", "bonferroni_p", "band_seed", "test_seed"},
    "independent_fold_sensitivity.csv": {"folds", "seed", "n", "pvalue", "peak_abs_estimate",
        "mean_full_width", "zero_included_all", "fold_heldout_arm_counts"},
}


def check_fields(value, path: Path) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            assert key.lower() not in FORBIDDEN_FIELDS, (path, key)
            check_fields(item, path)
    elif isinstance(value, list):
        for item in value:
            check_fields(item, path)


def main() -> None:
    manifest = json.loads((HERE / "publication_files.json").read_text())
    paths = [record["path"] for record in manifest["files"]]
    assert len(paths) == len(set(paths)), "Repeated artifact paths"
    for record in manifest["files"]:
        relative = Path(record["path"])
        assert not relative.is_absolute() and ".." not in relative.parts, relative
        assert relative.parts[0] == "reproduction", relative
        assert relative.suffix not in {".tex", ".bib", ".bbl", ".sas7bdat", ".xpt"}, relative
        path = ROOT / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], path
    clinical = HERE / "results/clinical_aggregates"
    clinical_count = 0
    for path in sorted(clinical.rglob("*")):
        if not path.is_file() or path.suffix == ".md":
            continue
        assert str(path.relative_to(ROOT)) in paths, path
        assert not any(word in path.name.lower() for word in
                       ("itte", "analysis_manifest", "processed_subject", "prediction_shapes")), path
        if path.suffix == ".csv":
            schema = (REVISION_CSV if path.parent.name == "coauthor_revision_20261026" else
                      SPLIT_CSV if path.parent.parent.name == "split_sample" else MAIN_CSV)
            assert path.name in schema, path
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                assert set(reader.fieldnames or []) == schema[path.name], (path, reader.fieldnames)
                assert not FORBIDDEN_FIELDS.intersection(name.lower() for name in reader.fieldnames), path
                assert all(len(row) == len(reader.fieldnames) for row in reader), path
        elif path.suffix == ".json":
            assert path.name in {"scope.json", "manifest.json"}, path
            check_fields(json.loads(path.read_text()), path)
        else:
            assert path.name == "ctate_profiles.yaml", path
        clinical_count += 1
    assert clinical_count == 46, clinical_count
    assert not (ROOT / "manuscript").exists(), "Manuscript is maintained separately"
    assert not (ROOT / "submission").exists(), "arXiv files are maintained separately"
    print(json.dumps({"additional_artifacts": len(paths), "clinical_aggregate_files": clinical_count,
                      "artifact_checksums": "PASS", "clinical_release_allowlist": "PASS",
                      "manuscript_and_arxiv_files": "Maintained separately"}, indent=2))


if __name__ == "__main__":
    main()
