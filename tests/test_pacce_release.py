"""Release failures that could expose controlled clinical objects."""
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import zipfile

import pytest


SPEC = importlib.util.spec_from_file_location(
    "verify_pacce_release", Path(__file__).resolve().parents[1] / "reproduction/verify_pacce_release.py"
)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def zip_bytes(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return stream.getvalue()


def stage(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    subprocess.run(["git", "add", "--", name], cwd=root, check=True, capture_output=True)


@pytest.fixture
def indexed_release(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    # Use the retained historical aggregate snapshot as a realistic small Git index.
    required = (set(release.CLINICAL_SCHEMAS) | release.CLINICAL_JSON | release.CLINICAL_METADATA)
    required = {name for name in required if not name.startswith(release.REVISION)}
    manifest = json.loads((release.ROOT / release.MANIFEST).read_text())
    manifest["files"] = [record for record in manifest["files"] if record["path"] in required]
    for name in required:
        stage(tmp_path, name, (release.ROOT / name).read_bytes())
    stage(tmp_path, release.MANIFEST, json.dumps(manifest).encode())
    return tmp_path


def test_index_detects_staged_private_bytes_after_worktree_is_sanitized(indexed_release):
    name = "exports/pacce_subjects.csv"
    stage(indexed_release, name, b"USUBJID,baseline\nRESTRICTED_FIXTURE,0\n")
    (indexed_release / name).write_text("summary,n\naggregate,100\n")
    with pytest.raises(release.ReleaseError, match="restricted subject"):
        release.audit_index(indexed_release)


def test_untracked_private_inputs_are_not_part_of_publication(indexed_release):
    (indexed_release / "pacce.sas7bdat").write_bytes(b"PRIVATE TEST FIXTURE")
    report = release.audit_index(indexed_release)
    assert report["status"] == "PASS"
    assert report["clinical_aggregate_files"] == 44


def test_changed_aggregate_requires_new_reviewed_manifest_hash(indexed_release):
    name = release.CLINICAL + "main_cohort/tate_estimates.csv"
    stage(indexed_release, name, (indexed_release / name).read_bytes() + b"\n")
    with pytest.raises(release.ReleaseError, match="SHA256"):
        release.audit_index(indexed_release)


def test_new_individual_curve_with_no_identifier_still_fails(indexed_release):
    stage(indexed_release, "exports/pacce_itte_curves.csv", b"t,center,lower,upper\n0,0,-1,1\n")
    with pytest.raises(release.ReleaseError, match="individual-result"):
        release.audit_index(indexed_release)


def test_unknown_clinical_aggregate_filename_cannot_bypass_allowlist(indexed_release):
    stage(indexed_release, release.CLINICAL + "main_cohort/new_table.csv", b"n,mean\n100,0\n")
    with pytest.raises(release.ReleaseError, match="explicit release allowlist"):
        release.audit_index(indexed_release)


def test_clinical_table_with_generic_identifier_column_requires_allowlist(indexed_release):
    stage(indexed_release, "exports/pacce_baseline.csv", b"id,age\nTEST_FIXTURE,61\n")
    with pytest.raises(release.ReleaseError, match="reviewed publication locations"):
        release.audit_index(indexed_release)


@pytest.mark.parametrize("name,data", [
    ("imports/pacce.sas7bdat", b"NOT ACTUAL CLINICAL DATA"),
    ("imports/renamed.bin", release.SAS_MAGIC + b"TEST FIXTURE"),
    ("imports/renamed.dat", b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!"),
    ("imports/version8.dat", b"HEADER RECORD*******LIBV8 HEADER RECORD!!!!!!!"),
])
def test_raw_sas_and_transport_magic_are_blocked(name, data):
    with pytest.raises(release.ReleaseError, match="raw SAS/XPT"):
        release.Inspector().inspect(name, data)


def test_nested_compressed_archive_cannot_hide_raw_trial_file():
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        content = b"CONTROLLED TEST FIXTURE"
        entry = tarfile.TarInfo("inputs/trial.xpt")
        entry.size = len(content)
        archive.addfile(entry, io.BytesIO(content))
    archive = zip_bytes({"source/bundle.tar.gz": gzip.compress(stream.getvalue())})
    with pytest.raises(release.ReleaseError, match="raw SAS/XPT"):
        release.Inspector().inspect("source.zip", archive)


def test_json_identifiers_are_detected_in_nested_archive_without_clinical_filename():
    archive = zip_bytes({"nested.zip": zip_bytes({"data.json": b'{"cohort":{"subject_ids":["TEST"]}}'})})
    with pytest.raises(release.ReleaseError, match="restricted subject"):
        release.Inspector().inspect("source.zip", archive)


def test_notebook_source_references_are_allowed_but_saved_outputs_are_blocked():
    notebook = {"cells": [{"cell_type": "code", "source": ["ids = frame['USUBJID']"], "outputs": []}]}
    release.Inspector().inspect("pacce_reanalysis.ipynb", json.dumps(notebook).encode())
    notebook["cells"][0]["outputs"] = [{"output_type": "stream", "text": "CONTROLLED TEST FIXTURE"}]
    with pytest.raises(release.ReleaseError, match="outputs/attachments"):
        release.Inspector().inspect("pacce_reanalysis.ipynb", json.dumps(notebook).encode())


def test_synthetic_pacce_arrays_and_source_field_references_are_allowed():
    npz = next((release.ROOT / "reproduction/results/remaining_production/10_pacce_null").rglob("truth.npz"))
    inspector = release.Inspector()
    inspector.inspect(str(npz.relative_to(release.ROOT)), npz.read_bytes())
    inspector.inspect("source/pacce.py", b"frame['USUBJID']; prediction_shapes = {}\n")
    assert inspector.archive_members == 6


def test_archive_traversal_is_rejected_without_extracting_files():
    with pytest.raises(release.ReleaseError, match="unsafe member"):
        release.Inspector().inspect("source.zip", zip_bytes({"../data.csv": b"n,mean\n100,0\n"}))


def test_hash_update_cannot_authorize_more_than_six_revision_profile_rows(indexed_release):
    manifest = json.loads((indexed_release / release.MANIFEST).read_text())
    for name in release.REVISION_CSV:
        path = release.REVISION + name
        data = (release.ROOT / path).read_bytes()
        if name == "ctate_profile_summary.csv":
            data += data.splitlines(keepends=True)[-1]
        stage(indexed_release, path, data)
        manifest["files"].append({"path": path, "sha256": hashlib.sha256(data).hexdigest()})
    stage(indexed_release, release.MANIFEST, json.dumps(manifest).encode())
    with pytest.raises(release.ReleaseError, match="exactly six"):
        release.audit_index(indexed_release)
