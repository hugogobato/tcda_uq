"""Check the Git index against the reviewed PACCE publication scope.

No patient data are needed. Only indexed blobs (the next commit), including
archive members, are read; private untracked and unstaged files are ignored.
This is a release allowlist and schema check, not a proof of anonymization.
"""
from __future__ import annotations

import argparse
import ast
import bz2
import csv
import gzip
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CLINICAL = "reproduction/results/clinical_aggregates/"
REVISION = CLINICAL + "coauthor_revision_20261026/"
MANIFEST = "reproduction/publication_files.json"
RAW_SUFFIXES = {".sas7bdat", ".sas7bcat", ".xpt", ".sd2", ".sd7"}
SOURCE_SUFFIXES = {".py", ".sh", ".md", ".rst", ".toml", ".cff"}
FORBIDDEN_FIELDS = {
    "subjid", "usubjid", "subjectid", "subject_id", "subjectids", "subject_ids",
    "patientid", "patient_id", "patientids", "patient_ids", "pilot_ids", "analysis_ids",
    "eligible_ids", "processed_subjects", "individual_trajectories", "itte_curves",
    "prediction_shapes", "sas_dir", "input_sha256",
}
CLINICAL_MARKER = re.compile(r"(?:pacce|clinical|patients?|subjects?)", re.I)
INDIVIDUAL_ARTIFACT = re.compile(
    r"(?:^|[/_.-])(?:processed_subjects?|subject_level|patient_level|analysis_manifest|"
    r"prediction_shapes|individual_(?:effects?|trajectories|curves)|"
    r"itte(?:_curves?|_predictions?)?)(?:[./_-]|$)", re.I
)
SYNTHETIC_ROOTS = (
    "reproduction/results/primary_production/", "reproduction/results/remaining_production/",
    "reproduction/results/exploratory/", "reproduction/results/new_runs/",
    "reproduction/results/empirical_pool_inputs/", "reproduction/legacy/out/",
    "reproduction/legacy/figures/",
)
SAS_MAGIC = bytes.fromhex("0000000000000000c2ea8160b31411cfbd92080009c7318c181f1011")
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 20000
MAX_DEPTH = 8

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
SPLIT_CSV = {
    "adjusted_selected_contrast.csv": {"t", "estimate", "lower", "upper"},
    "fixed_profile_diagnostics.csv": {"backbone", "burden", "exact_matching", "bin_matching",
                                      "peak_abs_estimate", "width", "zero_excluded_fraction"},
    "selection_weight_sensitivity.csv": {"gamma", "n_eligible", "n_released_analysis", "peak_abs_weighted_contrast"},
}
REVISION_CSV = {
    "ctate_profile_summary.csv": {"profile", "age_years", "folds", "n", "n_matching", "peak_abs_tau",
                                  "t_at_peak", "mean_halfwidth", "mean_full_width", "excursion_frac",
                                  "sup_stat", "sup_pvalue", "bonferroni_p", "band_seed", "test_seed"},
    "independent_fold_sensitivity.csv": {"folds", "seed", "n", "pvalue", "peak_abs_estimate",
                                         "mean_full_width", "zero_included_all", "fold_heldout_arm_counts"},
}
CLINICAL_SCHEMAS = {CLINICAL + "main_cohort/" + name: fields for name, fields in MAIN_CSV.items()}
CLINICAL_JSON = set()
for _cohort in ("pacce_pilot_grid1", "pacce_selected"):
    for _basis in (3, 5, 9):
        _directory = CLINICAL + f"split_sample/{_cohort}_basis{_basis}/"
        CLINICAL_SCHEMAS.update({_directory + name: fields for name, fields in SPLIT_CSV.items()})
        CLINICAL_JSON.update({_directory + "manifest.json", _directory + "scope.json"})
CLINICAL_SCHEMAS.update({REVISION + name: fields for name, fields in REVISION_CSV.items()})
CLINICAL_METADATA = {CLINICAL + "main_cohort/ctate_profiles.yaml"}
CLINICAL_READMES = {CLINICAL + "README.md", REVISION + "README.md"}
CLINICAL_ALLOWLIST = set(CLINICAL_SCHEMAS) | CLINICAL_JSON | CLINICAL_METADATA | CLINICAL_READMES
PROFILE_LABELS = {f"{arm}_{burden}_burden" for arm in ("oxali", "iri") for burden in ("low", "mid", "high")}


class ReleaseError(ValueError):
    """The indexed publication does not satisfy the release scope."""


def fail(path: str, reason: str) -> None:
    # Do not include data values or complete parsed records in diagnostics.
    raise ReleaseError(f"{path}: {reason}")


def field_name(value: str) -> str:
    return re.sub(r"[\s-]+", "_", value.strip().lower())


def check_fields(value, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if field_name(str(key)) in FORBIDDEN_FIELDS:
                fail(path, "restricted subject/provenance field in structured data")
            check_fields(item, path)
    elif isinstance(value, list):
        for item in value:
            check_fields(item, path)


def index_blobs(root: Path):
    """Yield paths and exact stage-zero blob bytes, without reading the worktree."""
    entries = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=root).split(b"\0")
    with subprocess.Popen(["git", "cat-file", "--batch"], cwd=root,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        assert process.stdin is not None and process.stdout is not None
        for entry in entries:
            if not entry:
                continue
            metadata, raw_path = entry.split(b"\t", 1)
            mode, object_id, stage = metadata.split()
            path = raw_path.decode("utf-8", errors="strict")
            if stage != b"0":
                fail(path, "unmerged index entry")
            if mode not in {b"100644", b"100755"}:
                fail(path, "symlinks and submodules require explicit release review")
            process.stdin.write(object_id + b"\n")
            process.stdin.flush()
            response = process.stdout.readline().split()
            if len(response) != 3 or response[1] != b"blob":
                fail(path, "cannot read indexed blob")
            size = int(response[2])
            if size > MAX_MEMBER_BYTES:
                fail(path, "blob exceeds inspection size limit")
            data = process.stdout.read(size)
            if len(data) != size or process.stdout.read(1) != b"\n":
                fail(path, "incomplete indexed blob")
            yield path, data
        process.stdin.close()
        if process.wait() != 0:
            raise ReleaseError("git cat-file failed")


class Inspector:
    def __init__(self):
        self.archive_members = 0
        self.archive_count = 0
        self.expanded_bytes = 0
        self.notebooks = 0

    def inspect(self, path: str, data: bytes, *, depth: int = 0, synthetic: bool = False) -> None:
        name = PurePosixPath(path.rsplit("!", 1)[-1])
        suffix = name.suffix.lower()
        synthetic = synthetic or path.startswith(SYNTHETIC_ROOTS)
        if depth > MAX_DEPTH:
            fail(path, "nested archive depth exceeds inspection limit")
        transport = (data.startswith(b"HEADER RECORD*******")
                     and (b"LIBRARY HEADER RECORD" in data[:80] or b"LIBV8 HEADER RECORD" in data[:80]))
        if suffix in RAW_SUFFIXES or data.startswith(SAS_MAGIC) or transport:
            fail(path, "raw SAS/XPT clinical data are prohibited")
        if suffix in {".7z", ".rar"} or data.startswith((b"7z\xbc\xaf\x27\x1c", b"Rar!")):
            fail(path, "archive format cannot be inspected by this guard")
        source = suffix in SOURCE_SUFFIXES or path in CLINICAL_METADATA
        clinical = bool(CLINICAL_MARKER.search(path))
        if not source and suffix != ".ipynb" and INDIVIDUAL_ARTIFACT.search(str(name)) and clinical and not synthetic:
            fail(path, "clinical individual-result or subject-manifest artifact is prohibited")

        if zipfile.is_zipfile(io.BytesIO(data)):
            self.archive_count += 1
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    self._member(path, member.filename, member.file_size)
                    if (member.external_attr >> 16) & 0o170000 == 0o120000:
                        fail(path, "archive contains a symbolic link")
                    self.inspect(path + "!" + member.filename, archive.read(member), depth=depth + 1, synthetic=synthetic)
            if clinical and not synthetic and suffix == ".npz":
                fail(path, "clinical array bundles are outside the aggregate allowlist")
            return
        if suffix in {".zip", ".npz"}:
            fail(path, "invalid or uninspectable ZIP container")
        # A TAR is recognized by content; compressed TAR streams are also handled.
        if data[257:262] == b"ustar" or suffix in {".tar", ".tgz", ".tbz2", ".txz"}:
            self.archive_count += 1
            try:
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
                    for member in archive:
                        if member.isdir():
                            continue
                        if not member.isfile():
                            fail(path, "archive contains a link or special file")
                        self._member(path, member.name, member.size)
                        stream = archive.extractfile(member)
                        assert stream is not None
                        self.inspect(path + "!" + member.name, stream.read(), depth=depth + 1, synthetic=synthetic)
            except tarfile.TarError:
                fail(path, "invalid or uninspectable TAR container")
            return
        compression = None
        if data.startswith(b"\x1f\x8b"):
            compression = lambda stream: gzip.GzipFile(fileobj=stream, mode="rb")
        elif data.startswith(b"BZh"):
            compression = lambda stream: bz2.BZ2File(stream, "rb")
        elif data.startswith(b"\xfd7zXZ\x00"):
            compression = lambda stream: lzma.LZMAFile(stream, "rb")
        if compression:
            self.archive_count += 1
            with compression(io.BytesIO(data)) as stream:
                unpacked = stream.read(MAX_MEMBER_BYTES + 1)
            member_name = str(name.with_suffix(""))
            self._member(path, member_name, len(unpacked))
            self.inspect(path + "!" + member_name, unpacked, depth=depth + 1, synthetic=synthetic)
            return

        if suffix == ".ipynb":
            self.notebooks += 1
            notebook = json.loads(data)
            check_fields(notebook, path)
            for cell in notebook.get("cells", []):
                if cell.get("outputs") or cell.get("attachments"):
                    fail(path, "notebook outputs/attachments must be cleared before release")
            # Source cells may refer to controlled field names without containing records.
            return
        if source:
            return
        if suffix == ".json":
            check_fields(json.loads(data), path)
        elif suffix in {".jsonl", ".ndjson"}:
            for line in data.splitlines():
                if line.strip():
                    check_fields(json.loads(line), path)
        elif suffix in {".csv", ".tsv"}:
            text = data.decode("utf-8-sig")
            rows = csv.reader(io.StringIO(text), delimiter="\t" if suffix == ".tsv" else ",")
            header = next(rows, [])
            if any(field_name(column) in FORBIDDEN_FIELDS for column in header):
                fail(path, "restricted subject/provenance column in table")
        elif data.startswith(b"\x93NUMPY"):
            version = data[6]
            size_bytes = 2 if version == 1 else 4
            offset = 8 + size_bytes
            header_size = int.from_bytes(data[8:offset], "little")
            metadata = ast.literal_eval(data[offset:offset + header_size].decode("latin1").strip())
            dtype = metadata.get("descr", "")
            if isinstance(dtype, list):
                if any(field_name(str(field[0])) in FORBIDDEN_FIELDS for field in dtype):
                    fail(path, "restricted subject field in array metadata")
            if "O" in str(dtype):
                fail(path, "object arrays require separate review and cannot be safely inspected")
        elif clinical and not synthetic and suffix not in {".pdf", ".png", ".txt", ".log"}:
            fail(path, "clinical binary or table format is outside the reviewed aggregate scope")

    def _member(self, parent: str, name: str, size: int) -> None:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            fail(parent, "archive has an unsafe member path")
        self.archive_members += 1
        self.expanded_bytes += size
        if size > MAX_MEMBER_BYTES or self.expanded_bytes > MAX_EXPANDED_BYTES or self.archive_members > MAX_MEMBERS:
            fail(parent, "archive exceeds inspection resource limit")


def audit_index(root: Path = ROOT) -> dict:
    inspector = Inspector()
    clinical_blobs = {}
    clinical_figures = {}
    manifest_data = None
    tracked = 0
    for path, data in index_blobs(root):
        tracked += 1
        inspector.inspect(path, data)
        if path == MANIFEST:
            manifest_data = data
        if path.startswith(CLINICAL):
            if path not in CLINICAL_ALLOWLIST:
                fail(path, "clinical file is outside the explicit release allowlist")
            clinical_blobs[path] = data
        elif CLINICAL_MARKER.search(path) and PurePosixPath(path).suffix.lower() in {".pdf", ".png"}:
            clinical_figures[path] = data
        elif (CLINICAL_MARKER.search(path) and not path.startswith(SYNTHETIC_ROOTS)
              and PurePosixPath(path).suffix.lower() not in SOURCE_SUFFIXES | {".ipynb"}):
            fail(path, "clinical data/result artifact is outside the reviewed publication locations")
    if manifest_data is None:
        fail(MANIFEST, "indexed publication hash manifest is missing")
    records = json.loads(manifest_data).get("files", [])
    hashes = {}
    for record in records:
        path = record["path"]
        if path in hashes:
            fail(MANIFEST, "duplicate artifact path")
        hashes[path] = record["sha256"]
    released = []
    for path, data in sorted({**clinical_blobs, **clinical_figures}.items()):
        sha256 = hashlib.sha256(data).hexdigest()
        if path not in CLINICAL_READMES:
            if hashes.get(path) != sha256:
                fail(path, "indexed bytes do not match the publication SHA256 allowlist")
            released.append({"path": path, "sha256": sha256})
        if path in CLINICAL_SCHEMAS:
            reader = csv.reader(io.StringIO(data.decode("utf-8-sig")))
            header = next(reader, [])
            if len(header) != len(set(header)) or set(header) != CLINICAL_SCHEMAS[path]:
                fail(path, "clinical CSV columns do not match the reviewed aggregate schema")
            rows = list(reader)
            if any(len(row) != len(header) for row in rows):
                fail(path, "clinical CSV has a malformed row")
            if path.startswith(REVISION) and len(rows) != 6:
                fail(path, "revision table must contain exactly six aggregate rows")
            if path == REVISION + "ctate_profile_summary.csv":
                records = [dict(zip(header, row)) for row in rows]
                if ({row["profile"] for row in records} != PROFILE_LABELS
                        or any(float(row["age_years"]) != 61 or row["folds"] != "treatment_stratified" for row in records)):
                    fail(path, "revision CTATE table must use the six fixed age-61 aggregate profiles")
            elif path == REVISION + "independent_fold_sensitivity.csv":
                records = [dict(zip(header, row)) for row in rows]
                rules = [row["folds"] for row in records]
                if rules.count("treatment_stratified") != 1 or rules.count("independent_KFold") != 5:
                    fail(path, "fold summary must contain one stratified and five independent aggregate runs")
                for row in records:
                    counts = json.loads(row["fold_heldout_arm_counts"])
                    if (not isinstance(counts, list) or len(counts) != 2
                            or any(not isinstance(pair, list) or len(pair) != 2
                                   or any(type(count) is not int or count < 0 for count in pair) for pair in counts)):
                        fail(path, "fold support must consist only of two aggregate arm counts per fold")
        elif path in CLINICAL_JSON:
            check_fields(json.loads(data), path)
    # Deletions of historical results and missing allowlisted artifacts are also failures.
    expected = set(CLINICAL_SCHEMAS) | CLINICAL_JSON | CLINICAL_METADATA
    # The correction is optional until staged; when any part is staged both tables are required.
    if not any(path.startswith(REVISION) for path in clinical_blobs) and not any(path.startswith(REVISION) for path in hashes):
        expected -= {REVISION + name for name in REVISION_CSV}
    missing = sorted(expected - clinical_blobs.keys())
    if missing:
        fail(missing[0], "reviewed clinical aggregate file missing from the index")
    return {
        "status": "PASS", "scope": "Git index blobs only; untracked/unstaged private files excluded",
        "limitation": "Release allowlist and schema checks; not a proof of anonymization or absence of all possible disclosures",
        "tracked_files": tracked, "clinical_aggregate_files": len(clinical_blobs) - len(set(clinical_blobs) & CLINICAL_READMES),
        "archives_inspected": inspector.archive_count, "archive_members_inspected": inspector.archive_members,
        "cleared_notebooks": inspector.notebooks, "reviewed_clinical_artifacts": released,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Git repository to inspect")
    parser.add_argument("--report", type=Path, help="Write the path/hash-only report as JSON")
    args = parser.parse_args()
    try:
        report = audit_index(args.root.resolve())
    except Exception as error:
        # Parse exceptions can contain data values; report only their type unless deliberately scoped.
        message = str(error) if isinstance(error, ReleaseError) else type(error).__name__
        parser.exit(1, "PACCE release check FAILED: " + message + "\n")
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered)
        summary = {key: value for key, value in report.items() if key != "reviewed_clinical_artifacts"}
        print(json.dumps(summary, indent=2))
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
