"""Verify the published production checkpoint identities and source provenance.

This read-only check runs without the scientific dependencies. It checks the
saved evidence, not the universal validity of an inference theorem.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_hashes(path):
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None, path
        return {name: hashlib.sha256(archive.read(name)).hexdigest()
                for name in archive.namelist()}


def main():
    provenance = load(HERE / "sources.json")
    bundle = HERE / "revision_source.zip"
    assert digest(bundle) == provenance["source_bundle_sha256"]
    frozen = archive_hashes(bundle)
    mapping = {row["run"]: row for row in load(HERE / "sources/primary_source_mapping.json")}
    historical = {}
    totals = {"primary": 0, "remaining": 0}
    sidecars = 0
    replay_script_hash = digest(HERE / "augment_pacce_mean_tests.py")
    for setting in provenance["retained_settings"]:
        family = setting["family"]
        run = HERE / "results" / (family + "_production") / setting["setting"]
        manifest = load(run / "manifest.json")
        cfg = manifest["configuration"]
        expected = cfg["reps"]
        assert setting["replications"] == expected, run
        if family == "primary":
            record = mapping[setting["setting"]]
            archive = HERE / "sources" / Path(record["source_archive"]).name
            assert digest(archive) == record["archive_sha256"], archive
            if archive not in historical:
                historical[archive] = archive_hashes(archive)
            hashes = historical[archive]
        else:
            hashes = frozen
            assert expected == 200
            assert not list(run.glob("failure_*.json")), run
        for name, checksum in manifest["source_sha256"].items():
            assert hashes.get(name) == checksum, (run, name)
        paths = sorted(run.glob("rep_*.json"))
        assert [path.name for path in paths] == [f"rep_{i:06d}.json" for i in range(expected)], run
        for rep, path in enumerate(paths):
            data = load(path)
            assert data["rep"] == rep and data["seed"] == cfg["seed"] + rep * 100003, path
            assert data["rows"] and all(0 <= row["coverage"] <= 1 for row in data["rows"]), path
            if family == "remaining" and cfg["kind"] == "pacce":
                replay = load(run / "mean_zero_tests" / path.name)
                assert replay["rep"] == rep and replay["seed"] == data["seed"]
                assert replay["original_checkpoint_sha256"] == digest(path)
                assert replay["sidecar_script_sha256"] == replay_script_hash
                assert replay["frozen_adapter_sha256"] == frozen["revision_20261010/rerun/revised_experiments.py"]
                for row in replay["rows"]:
                    original = next(item for item in data["rows"]
                                    if item["metric"] == "TATE" and item["degree"] == row["degree"])
                    for key, value in row["agreement_with_original"].items():
                        assert abs(value - original[key]) <= 1e-12 * abs(original[key]) + 1e-14
                    assert isinstance(row["reject_zero"], bool)
                sidecars += 1
        totals[family] += expected
    assert totals == {"primary": 6600, "remaining": 4600}, totals
    assert sidecars == 1400
    assert sum(totals.values()) == provenance["total_production_replications"] == 11200
    print(json.dumps({"production_replications": sum(totals.values()),
                      "by_family": totals, "pacce_mean_stage_replays": sidecars,
                      "source_hashes_and_checkpoint_identities": "PASS",
                      "scope": "Saved numerical records and source provenance; no theorem certificate"}, indent=2))


if __name__ == "__main__":
    main()
