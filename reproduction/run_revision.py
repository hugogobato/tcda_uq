"""Run the paper's fixed remaining studies from their audited frozen source.

The public library is maintained separately. A checksummed source snapshot
keeps numerical reproduction stable when defaults or dependency APIs change.
No clinical records are bundled. Requires the pinned scientific dependencies
from this repository's pyproject.toml.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
BUNDLE_SHA256 = "5d305d861b90420db7f05f4b3a58eaa9e6beef7fad74f8240dfe641be1f31883"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", nargs="+", default=["all"])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hours", type=float, default=8.5)
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()
    preset_dir = HERE / "presets"
    if args.list:
        sys.path.insert(0, str(preset_dir))
        from build_colab_remaining import presets
        for row in presets():
            print(f"{row['id']}: {row['title']} ({len(row['configs'])} configuration(s))")
        return 0
    if not 1 <= args.workers <= 4 or not 0 < args.hours <= 8.5 or args.reps < 1:
        parser.error("require 1–4 workers, 0 < hours <= 8.5 and reps >= 1")
    bundle = HERE / "revision_source.zip"
    if hashlib.sha256(bundle.read_bytes()).hexdigest() != BUNDLE_SHA256:
        raise ValueError("The frozen source bundle failed its SHA256 check")
    # Keep input paths stable across executor-only updates, so the scientific
    # manifest's standardization path remains compatible with checkpoint resume.
    checkout_key = hashlib.sha256(str(HERE.resolve()).encode()).hexdigest()[:12]
    work = Path(tempfile.gettempdir()) / f"tcda_uq_revision_{BUNDLE_SHA256[:12]}_{checkout_key}"
    work.mkdir(exist_ok=True)
    with zipfile.ZipFile(bundle) as archive:
        for name in archive.namelist():
            destination = (work / name).resolve()
            if work.resolve() not in destination.parents:
                raise ValueError("Unsafe source archive path")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.read_bytes() != archive.read(name):
                    raise ValueError(f"Frozen source was modified: {name}")
            else:
                destination.write_bytes(archive.read(name))
    frozen_rerun = work / "revision_20261010/rerun"
    frozen_presets = frozen_rerun / "colab_remaining"
    frozen_presets.mkdir(exist_ok=True)
    for name in ("run_local.py", "build_colab_remaining.py"):
        destination = frozen_presets / name
        source = (preset_dir / name).read_bytes()
        # These two scheduler files are outside the immutable scientific bundle.
        # Refreshing execution logic leaves all scientific source checks intact.
        destination.write_bytes(source)
    # Fresh reproduction outputs remain visible in Git, separately from the
    # immutable paper records. They are not hidden by .gitignore rules.
    for dirname in ("remaining_production", "remaining_execution_checks"):
        actual = HERE / "results" / "new_runs" / dirname
        actual.mkdir(parents=True, exist_ok=True)
        target = frozen_rerun / dirname
        if not target.exists():
            target.symlink_to(actual.resolve(), target_is_directory=True)
        if target.resolve() != actual.resolve():
            raise ValueError("Unexpected experiment output destination")
    command = [sys.executable, str(frozen_presets / "run_local.py"),
               "--study", *args.study, "--workers", str(args.workers),
               "--hours", str(args.hours), "--reps", str(args.reps)]
    return subprocess.call(command, cwd=work)


if __name__ == "__main__":
    raise SystemExit(main())
