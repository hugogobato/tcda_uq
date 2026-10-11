"""Recover native zero-mean tests, verifying the original mean fits exactly.

This repeats only the mean-fit stage of the frozen PACCE simulations. Sidecars
leave every original checkpoint unchanged. They are power/type-I diagnostics
for the same controlled simulation, not tests of the observed clinical cohort.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/tcda_uq_mpl")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FROZEN = None


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frozen(source_root):
    global FROZEN
    path = Path(source_root) / "revision_20261010/rerun/revised_experiments.py"
    sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location("frozen_pacce_mean_tests", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    FROZEN = module


def recover(task):
    run_dir, rep = Path(task[0]), task[1]
    cfg = json.loads((run_dir / "manifest.json").read_text())["configuration"]
    original_path = run_dir / f"rep_{rep:06d}.json"
    original = json.loads(original_path.read_text())
    out = run_dir / "mean_zero_tests" / original_path.name
    if out.is_file():
        previous = json.loads(out.read_text())
        assert previous["original_checkpoint_sha256"] == digest(original_path)
        return
    seed = cfg["seed"] + rep * 100003
    started = time.monotonic()
    X, A, po, pi, grid = FROZEN.generate(cfg, cfg["n"], seed)
    observed = po[np.arange(len(A)), A]
    known = FROZEN.KnownPropensity(cfg["kind"], cfg["scale"], cfg["scenario"] == "dropout_arm")
    fit = FROZEN.stable_cross_fit((observed, A, X), grid,
                                  propensity_estimator=known, random_state=seed + 1)
    bands = FROZEN.multiplier_bootstrap_bands(
        fit, alpha=cfg["alpha_confidence"], n_boot=cfg["n_boot"], rng=seed + 2,
        multiplier="gaussian", standardize=False)
    truth = np.load(run_dir / "truth.npz")["mean"]
    rows = []
    for degree, band in enumerate(bands):
        old = next(r for r in original["rows"] if r["metric"] == "TATE" and r["degree"] == degree)
        calculated = {"coverage": float(bool(band.covers(truth[degree]))),
                      "width": band.mean_width(),
                      "sup_bias": float(np.max(np.abs(band.center - truth[degree]))),
                      "mean_error": float(np.mean(band.center - truth[degree]))}
        for key, value in calculated.items():
            if not np.isclose(value, old[key], rtol=1e-12, atol=1e-14):
                raise RuntimeError(f"Mean-fit replay mismatch: {run_dir.name}, rep={rep}, {key}")
        rows.append({"degree": degree,
                     "reject_zero": bool(np.any((band.lower > 0) | (band.upper < 0))),
                     "sup_estimate": float(np.max(np.abs(band.center))),
                     "critical_half_width": float(np.max(band.upper - band.center)),
                     "agreement_with_original": calculated})
    FROZEN.atomic_json(out, {"rep": rep, "seed": seed, "n_observed": len(A),
                            "original_checkpoint_sha256": digest(original_path),
                            "frozen_adapter_sha256": digest(FROZEN.__file__),
                            "sidecar_script_sha256": digest(__file__),
                            "elapsed_seconds": time.monotonic() - started, "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path,
                        default=ROOT / "revision_20261010/rerun/remaining_frozen_workspace")
    parser.add_argument("--runs-root", type=Path,
                        default=ROOT / "revision_20261010/rerun/remaining_production")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between one and four")
    load_frozen(args.source_root.resolve())
    tasks, settings = [], []
    for directory in sorted(args.runs_root.glob("*_pacce_*/*")):
        if not (directory / "manifest.json").is_file():
            continue
        cfg = json.loads((directory / "manifest.json").read_text())["configuration"]
        assert cfg["kind"] == "pacce" and cfg["reps"] == 200
        records = sorted(directory.glob("rep_*.json"))
        if args.require_complete and len(records) != 200:
            raise RuntimeError(f"PACCE setting incomplete: {directory.name}")
        tasks.extend((str(directory), int(p.stem[4:])) for p in records)
        settings.append(str(directory))
    if args.require_complete and len(settings) != 7:
        raise RuntimeError("All seven PACCE settings are required")
    with ProcessPoolExecutor(max_workers=args.workers, initializer=load_frozen,
                             initargs=(str(args.source_root.resolve()),)) as pool:
        for count, _ in enumerate(pool.map(recover, tasks), 1):
            if count % 100 == 0:
                print(f"Verified {count}/{len(tasks)} native mean-zero tests", flush=True)
    summary = {"settings": len(settings), "verified_sidecars": len(tasks),
               "original_checkpoints_unchanged": True,
               "source_root": str(args.source_root.resolve()),
               "sidecar_script_sha256": digest(__file__),
               "frozen_adapter_sha256": digest(FROZEN.__file__),
               "scope": "Native Gaussian absolute-supremum zero-mean test replay; no new fitted samples"}
    (args.runs_root / "pacce_mean_test_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
