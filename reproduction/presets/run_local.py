"""Run fixed remaining-framework presets against the existing local checkout.

Studies and configurations run sequentially, with at most ``--workers`` workers
inside one simulation command. The overall deadline is shared across all
selected configurations. The read-only ``--list`` option does not create output.
"""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RERUN = ROOT / "revision_20261010" / "rerun"
SIMULATION_SCRIPT = RERUN / "revised_experiments.py"
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
from build_colab_remaining import config_tag, presets  # noqa: E402

N_REPS = 200
M_TEST = 200
N_BOOT = 999
RESOLUTION = 50
ORACLE_N = 10000
PROFILE_MC = 3000
SEED = 20261010
ZIP_RESERVE_SECONDS = 5 * 60
MIN_CONFIG_START_SECONDS = 20 * 60


def print_presets(rows):
    print(f"Project root: {ROOT}")
    print("Available fixed study IDs:")
    for row in rows:
        print(f"\n{row['id']} ({len(row['configs'])} configuration(s)): {row['title']}")
        for cfg in row["configs"]:
            print(f"  {config_tag(cfg)}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        nargs="+",
        metavar="ID",
        help="one or more fixed study IDs from --list, or the single value 'all'",
    )
    parser.add_argument("--list", action="store_true", help="print presets without creating output or running simulations")
    parser.add_argument("--workers", type=int, default=4, help="workers within each sequential configuration (default: 4)")
    parser.add_argument("--hours", type=float, default=8.5, help="overall wall-clock budget shared by all selected configurations (default: 8.5)")
    parser.add_argument("--reps", type=int, default=N_REPS, help="replications per configuration; non-200 values use the execution-check output parent")
    args = parser.parse_args(argv)

    if not args.list:
        if not args.study:
            parser.error("select study IDs or pass '--study all'; use --list to inspect available IDs")
        if "all" in args.study and args.study != ["all"]:
            parser.error("'all' cannot be combined with individual study IDs")
        if args.workers < 1 or args.workers > 4:
            parser.error("--workers must be between 1 and 4")
        if args.hours <= 0 or args.hours > 8.5:
            parser.error("--hours must be in (0, 8.5]")
        if args.reps < 1:
            parser.error("--reps must be at least 1")
    return parser, args


def count_checkpoints(output_dir):
    successes = sum(
        1 for path in output_dir.glob("rep_*.json")
        if len(path.stem) == 10 and path.stem[4:].isdigit()
    )
    failures = len(list(output_dir.glob("failure_*.json")))
    return successes, failures


def run(argv=None):
    parser, args = parse_args(argv)
    rows = presets()
    if args.list:
        print_presets(rows)
        return 0

    by_id = {row["id"]: row for row in rows}
    requested = args.study
    if requested == ["all"]:
        selected = rows
    else:
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            parser.error("unknown study ID(s): " + ", ".join(unknown) + "; use --list for valid IDs")
        wanted = set(requested)
        selected = [row for row in rows if row["id"] in wanted]

    output_parent = (
        RERUN / "remaining_production"
        if args.reps == N_REPS
        else RERUN / "remaining_execution_checks"
    )
    output_parent.mkdir(parents=True, exist_ok=True)
    notebook_started = time.monotonic()
    deadline = notebook_started + args.hours * 3600
    all_statuses = []
    budget_stopped = False
    attempted_configurations = 0
    total_configurations = sum(len(row["configs"]) for row in selected)

    print(
        f"Starting {len(selected)} study/studies with {total_configurations} configuration(s); "
        f"overall budget={args.hours:g} h, workers={args.workers}, reps={args.reps}.",
        flush=True,
    )
    print(f"Outputs: {output_parent}", flush=True)

    for row in selected:
        for cfg in row["configs"]:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds < MIN_CONFIG_START_SECONDS:
                budget_stopped = True
                print(
                    f"GLOBAL BUDGET STOP: skipped {total_configurations - attempted_configurations} "
                    "remaining configuration(s).",
                    flush=True,
                )
                break
            command_hours = min(
                args.hours,
                max(0.0, (remaining_seconds - ZIP_RESERVE_SECONDS) / 3600),
            )
            if command_hours <= 0:
                budget_stopped = True
                print("GLOBAL BUDGET STOP: no command time remains.", flush=True)
                break

            study_dir = output_parent / row["id"]
            output_dir = study_dir / config_tag(cfg)
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(SIMULATION_SCRIPT),
                "--kind", cfg["kind"],
                "--scenario", cfg["scenario"],
                "--n", str(cfg["n"]),
                "--reps", str(args.reps),
                "--m-test", str(M_TEST),
                "--resolution", str(RESOLUTION),
                "--n-boot", str(N_BOOT),
                "--oracle-n", str(ORACLE_N),
                "--profile-mc", str(PROFILE_MC),
                "--degrees", str(cfg["degrees"]),
                "--scale", str(cfg["scale"]),
                "--r", str(cfg["r"]),
                "--points", str(cfg["points"]),
                "--weights", cfg["weights"],
                "--targets", *cfg["targets"],
                "--workers", str(args.workers),
                "--hours", str(command_hours),
                "--shard-id", "0",
                "--num-shards", "1",
                "--seed", str(SEED),
                "--out", str(output_dir),
            ]
            attempted_configurations += 1
            print(
                f"Starting {row['id']} configuration {attempted_configurations}/{total_configurations}; "
                f"command budget={command_hours:.3f} h: {config_tag(cfg)}",
                flush=True,
            )
            # Separate queues may schedule the same preset later. Serialize
            # by configuration so a resume never races an active writer.
            with (output_dir / "execution.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                result = subprocess.run(command, check=False)
            successes, failures = count_checkpoints(output_dir)
            complete = successes == args.reps and result.returncode == 0
            all_statuses.append(complete)
            print(
                f"CHECKPOINTS: {successes}/{args.reps} success files; "
                f"{failures} recorded failure file(s); command exit={result.returncode}.",
                flush=True,
            )
            if complete:
                print("CONFIG STATUS: all requested success checkpoints are present.", flush=True)
            else:
                print(
                    "CONFIG STATUS: INCOMPLETE; the command failed or the checkpoint count differs from the request. "
                    "Inspect failure files and rerun the same study/configuration to resume.",
                    flush=True,
                )
        if budget_stopped:
            break

    elapsed = (time.monotonic() - notebook_started) / 3600
    if budget_stopped:
        print(f"RUN STATUS: BUDGET-STOPPED after {elapsed:.2f} h; later configurations were skipped.")
    elif len(all_statuses) == total_configurations and all(all_statuses):
        print(f"RUN STATUS: COMPLETE; all {total_configurations} configurations have requested checkpoints ({elapsed:.2f} h).")
    else:
        print(f"RUN STATUS: INCOMPLETE after {elapsed:.2f} h; at least one configuration has partial checkpoints.")
    return 0 if not budget_stopped and len(all_statuses) == total_configurations and all(all_statuses) else 2


if __name__ == "__main__":
    raise SystemExit(run())
