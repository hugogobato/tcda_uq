"""Phase 7 CPU orchestrator.

Runs, or prints, the CPU-safe Phase 7 workflow while respecting process-level
parallelism and BLAS thread limits. GEOM-Drugs (7B.3) is intentionally not part
of this runner because it needs the still-missing ``val_data.pickle``.

Examples:
    # Inspect the commands.
    .venv/bin/python experiments/phase7_run_cpu.py --preset smoke

    # Run a quick end-to-end smoke check.
    .venv/bin/python experiments/phase7_run_cpu.py --preset smoke --run --workers 2

    # Run the main CPU evidence sweep on an 8-core machine.
    .venv/bin/python experiments/phase7_run_cpu.py --preset mvp --run --workers 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


@dataclass(frozen=True)
class Step:
    name: str
    commands: tuple[tuple[str, ...], ...]
    parallel: bool = False


def _py(default=".venv/bin/python"):
    return default if Path(default).exists() else sys.executable


def _w(args, cap=None):
    workers = max(1, int(args.workers))
    return min(workers, cap) if cap is not None else workers


def _phase2_steps(py, out, args, ns, reps, n_boot, resolution, shards):
    shard_paths = [out / f"phase2_shard{i}.txt" for i in range(shards)]
    cmds = []
    for i, path in enumerate(shard_paths):
        cmds.append((
            py, "-u", "experiments/phase2_coverage.py",
            "--ns", *map(str, ns),
            "--reps", str(reps),
            "--n-boot", str(n_boot),
            "--resolution", str(resolution),
            "--num-shards", str(shards),
            "--shard-id", str(i),
            "--output", str(path),
        ))
    aggregate = (
        py, "experiments/aggregate_phase2_coverage.py",
        *map(str, shard_paths),
        "--reps", str(reps),
        "--output", str(out / "phase2_aggregated.txt"),
    )
    return [
        Step("phase2 TATE confidence shards", tuple(cmds), parallel=True),
        Step("phase2 aggregate", (aggregate,)),
    ]


def _phase3_steps(py, out, args, ns, reps, n_boot, resolution, shards):
    shard_paths = [out / f"phase3_shard{i}.txt" for i in range(shards)]
    cmds = []
    for i, path in enumerate(shard_paths):
        cmds.append((
            py, "-u", "experiments/phase3_ctate_coverage.py",
            "--ns", *map(str, ns),
            "--reps", str(reps),
            "--n-boot", str(n_boot),
            "--resolution", str(resolution),
            "--num-shards", str(shards),
            "--shard-id", str(i),
            "--output", str(path),
        ))
    aggregate = (
        py, "experiments/aggregate_phase3_coverage.py",
        *map(str, shard_paths),
        "--output", str(out / "phase3_aggregated.txt"),
    )
    return [
        Step("phase3 CTATE confidence shards", tuple(cmds), parallel=True),
        Step("phase3 aggregate", (aggregate,)),
    ]


def _steps(args):
    py = args.python or _py()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    workers = _w(args)
    worker_arg = str(workers)

    if args.preset == "smoke":
        steps = [
            Step("phase2 smoke", ((
                py, "-u", "experiments/phase2_coverage.py",
                "--ns", "50",
                "--reps", "2",
                "--n-boot", "50",
                "--resolution", "20",
                "--output", str(out / "phase2_aggregated.txt"),
            ),)),
            Step("phase3 smoke", ((
                py, "-u", "experiments/phase3_ctate_coverage.py",
                "--ns", "80",
                "--reps", "2",
                "--n-boot", "50",
                "--resolution", "20",
                "--output", str(out / "phase3_aggregated.txt"),
            ),)),
            Step("phase4 ITTE smoke", ((
                py, "-u", "experiments/phase4_itte_coverage.py",
                "--ns", "100",
                "--reps", "2",
                "--resolution", "20",
                "--m-test", "30",
                "--workers", str(_w(args, 2)),
                "--baseline",
                "--output", str(out / "phase4_wellspec.txt"),
            ),)),
            Step("phase5 adaptive smoke", ((
                py, "-u", "experiments/phase5_adaptive_coverage.py",
                "--ns", "120",
                "--reps", "2",
                "--resolution", "20",
                "--m-test", "30",
                "--workers", str(_w(args, 2)),
                "--methods", "marginal", "normalized",
                "--output", str(out / "phase5_hetero.txt"),
            ),)),
            Step("phase6 stabilized smoke", ((
                py, "-u", "experiments/phase6_stabilized_overlap.py",
                "--n", "120",
                "--reps", "2",
                "--resolution", "20",
                "--scales", "1.0", "2.5",
                "--m-test", "30",
                "--workers", str(_w(args, 2)),
                "--output-dir", str(out),
            ),)),
            Step("phase7 ORBIT real-data smoke", ((
                py, "-u", "experiments/phase7_realdata_demo.py",
                "--mode", "orbit",
                "--n", "80",
                "--m-test", "25",
                "--num-pts", "80",
                "--resolution", "25",
                "--output", str(out / "realdata_orbit.txt"),
                "--figure-dir", str(out / "figures"),
            ),)),
        ]
    elif args.preset == "mvp":
        shards = min(max(1, args.shards), workers)
        steps = []
        steps.extend(_phase2_steps(py, out, args, ns=[200, 500], reps=50,
                                   n_boot=500, resolution=40, shards=shards))
        steps.extend(_phase3_steps(py, out, args, ns=[200, 500, 1000], reps=80,
                                   n_boot=700, resolution=40, shards=shards))
        steps.extend([
            Step("phase4 ITTE core", ((
                py, "-u", "experiments/phase4_itte_coverage.py",
                "--ns", "200", "500", "1000",
                "--reps", "80",
                "--resolution", "40",
                "--m-test", "200",
                "--workers", worker_arg,
                "--baseline",
                "--output", str(out / "phase4_wellspec.txt"),
            ),)),
            Step("phase4 weak-overlap", tuple(
                (
                    py, "-u", "experiments/phase4_itte_coverage.py",
                    "--ns", "500",
                    "--reps", "80",
                    "--resolution", "40",
                    "--m-test", "200",
                    "--workers", worker_arg,
                    "--overlap", scale,
                    "--output", str(out / f"phase4_overlap_{scale}.txt"),
                )
                for scale in ("1.0", "2.5", "4.0")
            )),
            Step("phase5 adaptive heteroskedastic", ((
                py, "-u", "experiments/phase5_adaptive_coverage.py",
                "--ns", "500", "1000",
                "--reps", "80",
                "--resolution", "40",
                "--m-test", "250",
                "--hetero", "3.0",
                "--workers", worker_arg,
                "--output", str(out / "phase5_hetero.txt"),
            ),)),
            Step("phase6 stabilized overlap", ((
                py, "-u", "experiments/phase6_stabilized_overlap.py",
                "--n", "500",
                "--reps", "80",
                "--resolution", "40",
                "--scales", "1.0", "2.5", "4.0",
                "--m-test", "200",
                "--workers", worker_arg,
                "--output-dir", str(out),
            ),)),
            Step("phase7 ORBIT demo", ((
                py, "-u", "experiments/phase7_realdata_demo.py",
                "--mode", "orbit",
                "--n", "250",
                "--m-test", "100",
                "--num-pts", "300",
                "--resolution", "60",
                "--output", str(out / "realdata_orbit.txt"),
                "--figure-dir", str(out / "figures"),
            ),)),
        ])
    elif args.preset == "full":
        shards = min(max(1, args.shards), workers)
        steps = []
        steps.extend(_phase2_steps(py, out, args, ns=[50, 100, 200, 500, 1000],
                                   reps=200, n_boot=1500, resolution=50,
                                   shards=shards))
        steps.extend(_phase3_steps(py, out, args, ns=[200, 500, 1000],
                                   reps=200, n_boot=1500, resolution=50,
                                   shards=shards))
        steps.extend([
            Step("phase4 ITTE core", ((
                py, "-u", "experiments/phase4_itte_coverage.py",
                "--ns", "200", "500", "1000",
                "--reps", "200",
                "--resolution", "40",
                "--m-test", "200",
                "--workers", worker_arg,
                "--baseline",
                "--output", str(out / "phase4_wellspec.txt"),
            ),)),
            Step("phase4 overlap and DR sweeps", tuple(
                [(
                    py, "-u", "experiments/phase4_itte_coverage.py",
                    "--ns", "500",
                    "--reps", "200",
                    "--resolution", "40",
                    "--m-test", "200",
                    "--workers", worker_arg,
                    "--overlap", scale,
                    "--output", str(out / f"phase4_overlap_{scale}.txt"),
                ) for scale in ("1.0", "2.5", "4.0")]
                + [(
                    py, "-u", "experiments/phase4_itte_coverage.py",
                    "--ns", "800",
                    "--reps", "200",
                    "--resolution", "40",
                    "--m-test", "200",
                    "--workers", worker_arg,
                    "--misspec", misspec,
                    "--output", str(out / f"phase4_dr_{misspec}.txt"),
                ) for misspec in ("none", "pi", "mu", "both")]
            )),
            Step("phase5 adaptive heteroskedastic", ((
                py, "-u", "experiments/phase5_adaptive_coverage.py",
                "--ns", "500", "1000",
                "--reps", "200",
                "--resolution", "40",
                "--m-test", "300",
                "--hetero", "3.0",
                "--workers", worker_arg,
                "--output", str(out / "phase5_hetero.txt"),
            ),)),
            Step("phase5 homoskedastic control", ((
                py, "-u", "experiments/phase5_adaptive_coverage.py",
                "--ns", "1000",
                "--reps", "200",
                "--resolution", "40",
                "--m-test", "300",
                "--hetero", "0.0",
                "--workers", worker_arg,
                "--output", str(out / "phase5_homo.txt"),
            ),)),
            Step("phase6 stabilized overlap", ((
                py, "-u", "experiments/phase6_stabilized_overlap.py",
                "--n", "500",
                "--reps", "200",
                "--resolution", "40",
                "--scales", "1.0", "1.5", "2.0", "2.5", "3.0", "4.0",
                "--m-test", "200",
                "--workers", worker_arg,
                "--output-dir", str(out),
            ),)),
            Step("phase7 ORBIT demo", ((
                py, "-u", "experiments/phase7_realdata_demo.py",
                "--mode", "orbit",
                "--n", "300",
                "--m-test", "150",
                "--num-pts", "1000",
                "--resolution", "100",
                "--output", str(out / "realdata_orbit.txt"),
                "--figure-dir", str(out / "figures"),
            ),)),
        ])
        if args.include_sars:
            steps.append(Step("phase7 SARS qualitative demo", ((
                py, "-u", "experiments/phase7_realdata_demo.py",
                "--mode", "sars",
                "--n", "400",
                "--m-test", "100",
                "--resolution", "80",
                "--output", str(out / "realdata_sars.txt"),
                "--figure-dir", str(out / "figures"),
            ),)))
    else:
        raise ValueError(args.preset)

    # Level-B topological sweep (7A.2/7A.3/7A.4) and the 7A.5 regime analysis both
    # manage their own process pool, so they run as single (internally parallel)
    # steps. They write to siblings of `out` so the master table collects them.
    steps.append(Step("phase7 level-b topological", ((
        py, "-u", "experiments/phase7_level_b.py",
        "--preset", args.preset,
        "--workers", worker_arg,
        "--output-dir", str(out.parent / "phase7_level_b"),
    ),)))
    steps.append(Step("phase7 regime analysis", ((
        py, "-u", "experiments/phase7_regime_analysis.py",
        "--preset", args.preset,
        "--workers", worker_arg,
        "--output-dir", str(out.parent / "phase7_regime"),
    ),)))

    steps.append(Step("phase7 master table", ((
        py, "experiments/phase7_build_master_table.py",
        "--root", str(out.parent),
        "--output-dir", str(out),
    ),)))
    return steps


def _env():
    env = os.environ.copy()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    return env


def _print_step(step: Step):
    mode = "parallel" if step.parallel else "serial"
    print(f"\n# {step.name} ({mode})")
    for cmd in step.commands:
        print(shlex.join(cmd))


def _run_step(step: Step, env):
    print(f"\n# running: {step.name}", flush=True)
    t0 = time.time()
    if step.parallel:
        procs = [subprocess.Popen(cmd, env=env) for cmd in step.commands]
        codes = [p.wait() for p in procs]
        bad = [c for c in codes if c != 0]
        if bad:
            raise SystemExit(f"{step.name} failed with codes {codes}")
    else:
        for cmd in step.commands:
            subprocess.run(cmd, check=True, env=env)
    print(f"# done: {step.name} ({time.time() - t0:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["smoke", "mvp", "full"], default="smoke")
    ap.add_argument("--run", action="store_true", help="execute instead of printing commands")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shards", type=int, default=4,
                    help="Phase 2/3 shard count, capped by --workers")
    ap.add_argument("--python", default=None)
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/out/phase7"))
    ap.add_argument("--include-sars", action="store_true",
                    help="include the SARS-CoV-2 qualitative demo in the full preset")
    args = ap.parse_args()

    steps = _steps(args)
    if not args.run:
        print("# dry run; add --run to execute")
        print("# BLAS threads are set to 1 during execution; process workers do the parallelism.")
        for step in steps:
            _print_step(step)
        return

    env = _env()
    for step in steps:
        _run_step(step, env)


if __name__ == "__main__":
    main()
