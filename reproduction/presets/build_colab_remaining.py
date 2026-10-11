"""Build the remaining self-contained, fixed-preset Colab rerun notebooks.

The embedded source, resume, and output-download cells are copied verbatim from
the frozen 01_simulation_shard.ipynb. Only new artifacts in this directory are
written by this builder.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = ROOT / "revision_20261010" / "rerun" / "01_simulation_shard.ipynb"
PUBLIC_LAYOUT = OUT.name == "presets" and OUT.parent.name == "reproduction"
if PUBLIC_LAYOUT:
    OUT = OUT.parent / "colab"
    SOURCE_NOTEBOOK = OUT / "01_functional_step_bias.ipynb"

THREE_TARGETS = ["full", "overlap", "matching"]
SIX_TARGETS = ["full", "overlap", "matching", "tilted", "clip", "shrink"]
WEIGHT_MODES = ["known", "learned", "misspecified"]


def config(kind, scenario, n, degrees, points, scale, r, weights, targets):
    return {
        "kind": kind,
        "scenario": scenario,
        "n": n,
        "degrees": degrees,
        "points": points,
        "scale": scale,
        "r": r,
        "weights": weights,
        "targets": list(targets),
    }


def presets():
    rows = [
        {
            "id": "01_functional_step_bias",
            "title": "Functional step-bias diagnostic",
            "description": "Functional generator with the treatment-arm step perturbation, n=500, degree 2, known propensity, and full/overlap/matching targets.",
            "configs": [config("functional", "step", 500, 2, 40, 1, 3, "known", THREE_TARGETS)],
        },
        {
            "id": "02_functional_degree3_base",
            "title": "Functional degree-3 base diagnostic",
            "description": "Functional base generator, n=500, degree 3, known propensity, and full/overlap/matching targets.",
            "configs": [config("functional", "base", 500, 3, 40, 1, 3, "known", THREE_TARGETS)],
        },
        {
            "id": "03_functional_weight_target_sensitivities",
            "title": "Functional weight and target sensitivities",
            "description": "Six configurations: propensity scales 1 and 4 crossed with known, learned, and misspecified prediction weights. Each uses n=500, degree 2, and all six target modes.",
            "configs": [
                config("functional", "base", 500, 2, 40, scale, 3, weights, SIX_TARGETS)
                for scale in (1, 4)
                for weights in WEIGHT_MODES
            ],
        },
        {
            "id": "04_topological_scale4_weight_target_sensitivities",
            "title": "Topological weak-overlap weight and target sensitivities",
            "description": "Three configurations crossing known, learned, and misspecified prediction weights at propensity scale 4. Each uses topological r=3, n=500, 40 points per loop, degree 2, and all six target modes.",
            "configs": [
                config("topological", "base", 500, 2, 40, 4, 3, weights, SIX_TARGETS)
                for weights in WEIGHT_MODES
            ],
        },
    ]

    for r in (1, 2, 4, 5):
        rows.append(
            {
                "id": f"{len(rows) + 1:02d}_topological_r{r}",
                "title": f"Topological r={r} diagnostic",
                "description": f"Topological generator at r={r}, n=500, 40 points per loop, degree 2, propensity scale 1, known weights, and full/overlap/matching targets.",
                "configs": [config("topological", "base", 500, 2, 40, 1, r, "known", THREE_TARGETS)],
            }
        )

    rows.append(
        {
            "id": "09_orbit_1000_points",
            "title": "ORBIT full-resolution diagnostic",
            "description": "ORBIT generator with n=300, 1,000 points, degree 2, r=3, propensity scale 1, known weights, and full/overlap/matching targets.",
            "configs": [config("orbit", "base", 300, 2, 1000, 1, 3, "known", THREE_TARGETS)],
        }
    )

    pacce_scenarios = [
        "null",
        "sld_only",
        "topo_favorable",
        "topo_null",
        "nuisance_misp",
        "irregular_visits",
        "dropout_arm",
    ]
    for scenario in pacce_scenarios:
        rows.append(
            {
                "id": f"{len(rows) + 1:02d}_pacce_{scenario}",
                "title": f"PACCE simulation: {scenario}",
                "description": f"Controlled PACCE simulation scenario {scenario}, n=842, degree 1, 40 points, scale 1, known weights, and full/overlap/matching targets. This uses the embedded simulation generator and requires no clinical upload.",
                "configs": [config("pacce", scenario, 842, 1, 40, 1, 3, "known", THREE_TARGETS)],
            }
        )

    if len(rows) != 16:
        raise AssertionError(f"expected 16 presets, found {len(rows)}")
    output_paths = []
    for row in rows:
        for cfg in row["configs"]:
            output_paths.append((row["id"], config_tag(cfg)))
    if len(output_paths) != len(set(output_paths)):
        raise AssertionError("configuration output paths are not unique")
    return rows


def config_tag(cfg):
    target_tag = "-".join(cfg["targets"])
    return (
        f"{cfg['kind']}_{cfg['scenario']}_n{cfg['n']}_d{cfg['degrees']}"
        f"_p{cfg['points']}_r{cfg['r']:g}_scale{cfg['scale']:g}"
        f"_weights-{cfg['weights']}_targets-{target_tag}"
    )


def make_cell(source, cell_type="code"):
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def notebook_description(row):
    extra = ""
    if row["id"].startswith("03_") or row["id"].startswith("04_"):
        extra = (
            "\n\nRun All executes configurations sequentially under one 8.5-hour deadline. It prints checkpoint and "
            "failure counts after each configuration, skips later configurations when fewer than 20 minutes remain, "
            "and leaves five minutes for output ZIP packaging. If interrupted, rerun with the optional resume cell "
            "enabled and upload the prior output ZIP."
        )
    if row["id"].startswith("09_"):
        extra = (
            "\n\nThe 1,000-point ORBIT setting can be expensive. Measure runtime first; the 8.5-hour preset budget "
            "may need further sharding after timing, with distinct shard IDs across independent accounts."
        )
    return (
        f"# {row['title']}\n\n"
        f"{row['description']}\n\n"
        "**STATUS: NOT RUN. These are production requests, not completed production evidence.**\n\n"
        "These are finite-grid diagnostics and do not guarantee continuous-domain coverage. Assess the saved Monte Carlo "
        "standard errors for the numerical oracle and profile calculations before interpreting coverage. A smoke run only "
        "checks execution and is not coverage evidence. The notebook has fixed presets and runs without selector edits, "
        "a repository clone, or raw clinical uploads. Results are saved below `outputs/"
        f"{row['id']}/`, with a unique directory for every scientific configuration.\n\n"
        "The default is one shard (`SHARD_ID=0`, `NUM_SHARDS=1`), two workers, and an 8.5-hour global budget "
        "across configuration commands. Each command receives only the remaining time, minus a five-minute ZIP "
        "reserve. No new configuration starts with fewer than 20 minutes remaining. Checkpoint and failure counts "
        "are printed after each command, with partial runs distinguished from 200 completed success checkpoints. "
        "The next notebook cell still packages outputs when the global budget stops later configurations. Only "
        "increase sharding after measuring runtime, and keep all scientific settings and seeds identical across shards."
        f"{extra}\n"
    )


def run_cell(row):
    configs_json = json.dumps(row["configs"], separators=(",", ":"))
    return f'''# Fixed Run-all preset. No configuration selector or upload is required.
import json, subprocess, sys, time

PRESET_ID = {row['id']!r}
CONFIGS = json.loads({configs_json!r})
SHARD_ID, NUM_SHARDS = 0, 1
WORKERS = 2
NOTEBOOK_HOURS = 8.5
ZIP_RESERVE_SECONDS = 5 * 60
MIN_CONFIG_START_SECONDS = 20 * 60
N_REPS, M_TEST, N_BOOT = 200, 200, 999
RESOLUTION, ORACLE_N, PROFILE_MC = 50, 10000, 3000
notebook_started = time.monotonic()
notebook_deadline = notebook_started + NOTEBOOK_HOURS * 3600
PRESET_OUTPUT = OUTPUT / PRESET_ID
PRESET_OUTPUT.mkdir(parents=True, exist_ok=True)
config_statuses = []
budget_stopped = False

for config_index, cfg in enumerate(CONFIGS):
    remaining_seconds = notebook_deadline - time.monotonic()
    if remaining_seconds < MIN_CONFIG_START_SECONDS:
        budget_stopped = True
        print(
            f"NOTEBOOK BUDGET STOP: skipped {{len(CONFIGS) - config_index}} configuration(s); "
            "the output ZIP cell will run next.", flush=True
        )
        break
    command_hours = min(
        NOTEBOOK_HOURS,
        max(0.0, (remaining_seconds - ZIP_RESERVE_SECONDS) / 3600),
    )
    if command_hours <= 0:
        budget_stopped = True
        print("NOTEBOOK BUDGET STOP: no command time remains; the output ZIP cell will run next.", flush=True)
        break
    output_dir = PRESET_OUTPUT / (
        f"{{cfg['kind']}}_{{cfg['scenario']}}_n{{cfg['n']}}_d{{cfg['degrees']}}"
        f"_p{{cfg['points']}}_r{{cfg['r']:g}}_scale{{cfg['scale']:g}}"
        f"_weights-{{cfg['weights']}}_targets-{{'-'.join(cfg['targets'])}}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, 'revision_20261010/rerun/revised_experiments.py',
        '--kind', cfg['kind'], '--scenario', cfg['scenario'],
        '--n', str(cfg['n']), '--reps', str(N_REPS), '--m-test', str(M_TEST),
        '--resolution', str(RESOLUTION), '--n-boot', str(N_BOOT),
        '--oracle-n', str(ORACLE_N), '--profile-mc', str(PROFILE_MC),
        '--degrees', str(cfg['degrees']), '--scale', str(cfg['scale']),
        '--r', str(cfg['r']), '--points', str(cfg['points']),
        '--weights', cfg['weights'], '--targets', *cfg['targets'],
        '--workers', str(WORKERS), '--hours', str(command_hours),
        '--shard-id', str(SHARD_ID), '--num-shards', str(NUM_SHARDS),
        '--seed', '20261010', '--out', str(output_dir),
    ]
    print(
        f"Starting configuration {{config_index + 1}}/{{len(CONFIGS)}} with "
        f"{{command_hours:.3f}} command hours:", cfg, flush=True
    )
    result = subprocess.run(command, check=False)
    completed = sum(
        1 for path in output_dir.glob('rep_*.json')
        if len(path.stem) == 10 and path.stem[4:].isdigit()
    )
    failures = len(list(output_dir.glob('failure_*.json')))
    complete = completed == N_REPS and result.returncode == 0
    config_statuses.append(complete)
    print(
        f"CONFIG CHECKPOINTS: {{completed}}/{{N_REPS}} success files; "
        f"{{failures}} recorded failure file(s); command exit={{result.returncode}}.",
        flush=True,
    )
    if complete:
        print("CONFIG STATUS: all 200 requested success checkpoints are present.", flush=True)
    else:
        print(
            "CONFIG STATUS: INCOMPLETE; the command failed or the checkpoint count differs from 200. "
            "Inspect failures and rerun the same configuration to resume.", flush=True
        )

elapsed_hours = (time.monotonic() - notebook_started) / 3600
if budget_stopped:
    print(
        f"PRESET STATUS: BUDGET-STOPPED after {{elapsed_hours:.2f}} hours; "
        "some configurations were skipped. The next cell will package available outputs.", flush=True
    )
elif config_statuses and all(config_statuses):
    print(
        f"PRESET STATUS: COMPLETE; every configuration has all 200 success checkpoints "
        f"({{elapsed_hours:.2f}} hours).", flush=True
    )
else:
    print(
        f"PRESET STATUS: INCOMPLETE after {{elapsed_hours:.2f}} hours; one or more "
        "configurations have fewer than 200 success checkpoints. The next cell will package outputs.",
        flush=True,
    )
'''


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE_NOTEBOOK.read_text())
    # Preserve the checksummed frozen source setup, secure resume, and one-ZIP
    # download cells byte-for-byte at the source-string level.
    setup, resume, download = (copy.deepcopy(source["cells"][i]) for i in (1, 2, 4))
    rows = presets()
    notebook_names = []
    for row in rows:
        name = f"{row['id']}.ipynb"
        nb = copy.deepcopy(source)
        nb["cells"] = [
            make_cell(notebook_description(row), "markdown"),
            setup,
            resume,
            make_cell(run_cell(row)),
            download,
        ]
        for i, cell in enumerate(nb["cells"]):
            cell["id"] = f"cell-{i:03d}"
        (OUT / name).write_text(json.dumps(nb, indent=1) + "\n")
        notebook_names.append(name)

    readme = (PUBLIC_README_TEXT if PUBLIC_LAYOUT else README_TEXT).format(rows="\n".join(readme_rows(rows)))
    readme_path = OUT / "README.md"
    readme_path.write_text(readme)

    archive = OUT / "colab_remaining_presets.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(readme_path, "README.md")
        for name in notebook_names:
            zf.write(OUT / name, name)
    print(f"Wrote {len(notebook_names)} notebooks and {archive}")


def readme_rows(rows):
    descriptions = {
        row["id"]: row["description"] for row in rows
    }
    return [f"| `{row['id']}` | {descriptions[row['id']]} |" for row in rows]


README_TEXT = """# Remaining revised-framework Colab diagnostics

**Status: all notebooks below are NOT RUN production requests.** Their presence does not mean these studies or results are complete. Each notebook is self-contained, ready for Colab Run All, embeds the exact frozen source setup from `01_simulation_shard.ipynb`, and requires no repository clone or raw clinical upload. The PACCE entries use controlled simulation scenarios.

The notebooks request 200 replications, 200 test draws, 999 bootstrap draws, grid resolution 50, oracle sample size 10,000, profile Monte Carlo size 3,000, two workers, and an 8.5-hour per-command budget. They default to shard 0 of 1. Each scientific configuration has a unique output directory containing its study, scenario, sample size, degree, point count, r, scale, weights, and targets, so resume cannot silently mix settings.

These are finite-grid diagnostics and do not guarantee continuous-domain coverage. Assess the saved Monte Carlo standard errors for oracle and profile calculations before drawing conclusions. Small smoke runs establish execution only, not coverage. The ORBIT setting with 1,000 points may need further sharding after timing. Each notebook has one 8.5-hour budget across its configurations, reserves five minutes for ZIP packaging, stops starting configurations with fewer than 20 minutes remaining, and prints checkpoint and failure counts. The final notebook cell packages available outputs after a budget stop. If a run is interrupted, the optional resume cell accepts the output ZIP from the preceding run.

| Notebook | Fixed request |
| --- | --- |
{rows}

## Local runner

`run_local.py` runs selected studies sequentially against the existing local source and environment. It uses up to four worker processes per configuration and one 8.5-hour overall budget by default. Choose IDs from the table, or pass `all` explicitly. `--list` only prints available studies. Production runs (`--reps 200`) save below `revision_20261010/rerun/remaining_production/`; any other replication count saves below the separate `revision_20261010/rerun/remaining_execution_checks/` parent so a smoke run cannot be mistaken for production evidence.

```sh
rtk .venv/bin/python revision_20261010/rerun/colab_remaining/run_local.py --list
rtk .venv/bin/python revision_20261010/rerun/colab_remaining/run_local.py --study 01_functional_step_bias
rtk .venv/bin/python revision_20261010/rerun/colab_remaining/run_local.py --study 01_functional_step_bias --reps 1
rtk .venv/bin/python revision_20261010/rerun/colab_remaining/run_local.py --study all
```

Timing measured locally for the project configuration suggests ORBIT at 1,000 points takes about 73 seconds per replication; 200 replications with four workers take around 1–2 hours, plus about 9 minutes for the 10,000-unit oracle. With two workers this is around 2–3 hours. PACCE configurations take about 3–4 minutes with four workers; functional configurations take under a minute; topological three-target configurations take about 9–14 minutes and six-target configurations about 14–19 minutes with four workers. The full 23-configuration local program is estimated at roughly 3–4 hours. Colab timing differs and should be measured there.

`build_colab_remaining.py` reproduces all notebook files, this README, and `colab_remaining_presets.zip`. The archive contains exactly these 16 notebooks and this README.
"""


PUBLIC_README_TEXT = """# Colab reproduction presets

These 16 self-contained notebooks request 23 scientific configurations, with
200 replications per configuration. Upload any notebook to Colab and use Run
All. No repository clone, selector edit or clinical-data upload is required.
Each notebook includes its checksummed source, dependency installation, resume
option and a safe single-ZIP output download. The initial NOT RUN status
describes a fresh notebook execution.

All 23 configurations were completed locally. Saved production records are in
`../results/remaining_production/`, with original independent validation in
`../validation/`. These fitted samples are part of the 11,200-record production
program. Pilot runs and deterministic mean-stage replays are excluded.

| Notebook | Fixed request |
| --- | --- |
{rows}

From a repository checkout, use `python reproduction/run_revision.py --list`
and `python reproduction/run_revision.py --study all --workers 4` for fresh
local runs. See `docs/revision_reproduction.md` for source scope, primary runs,
mean-stage test replay and validation. Generated outputs remain visible to Git.

The shared notebook budget is 8.5 hours with two workers, leaving time for ZIP
packaging. Incomplete commands and checkpoint counts are reported explicitly.
The ORBIT preset takes about 2–3 hours locally with two workers; Colab timing
can differ. Use disjoint shard IDs only after measuring runtime, and preserve
all scientific parameters and seeds across shards.

These are finite-grid diagnostics. Estimated weights, numerical-reference
error, working-model bias and real clinical selection retain the qualifications
in the reproduction guide and statistical-scope documentation.

Run `python reproduction/presets/build_colab_remaining.py` to regenerate the
notebooks and this directory's single ZIP from the embedded source template.
"""


if __name__ == "__main__":
    build()
