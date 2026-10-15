"""Phase 7 -- build the unified estimand x uncertainty comparison table.

This is a reporting layer over the Phase 2-6 experiment outputs. It deliberately
accepts partial results: if Phase 2 production tables have not been run yet, the
master table is still built from Phases 3-6 and marks the missing component in
the verdict note.

Outputs:
  * ``experiments/out/phase7/master_table.csv``
  * ``experiments/out/phase7/master_table.md``
  * ``experiments/out/phase7/verdict.md``

Run:
    .venv/bin/python experiments/phase7_build_master_table.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Phase7Row:
    source: str
    level: str
    estimand: str
    uncertainty: str
    method: str
    n: str
    scenario: str
    dim: str
    coverage: str
    mean_width: str
    inf_pct: str
    extra: str


def _fmt(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.4g}"


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _txt(path: Path):
    try:
        return path.read_text()
    except FileNotFoundError:
        return ""


def _first_header_value(text: str, name: str):
    m = re.search(rf"\b{name}=([^\s|]+)", text)
    return m.group(1) if m else None


def parse_phase2(path: Path):
    rows = []
    for line in _txt(path).splitlines():
        tok = line.split()
        if len(tok) != 6 or not tok[0].isdigit() or not tok[2].isdigit():
            continue
        n, method, dim, cov, iwe, width = tok
        rows.append(
            Phase7Row(
                source=str(path),
                level="7A Level A",
                estimand="TATE",
                uncertainty="confidence",
                method=method,
                n=n,
                scenario="functional DGP",
                dim=dim,
                coverage=cov,
                mean_width=width,
                inf_pct="0.0",
                extra=f"iw_err={iwe}",
            )
        )
    return rows


def parse_phase3(path: Path):
    rows = []
    for line in _txt(path).splitlines():
        tok = line.split()
        if len(tok) < 8 or not tok[0].isdigit() or not tok[3].isdigit():
            continue
        n, x_name, method, dim = tok[:4]
        recovery, cov, iwe, width = tok[4:8]
        rows.append(
            Phase7Row(
                source=str(path),
                level="7A Level A",
                estimand="CTATE mean",
                uncertainty="confidence",
                method=method,
                n=n,
                scenario=f"functional DGP; x={x_name}",
                dim=dim,
                coverage=cov,
                mean_width=width,
                inf_pct="0.0",
                extra=f"recovery={recovery}; iw_err={iwe}",
            )
        )
    return rows


def _phase4_scenario(path: Path, text: str):
    overlap = _first_header_value(text, "overlap")
    misspec = _first_header_value(text, "misspec")
    if overlap or misspec:
        bits = []
        if overlap:
            bits.append(f"overlap={overlap}")
        if misspec:
            bits.append(f"misspec={misspec}")
        return "; ".join(bits)
    stem = path.stem
    if stem.startswith("overlap_"):
        return f"overlap={stem.split('_', 1)[1]}"
    if stem.startswith("dr_"):
        return f"misspec={stem.split('_', 1)[1]}"
    return stem


def parse_phase4(path: Path):
    text = _txt(path)
    scenario = _phase4_scenario(path, text)
    rows = []
    for line in text.splitlines():
        tok = line.split()
        if len(tok) != 7 or not tok[0].isdigit() or not tok[2].isdigit():
            continue
        n, method, dim, cov_itte, cov_mean, width, inf = tok
        uncertainty = "prediction" if method == "conformal" else "confidence baseline"
        extra = "" if cov_mean == "-" else f"cov_mean={cov_mean}"
        rows.append(
            Phase7Row(
                source=str(path),
                level="7A Level A",
                estimand="ITTE",
                uncertainty=uncertainty,
                method=method,
                n=n,
                scenario=f"functional DGP; {scenario}",
                dim=dim,
                coverage=cov_itte,
                mean_width=width,
                inf_pct=inf,
                extra=extra,
            )
        )
    return rows


def parse_phase5(path: Path):
    text = _txt(path)
    hetero = _first_header_value(text, "hetero") or "?"
    overlap = _first_header_value(text, "overlap") or "?"
    rows = []
    for line in text.splitlines():
        tok = line.split()
        if len(tok) != 9 or not tok[0].isdigit() or not tok[2].isdigit():
            continue
        n, method, dim, cov, width, inf, cov_gap, w_corr, w_cv = tok
        rows.append(
            Phase7Row(
                source=str(path),
                level="7A Level A",
                estimand="CTATE-level draw",
                uncertainty="adaptive prediction",
                method=method,
                n=n,
                scenario=f"functional DGP; hetero={hetero}; overlap={overlap}",
                dim=dim,
                coverage=cov,
                mean_width=width,
                inf_pct=inf,
                extra=f"cov_gap={cov_gap}; w_corr={w_corr}; w_cv={w_cv}",
            )
        )
    return rows


def parse_phase6_stabilized(path: Path):
    text = _txt(path)
    n = _first_header_value(text, "n") or ""
    misspec = _first_header_value(text, "misspec") or "none"
    rows = []
    for line in text.splitlines():
        tok = line.split()
        if len(tok) != 9 or _float(tok[0]) is None or not tok[2].isdigit():
            continue
        scale, method, dim, cov, cov_fin, inf, width, ess, max_wt = tok
        rows.append(
            Phase7Row(
                source=str(path),
                level="7A Level A",
                estimand="ITTE",
                uncertainty="stabilized prediction",
                method=method,
                n=n,
                scenario=f"functional DGP; overlap={scale}; misspec={misspec}",
                dim=dim,
                coverage=cov,
                mean_width=width,
                inf_pct=inf,
                extra=f"cov_finite={cov_fin}; ESS/ncal={ess}; max_wt={max_wt}",
            )
        )
    return rows


def parse_phase7_real(path: Path):
    rows = []
    for line in _txt(path).splitlines():
        tok = line.split()
        if len(tok) < 8 or tok[0] != "ROW":
            continue
        data = dict(item.split("=", 1) for item in tok[1:] if "=" in item)
        estimand = data.get("estimand", "").replace("_", " ")
        uncertainty = data.get("uncertainty", "").replace("_", " ")
        rows.append(
            Phase7Row(
                source=str(path),
                level=data.get("level", "7B"),
                estimand=estimand,
                uncertainty=uncertainty,
                method=data.get("method", "").replace("_", " "),
                n=data.get("n", ""),
                scenario=data.get("scenario", ""),
                dim=data.get("dim", ""),
                coverage=data.get("coverage", ""),
                mean_width=data.get("mean_width", ""),
                inf_pct=data.get("inf_pct", ""),
                extra=data.get("extra", ""),
            )
        )
    return rows


def parse_phase7_level_b(path: Path):
    """Level-B topological sweep (7A.2/7A.4): ``ROW key=val`` lines from level_b.txt."""
    rows = []
    for line in _txt(path).splitlines():
        tok = line.split()
        if len(tok) < 8 or tok[0] != "ROW":
            continue
        data = dict(item.split("=", 1) for item in tok[1:] if "=" in item)
        rows.append(
            Phase7Row(
                source=str(path),
                level="7A Level B",
                estimand=data.get("estimand", "").replace("_", " "),
                uncertainty=data.get("uncertainty", "").replace("_", " "),
                method=data.get("method", "").replace("_", " "),
                n=data.get("n", ""),
                scenario="topological DGP; " + data.get("scenario", ""),
                dim=data.get("dim", ""),
                coverage=data.get("coverage", ""),
                mean_width=data.get("mean_width", ""),
                inf_pct=data.get("inf_pct", ""),
                extra=data.get("extra", ""),
            )
        )
    return rows


def parse_regime(path: Path):
    """Regime analysis (7A.5): the ``regime_analysis.csv`` where-each-tool-wins table."""
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for rec in csv.DictReader(f):
            level = "7A Level B" if rec.get("level") == "B" else "7A Level A"
            rows.append(
                Phase7Row(
                    source=str(path),
                    level=level,
                    estimand=rec.get("estimand", ""),
                    uncertainty=rec.get("uncertainty", ""),
                    method=rec.get("method", ""),
                    n=rec.get("n", ""),
                    scenario="regime: " + rec.get("scenario", ""),
                    dim=rec.get("dim", ""),
                    coverage=rec.get("coverage", ""),
                    mean_width=rec.get("mean_width", ""),
                    inf_pct=rec.get("inf_pct", ""),
                    extra=f"reps={rec.get('reps', '')}",
                )
            )
    return rows


def collect_rows(root: Path):
    rows = []
    phase2_files = sorted((root / "phase2").glob("*.txt"))
    phase7_p2 = root / "phase7" / "phase2_aggregated.txt"
    if phase7_p2.exists():
        phase2_files.append(phase7_p2)
    else:
        phase2_files += sorted((root / "phase7").glob("phase2*.txt"))
    for path in phase2_files:
        rows.extend(parse_phase2(path))

    phase3 = root / "phase3" / "aggregated.txt"
    if phase3.exists():
        rows.extend(parse_phase3(phase3))
    phase7_p3 = root / "phase7" / "phase3_aggregated.txt"
    if phase7_p3.exists():
        rows.extend(parse_phase3(phase7_p3))
    else:
        rows.extend(parse_phase3(root / "phase7" / "phase3_smoke.txt"))

    for path in sorted((root / "phase4").glob("*.txt")):
        rows.extend(parse_phase4(path))
    for path in sorted((root / "phase7").glob("phase4*.txt")):
        rows.extend(parse_phase4(path))

    for path in sorted((root / "phase5").glob("*.txt")):
        rows.extend(parse_phase5(path))
    for path in sorted((root / "phase7").glob("phase5*.txt")):
        rows.extend(parse_phase5(path))

    phase6 = root / "phase6" / "stabilized_overlap.txt"
    if phase6.exists():
        rows.extend(parse_phase6_stabilized(phase6))
    for p6 in (root / "phase7" / "stabilized_overlap.txt",
               root / "phase7" / "phase6_stabilized_smoke.txt"):
        rows.extend(parse_phase6_stabilized(p6))

    for path in sorted((root / "phase7").glob("realdata_*.txt")):
        rows.extend(parse_phase7_real(path))

    # 7A.2/7A.4 Level-B topological sweep (own dir or bundled under phase7/)
    for folder in ("phase7_level_b", "phase7"):
        for path in sorted((root / folder).glob("level_b*.txt")):
            rows.extend(parse_phase7_level_b(path))
    # 7A.5 regime analysis
    for folder in ("phase7_regime", "phase7"):
        rows.extend(parse_regime(root / folder / "regime_analysis.csv"))
    return rows


def _sort_key(row: Phase7Row):
    order = {
        "TATE": 0,
        "CTATE mean": 1,
        "CTATE-level draw": 2,
        "ITTE": 3,
    }
    return (
        order.get(row.estimand, 99),
        row.level,
        _float(row.n) if _float(row.n) is not None else 1e9,
        row.scenario,
        row.method,
        _float(row.dim) if _float(row.dim) is not None else 1e9,
    )


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in sorted(rows, key=_sort_key):
            writer.writerow(asdict(row))


def write_markdown(rows, path: Path):
    cols = [
        "level",
        "estimand",
        "uncertainty",
        "method",
        "n",
        "scenario",
        "dim",
        "coverage",
        "mean_width",
        "inf_pct",
        "extra",
    ]
    lines = ["# Phase 7 Master Table", ""]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")
    for row in sorted(rows, key=_sort_key):
        data = asdict(row)
        lines.append("| " + " | ".join(data[c].replace("|", "/") for c in cols) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _mean(rows, **flt):
    vals = []
    for row in rows:
        if all(getattr(row, k) == v for k, v in flt.items()):
            x = _float(row.coverage)
            if x is not None:
                vals.append(x)
    return sum(vals) / len(vals) if vals else None


def write_verdict(rows, path: Path):
    have = {row.estimand for row in rows}
    missing = [name for name in ["TATE", "CTATE mean", "CTATE-level draw", "ITTE"] if name not in have]

    phase4_conf = _mean(rows, estimand="ITTE", uncertainty="prediction", method="conformal")
    phase4_bad = _mean(rows, estimand="ITTE", uncertainty="confidence baseline", method="confidence")

    # Pull a concrete weak-overlap headline when available.
    inf_naive = []
    inf_overlap = []
    for row in rows:
        if row.uncertainty != "stabilized prediction" or "overlap=4.0" not in row.scenario:
            continue
        if row.method == "naive":
            v = _float(row.inf_pct)
            if v is not None:
                inf_naive.append(v)
        if row.method == "overlap":
            v = _float(row.inf_pct)
            if v is not None:
                inf_overlap.append(v)

    lines = [
        "# Phase 7 Verdict",
        "",
        "Synthetic Level A is the coverage backbone because it has known TATE, CTATE, and ITTE truth.",
        "Real-data rows are demonstrations unless the row explicitly says it has synthetic counterfactual truth.",
        "",
    ]
    if missing:
        lines.append("Missing components in the current local outputs: " + ", ".join(missing) + ".")
        lines.append("")

    if phase4_conf is not None and phase4_bad is not None:
        lines.append(
            f"ITTE: conformal prediction averages {_fmt(phase4_conf)} coverage in parsed rows, "
            f"while the confidence-band baseline averages {_fmt(phase4_bad)} for individual curves."
        )

    # Level-B (topological pipeline in the loop): the headline external check.
    lb_conformal = _mean(rows, level="7A Level B", estimand="ITTE",
                         uncertainty="prediction", method="conformal")
    lb_tate = _mean(rows, level="7A Level B", estimand="TATE",
                    uncertainty="confidence", method="multiplier")
    lb_ctate = _mean(rows, level="7A Level B", estimand="CTATE mean",
                     uncertainty="confidence", method="ctate dr")
    if lb_conformal is not None and lb_tate is not None:
        ctate_bit = "" if lb_ctate is None else (
            f" The pointwise CTATE-mean band under-covers more sharply "
            f"({_fmt(lb_ctate)}) because its linear second stage cannot fit the step-function "
            f"loop-count conditional mean.")
        lines.append(
            f"Level B (real persistence -> silhouette pipeline): ITTE conformal averages "
            f"{_fmt(lb_conformal)} coverage across the topological sweep, while the asymptotic "
            f"TATE multiplier band averages {_fmt(lb_tate)} -- near nominal for the marginal mean "
            f"but degrading under weak overlap, since the functional-linear nuisance is "
            f"misspecified for the topological DGP and its finite-sample bias eats into the "
            f"narrower (pointwise) confidence widths.{ctate_bit} The distribution-free conformal "
            f"prediction band stays valid on the same misspecified nuisance -- the topology-specific "
            f"motivation for the conformal side."
        )
    if inf_naive and inf_overlap:
        lines.append(
            f"Weak overlap: at overlap scale 4.0, stabilized overlap weighting cuts dim-averaged "
            f"unbounded bands from {_fmt(sum(inf_naive) / len(inf_naive))}% to "
            f"{_fmt(sum(inf_overlap) / len(inf_overlap))}% in the parsed Phase 6 table."
        )
    lines.extend(
        [
            "Use asymptotic confidence bands for TATE and CTATE means; their widths target estimation error and shrink with n.",
            "Use conformal prediction bands for ITTE or CTATE-level draws; confidence bands are the wrong object for individual curves.",
            "Use adaptive conformal when aleatoric spread varies with X, and stabilized weighting when overlap is thin.",
            "GEOM-Drugs Phase 7B.3 is intentionally absent here because the dataset was not ready.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("experiments/out"))
    ap.add_argument("--output-dir", type=Path, default=Path("experiments/out/phase7"))
    args = ap.parse_args()

    rows = collect_rows(args.root)
    if not rows:
        raise SystemExit(f"no parseable experiment rows found under {args.root}")

    csv_path = args.output_dir / "master_table.csv"
    md_path = args.output_dir / "master_table.md"
    verdict_path = args.output_dir / "verdict.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    write_verdict(rows, verdict_path)
    print(f"wrote {len(rows)} rows")
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")
    print(f"verdict: {verdict_path}")


if __name__ == "__main__":
    main()
