"""Aggregate Phase 2.6 coverage shards into one weighted table."""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
from pathlib import Path
import re


METHOD_ORDER = ["pointwise", "multiplier", "liebl_reimherr", "pini_vantini"]
ROW_RE = re.compile(
    r"^\s*(?P<n>\d+)\s+(?P<method>\S+)\s+(?P<dim>\d+)\s+"
    r"(?P<cov>[0-9.]+)\s+(?P<iwe>[0-9.]+)\s+(?P<width>[0-9.]+)\s*$"
)
HEADER_REPS_RE = re.compile(r"\breps=(?P<local>\d+)(?:/(?P<total>\d+))?\b")


def _expand(paths):
    expanded = []
    for path in paths:
        matches = sorted(glob.glob(path))
        expanded.extend(matches or [path])
    return [Path(p) for p in expanded]


def _file_weight(path, default_reps):
    text = path.read_text().splitlines()
    for line in text:
        if line.startswith("Phase 2.6 coverage study"):
            m = HEADER_REPS_RE.search(line)
            if m:
                return int(m.group("local"))
    return default_reps


def _read_rows(path, weight):
    for line in path.read_text().splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        yield (
            int(m.group("n")),
            m.group("method"),
            int(m.group("dim")),
            float(m.group("cov")),
            float(m.group("iwe")),
            float(m.group("width")),
            weight,
        )


def aggregate(files, default_reps):
    sums = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for path in files:
        weight = _file_weight(path, default_reps)
        for n, method, dim, cov, iwe, width, row_weight in _read_rows(path, weight):
            acc = sums[(n, method, dim)]
            acc[0] += cov * row_weight
            acc[1] += iwe * row_weight
            acc[2] += width * row_weight
            acc[3] += row_weight
    return sums


def _sort_key(item):
    n, method, dim = item
    try:
        method_pos = METHOD_ORDER.index(method)
    except ValueError:
        method_pos = len(METHOD_ORDER)
    return n, method_pos, dim


def format_table(sums, target_reps, alpha):
    lines = [
        (
            f"Phase 2.6 coverage study | target={1-alpha:.2f} "
            f"| reps={target_reps} | aggregated"
        ),
        f"{'n':>5} {'method':>15} {'dim':>3} {'cov(simult)':>12} "
        f"{'iw_err':>8} {'mean_width':>11}",
    ]
    for key in sorted(sums, key=_sort_key):
        n, method, dim = key
        cov_sum, iwe_sum, width_sum, weight = sums[key]
        lines.append(
            f"{n:>5} {method:>15} {dim:>3} {cov_sum / weight:>12.3f} "
            f"{iwe_sum / weight:>8.3f} {width_sum / weight:>11.4f}"
        )
        if weight != target_reps:
            lines.append(f"# n={n} {method} dim={dim} aggregated_reps={weight}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--reps", type=int, default=150)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    files = _expand(args.inputs)
    sums = aggregate(files, args.reps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(format_table(sums, args.reps, args.alpha))


if __name__ == "__main__":
    main()
