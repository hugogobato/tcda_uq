"""Aggregate sharded Phase 3.4 CTATE coverage outputs into one table.

Each shard file (from ``phase3_ctate_coverage.py --num-shards K --shard-id i``)
holds per-shard *means* over that shard's equal slice of reps. With equal-sized
shards the grand mean over all reps is just the mean of the shard means, grouped
by ``(n, x, method, dim)``.

Usage:
    python experiments/aggregate_phase3_coverage.py experiments/out/phase3/shard*.txt \
        [--output experiments/out/phase3/aggregated.txt]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def _parse(paths):
    """Return {(n, x, method, dim): [rows...]} where each row is 4 floats."""
    rows = defaultdict(list)
    x_order, method_order = [], []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            tok = line.split()
            if len(tok) != 8 or not tok[0].isdigit():
                continue
            n, x, method, dim = int(tok[0]), tok[1], tok[2], int(tok[3])
            try:
                vals = [float(t) for t in tok[4:]]
            except ValueError:
                continue
            rows[(n, x, method, dim)].append(vals)
            if x not in x_order:
                x_order.append(x)
            if method not in method_order:
                method_order.append(method)
    return rows, x_order, method_order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    rows, x_order, method_order = _parse(args.files)
    if not rows:
        raise SystemExit("no data rows found in the given files")

    ns = sorted({k[0] for k in rows})
    dims = sorted({k[3] for k in rows})
    n_shards = max(len(v) for v in rows.values())

    lines = [f"# aggregated {len(args.files)} shard file(s), up to {n_shards} shards/key",
             f"{'n':>5} {'x':>5} {'method':>11} {'dim':>3} {'recovery':>9} "
             f"{'cov(sim)':>12} {'iw_err':>8} {'mean_width':>11} {'shards':>7}"]
    for n in ns:
        for x in x_order:
            for d in dims:
                for m in method_order:
                    vals = rows.get((n, x, m, d))
                    if not vals:
                        continue
                    k = len(vals)
                    avg = [sum(col) / k for col in zip(*vals)]
                    lines.append(
                        f"{n:>5} {x:>5} {m:>11} {d:>3} {avg[0]:>9.3f} "
                        f"{avg[1]:>12.3f} {avg[2]:>8.3f} {avg[3]:>11.4f} {k:>7}")

    out = "\n".join(lines)
    print(out)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(out + "\n")
        print(f"\n# written to {args.output}")


if __name__ == "__main__":
    main()
