"""Synchronous P2 production chunk runner (multiprocessing inside one call).

The sandbox reaps detached background jobs, so production replications run in
parallel shards INSIDE this process tree, one chunk per invocation.

Usage:
  python experiments/pacce_p2_chunk.py --scenarios null,sld_only \
      --reps 250 --n 842 --workers 8 --tag mainA
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import experiments.pacce_simulation as M


def _work(job):
    scen, seed, n, alpha, hom = job
    t0 = time.monotonic()
    std_consts = M.frozen_standardization()
    tseq = np.linspace(M.SPEC_V1["silhouette"]["interval"][0],
                       M.SPEC_V1["silhouette"]["interval"][1], 60)
    tp = M.OUT / f"truth_{scen}_res60_h{hom}.json"
    if tp.exists():
        truth = json.load(open(tp))
    else:
        truth = M.scenario_truths(scen, resolution=60, std_consts=std_consts,
                                  hom=hom)
        with open(tp, "w") as f:
            json.dump(truth, f)
    try:
        sim = M.PacceSimulator(scen, n=n, seed=seed, std_consts=std_consts,
                               resolution=60, hom=hom)
        d = sim.one_draw()
        d["_dgp"] = dict(scenario=scen, std_consts=std_consts, resolution=60)
        rec = M.estimate_rep(d, tseq, truth, alpha=alpha, seed=seed)
        rec.update(scenario=scen, seed=seed, n=n,
                   elapsed=time.monotonic() - t0)
    except Exception as e:
        rec = dict(scenario=scen, seed=seed, n=n,
                   error=f"{type(e).__name__}: {e}",
                   elapsed=time.monotonic() - t0)
    return rec


def main():

    from multiprocessing import Pool

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--reps", type=int, required=True)
    ap.add_argument("--n", type=int, default=842)
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--hom", type=int, default=0)
    ap.add_argument("--tag", default="chunk")
    args = ap.parse_args()

    jobs = [(s, seed, args.n, args.alpha, args.hom)
            for s in args.scenarios.split(",")
            for seed in range(args.seed0, args.seed0 + args.reps)]
    print(f"{len(jobs)} replications on {args.workers} workers", flush=True)
    t0 = time.monotonic()
    with Pool(args.workers) as pool:
        rows = []
        for i, rec in enumerate(pool.imap_unordered(_work, jobs, chunksize=2)):
            rows.append(rec)
            if (i + 1) % 50 == 0:
                el = time.monotonic() - t0
                print(f"  {i+1}/{len(jobs)} done ({el:.0f}s, "
                      f"eta {el / (i+1) * (len(jobs)-i-1):.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    out = M.OUT / f"{args.tag}_reps{args.reps}_n{args.n}.parquet"
    df.to_parquet(out, index=False)
    ok = df[df["error"].isna()] if "error" in df.columns else df
    if len(ok) and "tate_covered" in ok:
        summ = ok.groupby("scenario")[["tate_covered", "tate_reject",
                                       "sld_covered", "itte_cov"]].mean()
        print(summ.round(3).to_string())
    n_err = int(df["error"].notna().sum()) if "error" in df.columns else 0
    print(f"wrote {out} ({len(df)} rows, {n_err} errors, "
          f"{time.monotonic() - t0:.0f}s)")


if __name__ == "__main__":
    main()
