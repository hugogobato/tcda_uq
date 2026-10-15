#!/usr/bin/env bash
# Phase 4.5 validation sweeps. Runs sequentially (each invocation uses 8 CPU
# workers) so peak RAM stays bounded. Results land in experiments/out/phase4/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=experiments/out/phase4
mkdir -p "$OUT"
W=${WORKERS:-8}

echo "== well-specified coverage + confidence-vs-prediction contrast =="
$PY -u experiments/phase4_itte_coverage.py \
    --ns 200 500 1000 --reps 200 --alpha 0.1 --workers "$W" --baseline \
    --output "$OUT/wellspec.txt"

echo "== overlap sweep (width & inf% vs positivity) =="
for S in 1.0 2.5 4.0; do
  $PY -u experiments/phase4_itte_coverage.py \
      --ns 600 --reps 150 --alpha 0.1 --workers "$W" --overlap "$S" \
      --output "$OUT/overlap_${S}.txt"
done

echo "== double-robustness sweep (mis-specify pi / mu / both) =="
for MS in none pi mu both; do
  $PY -u experiments/phase4_itte_coverage.py \
      --ns 800 --reps 150 --alpha 0.1 --workers "$W" --misspec "$MS" \
      --output "$OUT/dr_${MS}.txt"
done

echo "== ALL PHASE 4 SWEEPS DONE =="
