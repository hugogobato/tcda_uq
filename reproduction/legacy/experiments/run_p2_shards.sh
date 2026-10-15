#!/bin/zsh
# P2 production driver: 6 concurrent shards, tier A then tier B, fully detached.
cd /home/hugo_souto/Stuff/Research/CP_TATE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
PYTHON=.venv/bin/python
LOG=results/pacce_simulation/logs

mkdir -p $LOG

run_tier() {
  local scen=$1 reps=$2 tag=$3 seed0=$4 n=$5
  for k in $(seq 0 5); do
    env PYTHONPATH=. $PYTHON experiments/pacce_simulation.py \
      --scenarios $scen --reps $reps --n $n --shard $k --num_shards 6 \
      --seed0 $seed0 --tag $tag > $LOG/${tag}$k.log 2>&1 &
    pids+=($!)
  done
  wait ${pids[@]}
}

pids=()
run_tier null,sld_only,topo_favorable 500 mainA 1000 842 &
wait $!

pids=()
run_tier topo_null,nuisance_misp,irregular_visits,dropout_arm 250 mainB 7000 842 &
wait $!

pids=()
run_tier null,sld_only,topo_favorable 300 n300 9000 300 &
wait $!

echo "ALL TIERS DONE" > $LOG/done.txt
