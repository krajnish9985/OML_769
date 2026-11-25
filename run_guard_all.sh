#!/bin/bash
# All systems WITH the script guard, Aya + Dolly, hin/mar/tam, same settings as the runs without it.
#   bash run_guard_all.sh
# One background job per system (languages run one after another inside it); edit GPUS to free GPUs.
# Outputs go to NEW folders (results/*_guard), so the results without the guard are not touched.
# Logs: logs/<name>.log    Check progress: tail -n 2 logs/*.log
set -e
export OMP_NUM_THREADS=2
GPUS=(2 3 4 5 6)
COMMON="--data aya dolly --lang hin mar tam --script-guard"
mkdir -p logs

run() {
  gpu=$1; name=$2; shift 2
  nohup python run_generation.py "$@" $COMMON --gpu "$gpu" > "logs/$name.log" 2>&1 &
  echo "started $name on GPU $gpu (pid $!)"
}

run "${GPUS[0]}" S1_guard         --system S1 --tag guard
run "${GPUS[1]}" S2_surface_guard --system S2 --rules surface --variant hard --tau 0.5 --tag surface_hard_guard
run "${GPUS[2]}" S2_all_guard     --system S2 --rules all --variant hard --tau 0.5 --tag all_hard_guard
run "${GPUS[3]}" S3_guard         --system S3 --variant hard --tau 0.05 --tag hard_tau0.05_guard
run "${GPUS[4]}" S4_guard         --system S4 --variant hard --tau 0.3 --tag hard_tau0.3_guard
