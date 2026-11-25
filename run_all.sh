# # #!/bin/bash
# # # All five systems on Aya + Dolly, hin/mar/tam, with the chosen settings, one background job per GPU.
# # #   PAIR=qwen bash run_all.sh            # Qwen pair, no guard   -> results_qwen/S1, S2_surface_hard, ...
# # #   PAIR=qwen bash run_all.sh guard      # Qwen pair, with guard -> results_qwen/*_guard
# # #   bash run_all.sh                      # BLOOMZ pair (results/), same as the earlier runs
# # #   TEMP=1.0 bash run_all.sh             # temperature 1.0 -> results/S1_T1.0, S2_surface_hard_T1.0, ...
# # #   DATA=flores LANGS=mar bash run_all.sh   # English -> Marathi translation (adds flores_mar.jsonl to each folder)
# # #   (TEMP, DATA, LANGS combine; defaults 0.7, "aya dolly", "hin mar tam")
# # # Edit GPUS to free GPUs. Logs: logs/<pair>_<name>.log    Progress: tail -n 2 logs/*.log
# # set -e
# # export OMP_NUM_THREADS=2
# # GPUS=(0 1 2 3 4)
# # SET=${1:-noguard}
# # PAIR=${PAIR:-bloomz}
# # export PAIR
# # case "$SET" in
# #   noguard) G="";              S="" ;;
# #   guard)   G="--script-guard"; S="_guard" ;;
# #   *) echo "argument must be noguard or guard"; exit 1 ;;
# # esac
# # TEMP=${TEMP:-0.7}
# # DATA=${DATA:-aya dolly}
# # LANGS=${LANGS:-hin mar tam}
# # if [ "$TEMP" != 0.7 ]; then S="${S}_T$TEMP"; fi   # folder suffix, so T=0.7 results are never overwritten
# # COMMON="--data $DATA --lang $LANGS --temperature $TEMP $G"
# # mkdir -p logs

# # run() {
# #   gpu=$1; name=$2; shift 2
# #   nohup python run_generation.py "$@" $COMMON --gpu "$gpu" > "logs/${PAIR}_$name$S.log" 2>&1 &
# #   echo "started $name$S ($PAIR) on GPU $gpu (pid $!)"
# # }

# # if [ -n "$S" ]; then T1="--tag ${S#_}"; else T1=""; fi
# # run "${GPUS[0]}" S1         --system S1 $T1
# # run "${GPUS[1]}" S2_surface --system S2 --rules surface --variant hard --tau 0.5 --tag "surface_hard$S"
# # run "${GPUS[2]}" S2_all     --system S2 --rules all --variant hard --tau 0.5 --tag "all_hard$S"
# # run "${GPUS[3]}" S3         --system S3 --variant hard --tau 0.05 --tag "hard_tau0.05$S"
# # run "${GPUS[4]}" S4         --system S4 --variant hard --tau 0.3 --tag "hard_tau0.3$S"

# #!/bin/bash
# # All five systems on Aya + Dolly, hin/mar/tam, with the chosen settings, one background job per GPU.
# #   PAIR=qwen bash run_all.sh            # Qwen pair, no guard   -> results_qwen/S1, S2_surface_hard, ...
# #   PAIR=qwen bash run_all.sh guard      # Qwen pair, with guard -> results_qwen/*_guard
# #   bash run_all.sh                      # BLOOMZ pair (results/), same as the earlier runs
# #   TEMP=1.0 bash run_all.sh             # temperature 1.0 -> results/S1_T1.0, S2_surface_hard_T1.0, ...
# #   DATA=flores LANGS=mar bash run_all.sh   # English -> Marathi translation (adds flores_mar.jsonl to each folder)
# #   (TEMP, DATA, LANGS combine; defaults 0.7, "aya dolly", "hin mar tam")
# #   ONE_GPU=0 bash run_all.sh            # only one free GPU: the 5 systems run one after another on GPU 0
# #                                          (in the background; survives logout; same folders and logs)
# # Edit GPUS to free GPUs. Logs: logs/<pair>_<name>.log    Progress: tail -n 2 logs/*.log
# set -e
# export OMP_NUM_THREADS=2
# GPUS=(0 1 2 3 4)
# SET=${1:-noguard}
# PAIR=${PAIR:-bloomz}
# export PAIR
# case "$SET" in
#   noguard) G="";              S="" ;;
#   guard)   G="--script-guard"; S="_guard" ;;
#   *) echo "argument must be noguard or guard"; exit 1 ;;
# esac
# TEMP=${TEMP:-0.7}
# [[ "$TEMP" =~ ^[0-9]+(\.[0-9]+)?$ ]] || TEMP=0.7   # some systems set TEMP to a temp folder
# DATA=${DATA:-aya dolly}
# LANGS=${LANGS:-hin mar tam}
# if [ "$TEMP" != 0.7 ]; then S="${S}_T$TEMP"; fi   # folder suffix, so T=0.7 results are never overwritten
# COMMON="--data $DATA --lang $LANGS --temperature $TEMP $G"
# mkdir -p logs

# ONE_GPU=${ONE_GPU:-}
# JOBS=()   # sequential mode: commands collected here, run one by one at the end

# run() {
#   gpu=$1; name=$2; shift 2
#   if [ -n "$ONE_GPU" ]; then
#     JOBS+=("python run_generation.py $* $COMMON --gpu $ONE_GPU < /dev/null > logs/${PAIR}_$name$S.log 2>&1")
#     return
#   fi
#   nohup python run_generation.py "$@" $COMMON --gpu "$gpu" > "logs/${PAIR}_$name$S.log" 2>&1 &
#   echo "started $name$S ($PAIR) on GPU $gpu (pid $!)"
# }

# if [ -n "$S" ]; then T1="--tag ${S#_}"; else T1=""; fi
# run "${GPUS[0]}" S1         --system S1 $T1
# run "${GPUS[1]}" S2_surface --system S2 --rules surface --variant hard --tau 0.5 --tag "surface_hard$S"
# run "${GPUS[2]}" S2_all     --system S2 --rules all --variant hard --tau 0.5 --tag "all_hard$S"
# run "${GPUS[3]}" S3         --system S3 --variant hard --tau 0.05 --tag "hard_tau0.05$S"
# run "${GPUS[4]}" S4         --system S4 --variant hard --tau 0.3 --tag "hard_tau0.3$S"

# if [ -n "$ONE_GPU" ]; then
#   SEQ_LOG="logs/${PAIR}_sequential$S.log"
#   printf '%s\n' "${JOBS[@]}" > "$SEQ_LOG.cmds"
#   # each job starts after the previous one ends (a failed job does not stop the rest)
#   nohup bash -c 'while IFS= read -r cmd; do echo "$(date) start: $cmd"; eval "$cmd" || echo "$(date) FAILED"; done < "$0"; echo "$(date) all done"' \
#     "$SEQ_LOG.cmds" > "$SEQ_LOG" 2>&1 &
#   echo "started ${#JOBS[@]} jobs one after another on GPU $ONE_GPU (pid $!). Progress: cat $SEQ_LOG; tail -n 2 logs/*.log"
# fi


#!/bin/bash
# All five systems on Aya + Dolly, hin/mar/tam, with the chosen settings, one background job per GPU.
#   PAIR=qwen bash run_all.sh            # Qwen pair, no guard   -> results_qwen/S1, S2_surface_hard, ...
#   PAIR=qwen bash run_all.sh guard      # Qwen pair, with guard -> results_qwen/*_guard
#   bash run_all.sh                      # BLOOMZ pair (results/), same as the earlier runs
#   TEMP=1.0 bash run_all.sh             # temperature 1.0 -> results/S1_T1.0, S2_surface_hard_T1.0, ...
#   DATA=flores LANGS=mar bash run_all.sh   # English -> Marathi translation (adds flores_mar.jsonl to each folder)
#   (TEMP, DATA, LANGS combine; defaults 0.7, "aya dolly", "hin mar tam")
#   CT=1 bash run_all.sh                 # modules also judge target-sampled tokens -> S2_surface_hard_ct, ...
#   ONE_GPU=0 bash run_all.sh            # only one free GPU: the 5 systems run one after another on GPU 0
#                                          (in the background; survives logout; same folders and logs)
# Edit GPUS to free GPUs. Logs: logs/<pair>_<name>.log    Progress: tail -n 2 logs/*.log
set -e
export OMP_NUM_THREADS=2
GPUS=(0 1 2 3 4)
SET=${1:-noguard}
PAIR=${PAIR:-bloomz}
export PAIR
case "$SET" in
  noguard) G="";              S="" ;;
  guard)   G="--script-guard"; S="_guard" ;;
  *) echo "argument must be noguard or guard"; exit 1 ;;
esac
CT=${CT:-}   # CT=1: modules also judge target-sampled tokens (--check-target), folders get _ct
if [ -n "$CT" ]; then G="$G --check-target"; S="${S}_ct"; fi
TEMP=${TEMP:-0.7}
[[ "$TEMP" =~ ^[0-9]+(\.[0-9]+)?$ ]] || TEMP=0.7   # some systems set TEMP to a temp folder
DATA=${DATA:-aya dolly}
LANGS=${LANGS:-hin mar tam}
if [ "$TEMP" != 0.7 ]; then S="${S}_T$TEMP"; fi   # folder suffix, so T=0.7 results are never overwritten
COMMON="--data $DATA --lang $LANGS --temperature $TEMP $G"
mkdir -p logs

ONE_GPU=${ONE_GPU:-}
JOBS=()   # sequential mode: commands collected here, run one by one at the end

run() {
  gpu=$1; name=$2; shift 2
  if [ -n "$ONE_GPU" ]; then
    JOBS+=("python run_generation.py $* $COMMON --gpu $ONE_GPU < /dev/null > logs/${PAIR}_$name$S.log 2>&1")
    return
  fi
  nohup python run_generation.py "$@" $COMMON --gpu "$gpu" > "logs/${PAIR}_$name$S.log" 2>&1 &
  echo "started $name$S ($PAIR) on GPU $gpu (pid $!)"
}

if [ -n "$S" ]; then T1="--tag ${S#_}"; else T1=""; fi
run "${GPUS[0]}" S1         --system S1 $T1
run "${GPUS[1]}" S2_surface --system S2 --rules surface --variant hard --tau 0.5 --tag "surface_hard$S"
run "${GPUS[2]}" S2_all     --system S2 --rules all --variant hard --tau 0.5 --tag "all_hard$S"
run "${GPUS[3]}" S3         --system S3 --variant hard --tau 0.05 --tag "hard_tau0.05$S"
run "${GPUS[4]}" S4         --system S4 --variant hard --tau 0.3 --tag "hard_tau0.3$S"

if [ -n "$ONE_GPU" ]; then
  SEQ_LOG="logs/${PAIR}_sequential$S.log"
  printf '%s\n' "${JOBS[@]}" > "$SEQ_LOG.cmds"
  # each job starts after the previous one ends (a failed job does not stop the rest)
  nohup bash -c 'while IFS= read -r cmd; do echo "$(date) start: $cmd"; eval "$cmd" || echo "$(date) FAILED"; done < "$0"; echo "$(date) all done"' \
    "$SEQ_LOG.cmds" > "$SEQ_LOG" 2>&1 &
  echo "started ${#JOBS[@]} jobs one after another on GPU $ONE_GPU (pid $!). Progress: cat $SEQ_LOG; tail -n 2 logs/*.log"
fi
