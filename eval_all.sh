# #!/bin/bash
# # Final evaluation on Aya + Dolly, all languages, all metrics.
# #   bash eval_all.sh 0            # runs without the guard  -> results/final_table.csv, results/final_eval.log
# #   bash eval_all.sh 0 guard      # runs with the guard     -> results/final_table_guard.csv, results/final_eval_guard.log
# #   bash eval_all.sh 0 both       # all 10 folders in one table -> results/final_table_both.csv, ...
# #   TEMP=1.0 bash eval_all.sh 0   # the T=1.0 folders  -> results/final_table_T1.0.csv
# #   DATA=flores bash eval_all.sh 0  # translation files  -> results/final_table_flores.csv (chrF++ vs FLORES reference)
# # 0 = GPU for Stanza and the judge model.
# # Qwen pair: PAIR=qwen bash eval_all.sh 0   (reads and writes results_qwen/ instead of results/)
# set -e
# export OMP_NUM_THREADS=2
# GPU=${1:-0}
# SET=${2:-noguard}
# R=results
# [ "${PAIR:-bloomz}" != bloomz ] && R="results_$PAIR"
# TEMP=${TEMP:-0.7}
# DATA=${DATA:-aya dolly}
# T=""; [ "$TEMP" != 0.7 ] && T="_T$TEMP"
# D=""; [ "$DATA" != "aya dolly" ] && D="_${DATA// /_}"

# if [ -n "$T" ]; then S1="$R/S1$T"; S1G="$R/S1_guard$T"; else S1="$R/S1"; S1G="$R/S1_guard"; fi
# BASE="$S1 $R/S2_surface_hard$T $R/S2_all_hard$T $R/S3_hard_tau0.05$T $R/S4_hard_tau0.3$T"
# GUARD="$S1G $R/S2_surface_hard_guard$T $R/S2_all_hard_guard$T $R/S3_hard_tau0.05_guard$T $R/S4_hard_tau0.3_guard$T"
# case "$SET" in
#   noguard) SYSTEMS=$BASE;  SUFFIX="" ;;
#   guard)   SYSTEMS=$GUARD; SUFFIX="_guard" ;;
#   both)    SYSTEMS="$BASE $GUARD"; SUFFIX="_both" ;;
#   *) echo "second argument must be noguard, guard or both"; exit 1 ;;
# esac
# for d in $SYSTEMS; do
#   [ -d "$d" ] || { echo "missing folder: $d"; exit 1; }
# done

# SUFFIX="$SUFFIX$T$D"
# python evaluate.py $SYSTEMS --data $DATA --stanza --judge sarvamai/sarvam-1 --gpu "$GPU" \
#   --csv "$R/final_table$SUFFIX.csv" 2>&1 | tee "$R/final_eval$SUFFIX.log"


#!/bin/bash
# Final evaluation on Aya + Dolly, all languages, all metrics.
#   bash eval_all.sh 0            # runs without the guard  -> results/final_table.csv, results/final_eval.log
#   bash eval_all.sh 0 guard      # runs with the guard     -> results/final_table_guard.csv, results/final_eval_guard.log
#   bash eval_all.sh 0 both       # all 10 folders in one table -> results/final_table_both.csv, ...
#   TEMP=1.0 bash eval_all.sh 0   # the T=1.0 folders  -> results/final_table_T1.0.csv
#   CT=1 bash eval_all.sh 0       # the --check-target folders -> results/final_table_ct.csv
#   DATA=flores bash eval_all.sh 0  # translation files  -> results/final_table_flores.csv (chrF++ vs FLORES reference)
# 0 = GPU for Stanza and the judge model.
# Qwen pair: PAIR=qwen bash eval_all.sh 0   (reads and writes results_qwen/ instead of results/)
set -e
export OMP_NUM_THREADS=2
GPU=${1:-0}
SET=${2:-noguard}
R=results
[ "${PAIR:-bloomz}" != bloomz ] && R="results_$PAIR"
TEMP=${TEMP:-0.7}
[[ "$TEMP" =~ ^[0-9]+(\.[0-9]+)?$ ]] || TEMP=0.7   # some systems set TEMP to a temp folder
DATA=${DATA:-aya dolly}
T=""; [ -n "${CT:-}" ] && T="_ct"          # CT=1: the --check-target folders
[ "$TEMP" != 0.7 ] && T="${T}_T$TEMP"
D=""; [ "$DATA" != "aya dolly" ] && D="_${DATA// /_}"

if [ -n "$T" ]; then S1="$R/S1$T"; S1G="$R/S1_guard$T"; else S1="$R/S1"; S1G="$R/S1_guard"; fi
BASE="$S1 $R/S2_surface_hard$T $R/S2_all_hard$T $R/S3_hard_tau0.05$T $R/S4_hard_tau0.3$T"
GUARD="$S1G $R/S2_surface_hard_guard$T $R/S2_all_hard_guard$T $R/S3_hard_tau0.05_guard$T $R/S4_hard_tau0.3_guard$T"
case "$SET" in
  noguard) SYSTEMS=$BASE;  SUFFIX="" ;;
  guard)   SYSTEMS=$GUARD; SUFFIX="_guard" ;;
  both)    SYSTEMS="$BASE $GUARD"; SUFFIX="_both" ;;
  *) echo "second argument must be noguard, guard or both"; exit 1 ;;
esac
for d in $SYSTEMS; do
  [ -d "$d" ] || { echo "missing folder: $d"; exit 1; }
done

SUFFIX="$SUFFIX$T$D"
python evaluate.py $SYSTEMS --data $DATA --stanza --judge sarvamai/sarvam-1 --gpu "$GPU" \
  --csv "$R/final_table$SUFFIX.csv" 2>&1 | tee "$R/final_eval$SUFFIX.log"
