#!/usr/bin/env bash
# Hyperparameter sensitivity of the graph models on the seven-year panel (T1, both baskets):
# seven variants around the paper's setting (graph_basket_runs.py TUNE), three seeds each,
# same split as tab:sevengraph. One process per run; finished runs are skipped, so restart with
#   setsid nohup ./run_graph_tuning.sh > ../results/benchmark/graph_tune.log 2>&1 < /dev/null &
# Outputs: results/benchmark/extended_2019_2025/<basket>/graph_tune/<variant>/seed_<s>/
set -u
cd "$(dirname "$0")"
LOGS=../results/benchmark/graph_tune_logs
mkdir -p "$LOGS"
WORKERS=${LAMDA_WORKERS:-5}
export OMP_NUM_THREADS=${LAMDA_THREADS:-3} MKL_NUM_THREADS=${LAMDA_THREADS:-3}
export LAMDA_PANEL=processed_v4 LAMDA_SPLIT="2019-01:2022-12,2023-01:2023-12,2024-01:2025-12" LOGS

jobs() {
  for basket in semiconductor consumer; do
    for variant in mse_loss lr_low lr_high weight_decay small large long; do
      for seed in 7 13 21; do echo "$basket $variant $seed"; done
    done
  done
}
run_one() {
  basket=$1 variant=$2 seed=$3
  echo "$(date '+%m-%d %H:%M') start $basket $variant seed $seed"
  LAMDA_BASKET=$basket LAMDA_OUT=extended_2019_2025/$basket LAMDA_TUNE=$variant LAMDA_SEEDS=$seed \
    ../backend/.venv/bin/python graph_basket_runs.py tune >> "$LOGS/${basket}_${variant}_seed${seed}.log" 2>&1
  echo "$(date '+%m-%d %H:%M') end   $basket $variant seed $seed (exit $?)"
}
export -f run_one
jobs | xargs -P "$WORKERS" -L 1 bash -c 'run_one "$@"' _
echo "$(date '+%m-%d %H:%M') TUNE QUEUE FINISHED"
