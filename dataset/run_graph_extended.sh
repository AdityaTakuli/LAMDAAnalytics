#!/usr/bin/env bash
# Graph models (benchmark trainer, unchanged) on the larger panels, one process per seed.
#   41-country consumer panel (processed_v3): T1, then T0 regression + classification
#   seven-year panel (processed_v4), both baskets: T1, train 2019-01..2022-12, validation 2023, test 2024-01..2025-12
# Completed runs are skipped, so the queue can be restarted after an interruption:
#   setsid nohup ./run_graph_extended.sh > ../results/benchmark/graph_queue.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
LOGS=../results/benchmark/graph_queue_logs
mkdir -p "$LOGS"
WORKERS=${LAMDA_WORKERS:-4}
export OMP_NUM_THREADS=${LAMDA_THREADS:-4} MKL_NUM_THREADS=${LAMDA_THREADS:-4}
SEVEN="2019-01:2022-12,2023-01:2023-12,2024-01:2025-12"

jobs() {
  for seed in 7 13 21 42 101; do echo "processed_v3 consumer extended_2021_2024/consumer t1 $seed -"; done
  for seed in 7 13 21 42 101; do echo "processed_v4 semiconductor extended_2019_2025/semiconductor t1 $seed $SEVEN"; done
  for seed in 7 13 21 42 101; do echo "processed_v4 consumer extended_2019_2025/consumer t1 $seed $SEVEN"; done
  for seed in 7 13 21 42 101; do echo "processed_v3 consumer extended_2021_2024/consumer default $seed -"; done
}

run_one() {
  panel=$1 basket=$2 out=$3 mode=$4 seed=$5 split=$6
  log="$LOGS/${panel}_${basket}_${mode}_seed${seed}.log"
  echo "$(date '+%m-%d %H:%M') start $panel $basket $mode seed $seed"
  if [ "$split" = "-" ]; then unset LAMDA_SPLIT; else export LAMDA_SPLIT=$split; fi
  LAMDA_PANEL=$panel LAMDA_BASKET=$basket LAMDA_OUT=$out LAMDA_SEEDS=$seed \
    ../backend/.venv/bin/python graph_basket_runs.py "$mode" >> "$log" 2>&1
  echo "$(date '+%m-%d %H:%M') end   $panel $basket $mode seed $seed (exit $?)"
}
export -f run_one
export LOGS

jobs | xargs -P "$WORKERS" -L 1 bash -c 'run_one "$@"' _
echo "$(date '+%m-%d %H:%M') QUEUE FINISHED"
