#!/usr/bin/env bash
# Graph models (paper-21 trainer, unchanged) on the larger panels, one process per seed.
#   41-country consumer panel (processed_v3): T1, then T0 regression + classification
#   seven-year panel (processed_v4), both baskets: T1, train 2019-01..2022-12, validation 2023, test 2024-01..2025-12
# Completed runs are skipped, so the queue can be restarted after an interruption:
#   setsid nohup ./rev21_run_graph_queue.sh > ../results/paper21_revision/graph_queue.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")"
LOGS=../results/paper21_revision/graph_queue_logs
mkdir -p "$LOGS"
WORKERS=${REV21_WORKERS:-4}
export OMP_NUM_THREADS=${REV21_THREADS:-4} MKL_NUM_THREADS=${REV21_THREADS:-4}
SEVEN="2019-01:2022-12,2023-01:2023-12,2024-01:2025-12"

jobs() {
  for seed in 7 13 21 42 101; do echo "processed_v3 consumer phaseB/consumer t1 $seed -"; done
  for seed in 7 13 21 42 101; do echo "processed_v4 semiconductor phaseD/semiconductor t1 $seed $SEVEN"; done
  for seed in 7 13 21 42 101; do echo "processed_v4 consumer phaseD/consumer t1 $seed $SEVEN"; done
  for seed in 7 13 21 42 101; do echo "processed_v3 consumer phaseB/consumer default $seed -"; done
}

run_one() {
  panel=$1 basket=$2 out=$3 mode=$4 seed=$5 split=$6
  log="$LOGS/${panel}_${basket}_${mode}_seed${seed}.log"
  echo "$(date '+%m-%d %H:%M') start $panel $basket $mode seed $seed"
  if [ "$split" = "-" ]; then unset REV21_SPLIT; else export REV21_SPLIT=$split; fi
  REV21_PANEL=$panel REV21_BASKET=$basket REV21_OUT=$out REV21_SEEDS=$seed \
    ../backend/.venv/bin/python rev21_consumer_graphs.py "$mode" >> "$log" 2>&1
  echo "$(date '+%m-%d %H:%M') end   $panel $basket $mode seed $seed (exit $?)"
}
export -f run_one
export LOGS

jobs | xargs -P "$WORKERS" -L 1 bash -c 'run_one "$@"' _
echo "$(date '+%m-%d %H:%M') QUEUE FINISHED"
