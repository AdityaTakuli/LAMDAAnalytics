#!/usr/bin/env bash
# Paper-21 revision: keep the Phase B/D Comtrade downloads running until every request is cached.
# Each worker is restarted whenever it exits (all scripts resume from their caches).
# Needs the local Psiphon proxy on 127.0.0.1:8081 (start it with `psiphon` from the repo root).
# Run detached:  setsid nohup ./rev21_run_downloads.sh > ../results/paper21_revision/supervisor.log 2>&1 &
cd "$(dirname "$0")"
PY=../backend/.venv/bin/python
LOG=../results/paper21_revision
CACHE=data/four_year_2021_2024/cache

count() { ls "$1" 2>/dev/null | grep -c "$2"; }

worker() {  # name, required count, cache dir, pattern, command...
  local name=$1 need=$2 dir=$3 pat=$4; shift 4
  while [ "$(count "$dir" "$pat")" -lt "$need" ]; do
    echo "$(date +%H:%M) start $name ($(count "$dir" "$pat")/$need)"
    env "$@" >> "$LOG/$name.log" 2>&1
    sleep 20
  done
  echo "$(date +%H:%M) done $name"
}

# Phase B: only large reporters remain; fetch them in quarters/months.
worker download_split_phaseB 96 $CACHE/comtrade/phaseB 'json$' $PY rev21_download_split.py phaseB &
# Phase D: full-year requests from three positions in the list, plus the split fetcher for stragglers.
worker download_phaseD_w1 123 $CACHE/comtrade/phaseD 'json$' $PY rev21_download_phaseD.py &
worker download_phaseD_w2 123 $CACHE/comtrade/phaseD 'json$' REV21_REVERSE=1 $PY rev21_download_phaseD.py &
worker download_phaseD_w3 123 $CACHE/comtrade/phaseD 'json$' REV21_MIDDLE=1 $PY rev21_download_phaseD.py &
wait
echo "$(date +%H:%M) ALL DOWNLOADS COMPLETE"
