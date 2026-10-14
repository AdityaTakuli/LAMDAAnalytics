#!/usr/bin/env bash
# Keep the extended-panel Comtrade downloads running until every request is cached.
# Each worker is restarted whenever it exits (all scripts resume from their caches).
# Needs the local Psiphon proxy on 127.0.0.1:8081 (start it with `psiphon` from the repo root).
# Run detached:  setsid nohup ./run_downloads.sh > ../results/benchmark/supervisor.log 2>&1 &
cd "$(dirname "$0")"
PY=../backend/.venv/bin/python
LOG=../results/benchmark
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

# Extended panel, 2021-2024: only large reporters remain; fetch them in quarters/months.
worker download_split_extended_2021_2024 96 $CACHE/comtrade/extended_2021_2024 'json$' $PY download_comtrade_split.py extended_2021_2024 &
# Extended panel, 2019-2025: full-year requests from three positions in the list, plus the split fetcher for stragglers.
worker download_extended_2019_2025_w1 123 $CACHE/comtrade/extended_2019_2025 'json$' $PY download_comtrade_2019_2025.py &
worker download_extended_2019_2025_w2 123 $CACHE/comtrade/extended_2019_2025 'json$' LAMDA_REVERSE=1 $PY download_comtrade_2019_2025.py &
worker download_extended_2019_2025_w3 123 $CACHE/comtrade/extended_2019_2025 'json$' LAMDA_MIDDLE=1 $PY download_comtrade_2019_2025.py &
wait
echo "$(date +%H:%M) ALL DOWNLOADS COMPLETE"
