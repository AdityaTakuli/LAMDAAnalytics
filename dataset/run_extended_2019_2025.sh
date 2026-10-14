#!/usr/bin/env bash
# Extended panel, 2019-2025: wait for the 2019-2025 GDELT features, build the panel, run the analyses.
cd "$(dirname "$0")"
PY=../backend/.venv/bin/python
LOG=../results/benchmark
while pgrep -f gdelt_features_extended.py > /dev/null; do sleep 30; done
[ -f data/four_year_2021_2024/processed_v4/gdelt_monthly_features.csv ] || { echo "GDELT features missing"; exit 1; }
tail -n 30 $LOG/gdelt_extended_2019_2025_features.log | grep check
$PY build_extended_2019_2025_panel.py > $LOG/extended_2019_2025_build.log 2>&1 || { echo "BUILD FAILED"; tail -n 5 $LOG/extended_2019_2025_build.log; exit 1; }
echo "BUILD OK"
$PY extended_rolling_blocks.py > $LOG/extended_2019_2025_analyses.log 2>&1 && echo "ANALYSES OK" || echo "ANALYSES FAILED"
