#!/usr/bin/env bash
# Phase D chain: wait for the 2019-2025 GDELT features, build the panel, run the analyses.
cd "$(dirname "$0")"
PY=../backend/.venv/bin/python
LOG=../results/paper21_revision
while pgrep -f rev21_gdelt_features.py > /dev/null; do sleep 30; done
[ -f data/four_year_2021_2024/processed_v4/gdelt_monthly_features.csv ] || { echo "GDELT features missing"; exit 1; }
tail -n 30 $LOG/gdelt_phaseD_features.log | grep check
$PY rev21_build_phaseD_panel.py > $LOG/phaseD_build.log 2>&1 || { echo "BUILD FAILED"; tail -n 5 $LOG/phaseD_build.log; exit 1; }
echo "BUILD OK"
$PY rev21_phaseD_analyses.py > $LOG/phaseD_analyses.log 2>&1 && echo "ANALYSES OK" || echo "ANALYSES FAILED"
