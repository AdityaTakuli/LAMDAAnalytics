#!/usr/bin/env bash
# Rebuild the paper's canonical numbers from the fused corpus and frozen graph runs.
# Usage: ./reproduce.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/backend/.venv/bin/python" ]]; then
  PY="$ROOT/backend/.venv/bin/python"
elif [[ -x "$ROOT/dataset/.venv/bin/python" ]]; then
  PY="$ROOT/dataset/.venv/bin/python"
else
  PY="${PYTHON:-python3}"
fi

echo "Using $PY"
"$PY" -c "import pandas, numpy, sklearn, lightgbm" >/dev/null

cd "$ROOT/dataset"
"$PY" build_canonical.py

echo "Wrote results/canonical_numbers.csv"
echo "Frozen graph ensembles remain in results/phase07/ (not retrained)."
