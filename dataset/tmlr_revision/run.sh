#!/usr/bin/env bash
# TMLR revision analyses. Idempotent: a step is skipped when its output exists.
#   ./run.sh          run missing steps
#   ./run.sh force    rerun everything
#   ./run.sh status   list outputs
# Delete this folder to remove the revision analyses; nothing outside it is written.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PYTHON:-$ROOT/backend/.venv/bin/python}"
OUT="$HERE/out"
mkdir -p "$OUT"
cd "$ROOT/dataset"

if [[ "${1:-}" == "status" ]]; then ls -la "$OUT" "$OUT/figures" 2>/dev/null; exit 0; fi
force="${1:-}"

step () {  # step <output file> <log name> <args...>
  local target="$1" log="$2"; shift 2
  if [[ "$force" != "force" && -s "$target" ]]; then echo "skip  $(basename "$target")"; return; fi
  echo "run   $*"
  "$PY" -W ignore "$@" > "$OUT/$log.log" 2>&1
}

step "$OUT/t0_corrected_numbers.csv" missing_and_timing tmlr_revision/missing_and_timing.py
step "$OUT/t0_origin_mean.csv"       extras             tmlr_revision/missing_and_timing.py --extras
step "$OUT/dependence_tests.csv"     dependence_tests   tmlr_revision/dependence_tests.py
step "$OUT/selection_focal.csv"      selection_checks   tmlr_revision/selection_checks.py
step "$OUT/known_part_rolling.csv"   known_part         tmlr_revision/known_part.py
step "$OUT/false_winner_wild_check.csv" false_winner_sim tmlr_revision/false_winner_sim.py --wild-check
echo "done -> $OUT"
