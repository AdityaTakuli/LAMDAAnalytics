#!/usr/bin/env bash
# Timed live /analyze against a running server. Redirects to a .log (not a screenshot).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${LAMDA_EVIDENCE_DIR:-$ROOT/evidence/runs/$STAMP}"
mkdir -p "$OUT"
BASE="${LAMDA_BASE_URL:-http://127.0.0.1:8007}"
PAYLOAD="${1:-$ROOT/evidence/payloads/analyze_example.json}"
PY="${ROOT}/backend/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"

{
  echo "=== timed POST /analyze (curl + python -m evidence.analyze_client) ==="
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "git=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "base=$BASE"
  echo "payload=$PAYLOAD"
  echo "--- curl ---"
  curl -sS --max-time 120 -w "\nhttp_code=%{http_code}\ntime_total_s=%{time_total}\n" \
    -X POST "$BASE/analyze" \
    -H "Content-Type: application/json" \
    --data @"$PAYLOAD" || true
  echo
  echo "--- python -m evidence.analyze_client ---"
  (cd "$ROOT" && "$PY" -m evidence.analyze_client --url "$BASE/analyze" --payload "$PAYLOAD" --timeout 120) || true
} | tee "$OUT/04_analyze_timed_live.log"

echo "Wrote $OUT/04_analyze_timed_live.log"
