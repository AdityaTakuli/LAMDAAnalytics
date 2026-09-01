#!/usr/bin/env bash
# Capture all six CLI transcripts into evidence/runs/<utc>/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${ROOT}/backend/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"
exec "$PY" "$ROOT/evidence/scripts/capture_cli_transcripts.py"
