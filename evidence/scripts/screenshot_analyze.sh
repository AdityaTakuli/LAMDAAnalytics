#!/usr/bin/env bash
# Live POST /analyze ? compact CLI transcript + terminal-style PNG for the paper.
# Needs uvicorn on LAMDA_BASE_URL (default http://127.0.0.1:8007).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${LAMDA_EVIDENCE_DIR:-$ROOT/evidence/runs/$STAMP}"
mkdir -p "$OUT"
BASE="${LAMDA_BASE_URL:-http://127.0.0.1:8007}"
PAYLOAD="${1:-$ROOT/evidence/payloads/analyze_example.json}"
PY="${ROOT}/backend/.venv/bin/python"
[[ -x "$PY" ]] || PY="$(command -v python3)"
FIG_DIR="${LAMDA_FIG_DIR:-$ROOT/Research Paper/figures}"
TXT="$OUT/analyze_cli_screenshot.txt"
PNG="$OUT/analyze_cli_screenshot.png"
FIG_PNG="$FIG_DIR/analyze_cli_screenshot.png"

if ! curl -sf --max-time 3 "$BASE/model/info" >/dev/null; then
  echo "ERROR: nothing listening at $BASE — start uvicorn first:" >&2
  echo "  cd $ROOT/backend && source .venv/bin/activate && uvicorn main:app --host 127.0.0.1 --port 8007" >&2
  exit 1
fi

BODY="$OUT/analyze_cli_raw.json"
META="$OUT/analyze_cli_meta.txt"
curl -sS --max-time 120 -D "$META" -o "$BODY" \
  -w "time_total_s=%{time_total}\nhttp_code=%{http_code}\n" \
  -X POST "$BASE/analyze" \
  -H "Content-Type: application/json" \
  --data @"$PAYLOAD" > "$OUT/analyze_cli_curl_timing.txt"

HTTP="$(awk -F= '/^http_code=/{print $2}' "$OUT/analyze_cli_curl_timing.txt")"
SECS="$(awk -F= '/^time_total_s=/{print $2}' "$OUT/analyze_cli_curl_timing.txt")"
UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOST="$(hostname -s 2>/dev/null || echo localhost)"
CWD_SHORT="~/Desktop/ARiES/LAMDAAnalytics"

"$PY" - "$BODY" "$PAYLOAD" "$BASE" "$HTTP" "$SECS" "$UTC" "$HOST" "$CWD_SHORT" > "$TXT" <<'PY'
import json, sys
from pathlib import Path

body_path, payload_path, base, http, secs, utc, host, cwd = sys.argv[1:9]
raw = json.loads(Path(body_path).read_text(encoding="utf-8"))
req = json.loads(Path(payload_path).read_text(encoding="utf-8"))
feats = raw.get("features") or {}
tgn = raw.get("tgn_result") or {}
comp = (tgn.get("risk_components") or {})
inp = raw.get("inputs") or req

mock = (
    feats.get("inventory_days") == 0.65
    and feats.get("news_vol_7d") == 0.78
    and tgn.get("risk_score") == 0.45
)

def fnum(x, n=3):
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return str(x)

print(f"{host}:{cwd}$ python -m evidence.analyze_client \\")
print(f"  --url {base}/analyze \\")
print("  --payload evidence/payloads/analyze_example.json")
print(f"utc={utc}")
print(f"url={base}/analyze")
print("--- request ---")
print(json.dumps(req, indent=2))
print("--- response ---")
print(f"http_code={http}")
print(f"time_total_s={float(secs):.3f}")
print(
    "WARNING: body matches hardcoded mock fallback"
    if mock
    else "INFO: body does not match hardcoded mock fallback"
)
print()
print(f"lane  {inp.get('component_type')}  {inp.get('seller_name')}")
print(f"      {inp.get('seller_location')}  ->  {inp.get('import_location')}")
print(f"id    {raw.get('request_id', '')}")
print()
print("features")
for k in (
    "inventory_days",
    "past_delay_days",
    "news_vol_7d",
    "neg_tone_frac_3d",
    "strike_flag_7d",
    "weather_anomaly_7d",
    "global_risk",
):
    print(f"  {k:<22} {fnum(feats.get(k), 4)}")
print()
print(f"tgn_result  risk_score={fnum(tgn.get('risk_score'), 3)}  {tgn.get('risk_label', '')}")
for k, v in sorted(comp.items(), key=lambda kv: -float(kv[1] or 0)):
    print(f"  {k:<22} {fnum(v, 3)}")
PY

"$PY" "$ROOT/evidence/scripts/render_terminal_png.py" "$TXT" \
  -o "$PNG" \
  --title "python -m evidence.analyze_client" \
  --wrap 78

mkdir -p "$FIG_DIR"
cp -f "$PNG" "$FIG_PNG"

echo
echo "======== CLI (screenshot this if you want a live capture) ========"
cat "$TXT"
echo "================================================================="
echo
echo "Wrote $TXT"
echo "Wrote $PNG"
echo "Copied $FIG_PNG"
echo "Re-run: bash evidence/scripts/screenshot_analyze.sh"
