# Reproducing paper evidence

Pin the submission to a **tag**, not moving `main`:

```bash
git tag -a v1.0-paper -m "Exact system described in LAMDA paper submission"
git push origin v1.0-paper   # when ready
```

Cite in the paper: *Code and fused corpus at tag `v1.0-paper`.*

## One-shot evidence capture

```bash
cd /path/to/LAMDAAnalytics
backend/.venv/bin/python evidence/capture_evidence.py
```

Writes timestamped artifacts under `evidence/runs/YYYYMMDDTHHMMSSZ/`:
- `manifest.json` — git commit, step timings
- `langgraph_mock_timing.txt` — mocked graph E2E (no external APIs)
- `generate_results_figures.log`
- `bootstrap_ci.json` / `bootstrap_ci.md`
- `figures_manifest.json`

## Six CLI transcripts (text logs, not screenshots)

```bash
# all six → evidence/runs/<utc>/*.log
bash evidence/scripts/run_six.sh
```

| # | File | What it proves |
| --- | --- | --- |
| 01 | `01_langgraph_mock_e2e.log` | Mocked LangGraph E2E, no external APIs |
| 02 | `02_generate_results_figures.log` | Figure generator stdout |
| 03 | `03_fast_endpoints.log` | Timed GET `/model/info`, `/analytics/overview`, `/monitoring/alerts` |
| 04 | `04_analyze_timed.log` | Timed `POST /analyze` (`python -m evidence.analyze_client`) |
| 05 | `05_forced_timeout.log` | `LAMDA_FORCE_TIMEOUT=1` + 1 s cap → mock body |
| 06 | `06_forced_agent_failure.log` | `LAMDA_FORCE_NEWS_FAIL=1` → news zeros, graph still returns |

## Live `/analyze` CLI screenshot (PNG)

Needs uvicorn on `:8007`. Writes a compact transcript and a dark-terminal PNG for the paper:

```bash
bash evidence/scripts/screenshot_analyze.sh
# → evidence/runs/<utc>/analyze_cli_screenshot.{txt,png}
# → Research Paper/figures/analyze_cli_screenshot.png
```

Live `/analyze` with a Groq key (refresh the ~11 s claim; Serp/Weather optional):

```bash
# terminal 1
cd backend && uvicorn main:app --host 127.0.0.1 --port 8007

# terminal 2
bash evidence/scripts/time_analyze.sh
# or:
python -m evidence.analyze_client \
  --url http://127.0.0.1:8007/analyze \
  --payload evidence/payloads/analyze_example.json \
  > evidence/runs/analyze_run_$(date -u +%Y-%m-%d).log 2>&1
```

Forced-timeout / agent-failure against a running server:

```bash
LAMDA_FORCE_TIMEOUT=1 AGENT_TIMEOUT_SECONDS=1 \
  uvicorn main:app --host 127.0.0.1 --port 8007
python -m evidence.analyze_client --url http://127.0.0.1:8007/analyze \
  --payload evidence/payloads/analyze_example.json \
  > evidence/runs/forced_timeout.log 2>&1

LAMDA_FORCE_NEWS_FAIL=1 uvicorn main:app --host 127.0.0.1 --port 8007
python -m evidence.analyze_client --url http://127.0.0.1:8007/analyze \
  --payload evidence/payloads/analyze_example.json \
  > evidence/runs/forced_news_fail.log 2>&1
```

## Bootstrap confidence intervals

```bash
backend/.venv/bin/python evidence/bootstrap_ci.py
```

Output: `evidence/bootstrap_ci.json`, `evidence/bootstrap_ci.md` (5000 resamples, seed 42).

## Publication figures

```bash
cd "Research Paper"
../backend/.venv/bin/python generate_results_figures.py
```

Outputs PDF + 900 DPI PNG under `Research Paper/figures/`. No hardcoded absolute paths.

## Full offline benchmark (optional re-run)

```bash
cd dataset
../backend/.venv/bin/python check_training_env.py --config config.yaml
../backend/.venv/bin/python benchmark_all_models.py
```

Primary results: `dataset/data/four_year_2021_2024/results/model_training/benchmark_comparison/`.

## Live `/analyze` timing (requires keys)

```bash
cd backend
cp .env.example .env   # fill MAPPLS, SERP, WEATHER, GROQ keys
uvicorn main:app --port 8007 &
sleep 2
curl -w "\ntime_total: %{time_total}s\n" -o /tmp/analyze.json -s \
  -X POST http://127.0.0.1:8007/analyze \
  -H "Content-Type: application/json" \
  -d '{"component_type":"Semiconductor","seller_location":"Hsinchu, Taiwan","import_location":"Los Angeles, USA","seller_name":"TSMC"}'
```

Record stdout in `evidence/runs/` for submission. Fast endpoints:

```bash
curl -w "%{time_total}\n" -o /dev/null -s http://127.0.0.1:8007/model/info
curl -w "%{time_total}\n" -o /dev/null -s http://127.0.0.1:8007/analytics/overview
curl -w "%{time_total}\n" -o /dev/null -s http://127.0.0.1:8007/monitoring/alerts
```

## Mocked orchestrator (CI-style, no keys)

```bash
cd backend
PYTHONPATH=. .venv/bin/python test_langgraph_network.py
```

## Audit checklist

See `evidence/CODE_AUDIT.md` for claim-by-claim code traces.
