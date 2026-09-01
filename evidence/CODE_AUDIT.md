# Code audit vs paper claims

Generated: 2026-09-01. Re-run: `python evidence/capture_evidence.py`.

Git commit at capture: see `evidence/runs/*/manifest.json`.

## Summary

| Claim | Paper | Code (verified) | Status |
| --- | --- | --- | --- |
| Live `TGNWrapper.predict()` | Always linear blend | `weighted_risk()` only; no torch forward | **MATCH** |
| Whole-graph timeout | 90 s on full `ainvoke` | `orchestrator.py`: `asyncio.wait_for(..., settings.agent_timeout_seconds)` default **90** | **MATCH** |
| Weather cache unused | Defined, not called | `weather_cache` in `cache.py`; `weather_agent.py` never imports it | **MATCH** |
| News cache | 15 min TTL | `ttl=900` in `cache.py`, used in `news_agent.py` | **MATCH** |
| SerpAPI retries | 3× exponential backoff | `@retry(stop_after_attempt(3), wait=wait_exponential(...))` on `serp_search` | **MATCH** |
| Geocode fallback order | OAuth2 → REST → table | `api_clients.geocode()` order matches | **MATCH** |
| Geocode table size | ~40 cities | **46** entries in `_KNOWN_LOCATIONS` | **DRIFT** — update paper to 46 |
| Political agent excluded | Not in seven-feature vector | `assemble_raw()` ignores `pol` | **MATCH** |
| Welford JSON persistence | Survives restarts | `scoring_state.json` load/save in `scoring.py` | **MATCH** |
| Route graph hubs | 25 | **25** in `SupplyChainMap.jsx` | **MATCH** |
| Route graph edges | 44 undirected | **47** undirected in `SupplyChainMap.jsx` | **DRIFT** — update paper to 47 |
| `/analyze` latency ~11 s | Table + §eval | **Measured 2026-09-01**: curl **10.6 s**, repeat **7.2 s**, not mock (`evidence/runs/20260901T145638Z/04_analyze_timed_live.log`). Groq gpt-oss-120b; Serp/Weather unset. Mock LangGraph E2E **~2.4 s** | **MATCH** (not Serp-backed News) |
| Comtrade rows | 31,305 | **31,305** (`nodes_monthly` panel uses fused subset) | **MATCH** |
| GSCPI months | 48 | **48** rows in `gscpi_monthly.csv` | **MATCH** |
| Weather daily rows | ~29k | **29,220** | **MATCH** |
| Strike rate (fused nodes) | ~73% | **72.7%** | **MATCH** |
| Table reg/clf test metrics | Tables XI/XII | Match `comparison.csv` post-fix | **MATCH** |
| Bootstrap CIs | Not in paper yet | `evidence/bootstrap_ci.json` | **ADD TO PAPER** |

## File references

```
backend/models/tgn_model.py          → predict() → weighted_risk only
backend/orchestrator/orchestrator.py → asyncio.wait_for(ainvoke, 90s)
backend/config/settings.py           → agent_timeout_seconds=90
backend/orchestrator/agents/weather_agent.py → no cache import
backend/orchestrator/agents/news_agent.py    → news_cache 900s
backend/orchestrator/utils/api_clients.py    → serp retry, geocode chain
backend/orchestrator/agents/normalizer_agent.py → political excluded
backend/orchestrator/utils/scoring.py        → Welford JSON
project2/src/SupplyChainMap.jsx      → 25 cities, 47 edges
```

## Recommended paper fixes (applied in `paper.tex` where noted)

1. 44 → **47** undirected hardship edges.
2. ~40 → **46** geocode fallback entries.
3. Add bootstrap CI table/footnote from `evidence/bootstrap_ci.md`.
4. Promote 2023 validation R² = 0.054 in regression subsection.
5. Data & code availability → tagged commit `v1.0-paper` (create before submission).
6. Live latency: 10.6 s / 7.2 s from `evidence/runs/20260901T145638Z/`; rerun with `evidence/scripts/time_analyze.sh`.

## Live checks (CLI transcripts)

See `evidence/runs/*/01_*.log` … `06_*.log` after `bash evidence/scripts/run_six.sh`.

| Check | How |
| --- | --- |
| Real `/analyze` with keys | `bash evidence/scripts/time_analyze.sh` (uvicorn + `.env`) |
| Forced timeout → mock body | `LAMDA_FORCE_TIMEOUT=1 AGENT_TIMEOUT_SECONDS=1` then `python -m evidence.analyze_client` |
| Forced news failure → zeros | `LAMDA_FORCE_NEWS_FAIL=1` (graph still returns) |
