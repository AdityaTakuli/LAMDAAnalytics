# AI Agent Pipeline Documentation

## Overview

LAMDA uses a **LangGraph `StateGraph`** multi-agent network. Specialized agent modules are wired as graph nodes with parallel fan-out for feature extraction, then sequential normalize → TGN → report.

## Architecture

```
Input Request
  → orchestrator.run_analysis()
    → LangGraph analysis_graph.ainvoke()
      → geocode
      → trade / news / weather / political / gscpi  (parallel)
      → normalize → tgn → report
  → AnalyzeResponse
```

### Key packages

| Path | Role |
|------|------|
| `orchestrator/orchestrator.py` | Entry: builds initial state, `ainvoke`, timeout |
| `orchestrator/graph/network.py` | Compiles the `StateGraph` |
| `orchestrator/graph/nodes.py` | Async node wrappers |
| `orchestrator/graph/state.py` | Shared `AnalysisState` |
| `orchestrator/agents/*` | Agent implementations |

## AI Agents (nodes)

### 1. Trade Agent (`trade_agent.py`)
**Purpose**: Trade flows and inventory patterns  
**Sources**: LLM  
**Output**: `TradeFeatures` — `inventory_days`, `past_delay_days`, edges

### 2. News Agent (`news_agent.py`)
**Purpose**: Disruption monitoring  
**Sources**: SERP → scrape → LLM  
**Output**: `NewsFeatures` — `news_vol_7d`, `neg_tone_frac_3d`, `strike_flag_7d`

### 3. Weather Agent (`weather_agent.py`)
**Purpose**: Logistics weather anomalies  
**Sources**: OpenWeather / WeatherAPI  
**Output**: `WeatherFeatures` — `weather_anomaly_7d`

### 4. Political Agent (`political_agent.py`)
**Purpose**: Sanctions / geopolitical risk  
**Sources**: LLM  
**Output**: `PoliticalFeatures`

### 5. GSCPI Agent (`gscpi_agent.py`)
**Purpose**: Global supply-chain pressure  
**Sources**: LLM (NY Fed GSCPI style estimate)  
**Output**: `GSCPIFeatures`

### 6. Normalizer (`normalizer_agent.py`)
Assembles raw features → z-score → sigmoid → `[0,1]`.

### 7. TGN (`tgn_model.py`)
Risk score + contributions. Loads `tgn_model.pth` when available; otherwise weighted blend. Torch is optional for the fallback path.

### 8. Reporter (`reporter_agent.py`)
High/Medium/Low labels, factor percentages, mitigation strategies.

## LangGraph Orchestration

```python
# backend/orchestrator/graph/network.py (conceptual)

graph.add_edge(START, "geocode")
for agent in ("trade", "news", "weather", "political", "gscpi"):
    graph.add_edge("geocode", agent)
    graph.add_edge(agent, "normalize")
graph.add_edge("normalize", "tgn")
graph.add_edge("tgn", "report")
graph.add_edge("report", END)
```

```python
# backend/orchestrator/orchestrator.py

final_state = await asyncio.wait_for(
    analysis_graph.ainvoke(initial_state),
    timeout=settings.agent_timeout_seconds,
)
return final_state["response"]
```

Parallelism is graph-native (multiple outgoing edges), not a manual `asyncio.gather` in the orchestrator.

## API Integration

1. **LLM (Groq)** — trade, news sentiment, political, GSCPI (`GROQ_API_KEY`)
2. **SERP API** — news search (`SERP_API_KEY`)
3. **Weather APIs** — forecast / anomaly (`WEATHER_API_KEY`, `WEATHER_PROVIDER`)
4. **Mappls** — geocoding (`MAPPLS_API_KEY`, optional client id/secret)

## Error Handling

- API failures: agent-level defaults / empty features
- Model failures: weighted risk scoring
- HTTP: retries via tenacity / client timeouts
- FastAPI `/analyze`: mock `AnalyzeResponse` if the graph raises (demo-friendly)

### Caching

- News & weather: ~15-minute TTL
- Geocoding helpers as available

## Configuration

```bash
# Required
MAPPLS_API_KEY=...
SERP_API_KEY=...
WEATHER_API_KEY=...
GROQ_API_KEY=...

# Optional
WEATHER_PROVIDER=openweather
HTTP_TIMEOUT_SECONDS=30
AGENT_TIMEOUT_SECONDS=40
SCORING_STATE_PATH=./data/scoring_state.json
LOG_LEVEL=INFO
ENABLE_GDELT=false
ENABLE_COMTRADE=false
ALLOW_GEMINI_FALLBACK=true
```

See `backend/.env.example`.

## Testing

| Script | Purpose |
|--------|---------|
| `backend/test_langgraph_network.py` | Mocked LangGraph E2E (no APIs) |
| `test_full_orchestrator.py` | Live full graph with real APIs |
| `test_simple_orchestrator.py` | Live simplified demo |
| `test_orchestrator.py` | Basic orchestrator call |
| `test_google_api.py` / `test_api_keys.py` | API connectivity |

```bash
cd backend
PYTHONPATH=. python test_langgraph_network.py
```

## Output Format

Same `AnalyzeResponse` schema as before (`features`, `tgn_result`, `concise`, `comprehensive`). The transport is unchanged; only the internal orchestration is LangGraph.

## Extending

1. Implement `orchestrator/agents/new_agent.py`
2. Add node in `graph/nodes.py`
3. Register edges in `graph/network.py` (`geocode → new → normalize`)
4. Extend `AnalysisState`, normalizer, and scoring weights

## Deployment Notes

- Install `langgraph` and `langchain-core` (listed in `backend/requirements.txt`)
- Prefer a venv: `python3 -m venv .venv && pip install -r requirements.txt`
- Docker image should install the same requirements; expose port `8007`

## Future Enhancements

1. Checkpointing / durable LangGraph runs
2. Conditional edges (skip agents by component type)
3. Streaming node progress to the UI via WebSocket
4. Redis-backed cache and scoring state
5. Plugin registry for third-party agents

---

For implementation detail, see `ORCHESTRATOR_README.md`, `AI_AGENT_PIPELINE.md`, and `backend/orchestrator/graph/`.
