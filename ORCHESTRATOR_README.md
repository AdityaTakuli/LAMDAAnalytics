# Supply Chain Risk Prediction Orchestrator

Multi-agent supply chain risk analysis powered by a **LangGraph `StateGraph`**, external APIs, and TGN inference.

## Architecture Overview

- **Framework**: [LangGraph](https://langchain-ai.github.io/langgraph/) (`StateGraph` + async nodes)
- **Agent nodes**: Geocode, Trade, News, Weather, Political, GSCPI, Normalizer, TGN, Reporter
- **Fan-out / fan-in**: five feature agents run in parallel after geocoding
- **TGN wrapper**: loads `tgn_model.pth` when available; otherwise weighted scoring
- **External APIs**: LLM (Groq), SERP, Weather, Mappls geocoding

### Graph Topology

```
START → geocode
          ├─ trade ──────┐
          ├─ news ───────┤
          ├─ weather ────┼→ normalize → tgn → report → END
          ├─ political ──┤
          └─ gscpi ──────┘
```

Entry point: `orchestrator.orchestrator.run_analysis()` → `analysis_graph.ainvoke(state)`.

## Directory Structure

```
backend/
├── main.py
├── requirements.txt             # langgraph, langchain-core, ...
├── .env.example
├── test_langgraph_network.py    # mocked LangGraph smoke test
├── config/
│   └── settings.py
├── models/
│   └── tgn_model.py
├── orchestrator/
│   ├── orchestrator.py          # thin wrapper around LangGraph
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py             # AnalysisState (TypedDict)
│   │   ├── nodes.py             # async node wrappers
│   │   └── network.py           # build_analysis_graph()
│   ├── agents/
│   │   ├── trade_agent.py
│   │   ├── news_agent.py
│   │   ├── weather_agent.py
│   │   ├── political_agent.py
│   │   ├── gscpi_agent.py
│   │   ├── normalizer_agent.py
│   │   └── reporter_agent.py
│   └── utils/
│       ├── api_clients.py
│       ├── geocoding.py
│       ├── scoring.py
│       ├── cache.py
│       ├── schema.py
│       └── timeutils.py
└── data/
    └── scoring_state.json
```

## Quick Start

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill MAPPLS_API_KEY, SERP_API_KEY, WEATHER_API_KEY, GROQ_API_KEY

uvicorn main:app --host 0.0.0.0 --port 8007 --reload
```

### Smoke test (no external APIs)

```bash
cd backend
PYTHONPATH=. .venv/bin/python test_langgraph_network.py
```

### Live test

```bash
python test_orchestrator.py
# or
curl -X POST http://localhost:8007/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "component_type": "Semiconductor",
    "seller_location": "Hsinchu, Taiwan",
    "import_location": "Los Angeles, USA",
    "seller_name": "TSMC"
  }'
```

## API Endpoints

### POST /analyze

Runs the LangGraph network end-to-end.

**Request:**
```json
{
  "component_type": "Semiconductor",
  "seller_location": "Hsinchu, Taiwan",
  "import_location": "Los Angeles, USA",
  "seller_name": "TSMC",
  "additional_factors": {}
}
```

**Response:** `AnalyzeResponse` with `features`, `tgn_result`, `concise`, and `comprehensive` reports.

Also available: `GET /model/info`, `GET /analytics/overview`, `GET /monitoring/alerts`.

## Agent / Node Details

| Node | Role | Sources |
|------|------|---------|
| Geocode | Lat/lon for weather | Mappls |
| Trade | Inventory / delay / edges | LLM |
| News | Disruption volume & tone | SERP → scrape → LLM |
| Weather | Anomaly flag | OpenWeather / WeatherAPI |
| Political | Sanctions & political risk | LLM |
| GSCPI | Global pressure index | LLM |
| Normalize | Z-score → [0,1] features | Local |
| TGN | Risk score + contributions | PyTorch or weighted fallback |
| Report | Concise + mitigation text | Local |

## LangGraph State

Shared state lives in `orchestrator/graph/state.py` (`AnalysisState`):

- Inputs: `request`, `request_id`, `created_at`
- Parallel outputs: `trade`, `news`, `weather`, `political`, `gscpi`
- Downstream: `features`, `tgn_result`, `concise`, `comprehensive`, `response`

Each parallel node writes a distinct key so fan-in does not need custom reducers.

## Performance

- Parallel feature agents via LangGraph edges (not manual `asyncio.gather`)
- Whole-graph timeout: `AGENT_TIMEOUT_SECONDS`
- 15-minute TTL cache for news/weather
- Retry / backoff in HTTP clients

## Extensibility

### Add a new feature agent

1. Implement logic in `orchestrator/agents/your_agent.py`
2. Add a node wrapper in `orchestrator/graph/nodes.py`
3. Register the node and edges in `orchestrator/graph/network.py` (geocode → agent → normalize)
4. Extend `AnalysisState` and `normalize_all` / `scoring.py` weights

### Swap models

Replace `TGNWrapper.predict()` in `models/tgn_model.py`; keep the `(score, contributions)` contract.

## Troubleshooting

1. **Missing keys** — set all required vars in `backend/.env`
2. **Import errors** — run with `PYTHONPATH=backend` or from `backend/`
3. **Timeouts** — raise `AGENT_TIMEOUT_SECONDS`
4. **LangGraph missing** — `pip install langgraph langchain-core`
5. **Torch optional** — TGN falls back to weighted scoring if torch / `.pth` unavailable

---

Orchestration is LangGraph-first; agent modules remain plain async functions reused as graph nodes.
