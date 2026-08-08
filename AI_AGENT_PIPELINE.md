# AI Agent Pipeline — LangGraph Network

This document describes the LangGraph multi-agent network that fetches real data, extracts features, scores risk with TGN, and produces reports.

## Pipeline Flow

```
Input Request
    → LangGraph StateGraph
        → geocode
        → [trade | news | weather | political | gscpi]  (parallel)
        → normalize → tgn → report
    → AnalyzeResponse
```

## Topology

```
START → geocode
          ├─ trade ──────┐
          ├─ news ───────┤
          ├─ weather ────┼→ normalize → tgn → report → END
          ├─ political ──┤
          └─ gscpi ──────┘
```

Defined in `backend/orchestrator/graph/network.py`.  
Invoked from `backend/orchestrator/orchestrator.py` via `analysis_graph.ainvoke(...)`.

## Step-by-Step Nodes

### 1. Geocode (`graph/nodes.py` → `utils/geocoding.py`)
Resolves seller and import locations to coordinates for weather.

### 2. Trade Agent (`agents/trade_agent.py`)
- LLM estimates inventory days, past delays, and trade edges
- Output: `TradeFeatures`

### 3. News Agent (`agents/news_agent.py`)
- SERP search → scrape pages → LLM sentiment / disruption features
- Output: `NewsFeatures` (`news_vol_7d`, `neg_tone_frac_3d`, `strike_flag_7d`)

### 4. Weather Agent (`agents/weather_agent.py`)
- Weather API forecast + statistical anomaly detection
- Output: `WeatherFeatures` (`weather_anomaly_7d`)

### 5. Political Agent (`agents/political_agent.py`)
- LLM sanctions / geopolitical risk
- Output: `PoliticalFeatures`

### 6. GSCPI Agent (`agents/gscpi_agent.py`)
- LLM estimate of NY Fed Global Supply Chain Pressure Index
- Output: `GSCPIFeatures`

### 7. Normalizer (`agents/normalizer_agent.py`)
- Assembles raw features, z-score + sigmoid to [0,1]
- Output: `NormalizedFeatureVector`

### 8. TGN (`models/tgn_model.py`)
- Loads `tgn_model.pth` when present; otherwise weighted blend
- Output: risk score + per-feature contributions

### 9. Reporter (`agents/reporter_agent.py`)
- Labels, percentages, impact text, mitigation strategies
- Output: concise + comprehensive reports → `AnalyzeResponse`

## LangGraph Orchestration

```python
from orchestrator.graph import analysis_graph

final_state = await analysis_graph.ainvoke({
    "request": analyze_request,
    "request_id": request_id,
    "created_at": now.isoformat(),
    "errors": [],
})
response = final_state["response"]
```

Parallelism comes from multiple edges out of `geocode`. Fan-in at `normalize` waits for all five feature nodes.

## Running

### Prerequisites

```bash
cd backend
pip install -r requirements.txt   # includes langgraph, langchain-core
cp .env.example .env              # MAPPLS / SERP / WEATHER / GROQ keys
```

### Tests

```bash
# Mocked LangGraph smoke test (recommended first)
PYTHONPATH=. python test_langgraph_network.py

# Live end-to-end (needs valid API keys)
python ../test_full_orchestrator.py
```

### API

```bash
curl -X POST http://127.0.0.1:8007/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "component_type": "Semiconductor",
    "seller_location": "Hsinchu, Taiwan",
    "import_location": "Los Angeles, USA",
    "seller_name": "TSMC"
  }'
```

## Expected Output

```json
{
  "request_id": "uuid",
  "created_at": "2026-08-09T00:00:00Z",
  "inputs": { },
  "features": {
    "inventory_days": 0.65,
    "past_delay_days": 0.42,
    "news_vol_7d": 0.78,
    "neg_tone_frac_3d": 0.23,
    "strike_flag_7d": 0.0,
    "weather_anomaly_7d": 0.15,
    "global_risk": 0.31
  },
  "tgn_result": {
    "risk_score": 0.45,
    "risk_label": "Medium",
    "risk_components": { }
  },
  "concise": [ ],
  "comprehensive": {
    "risk_distribution": [ ],
    "mitigation_strategies": { }
  }
}
```

## Performance

- Feature agents run concurrently inside LangGraph
- Graph-level timeout: `AGENT_TIMEOUT_SECONDS` (default 40s)
- News/weather TTL caching
- Graceful degradation when APIs fail

## Extending the Graph

1. Add agent function under `orchestrator/agents/`
2. Wrap it in `orchestrator/graph/nodes.py`
3. Wire edges in `orchestrator/graph/network.py`
4. Update `AnalysisState` and normalizer/scoring as needed
