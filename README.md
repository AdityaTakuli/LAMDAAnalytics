# LAMDA Supply Chain Risk Analysis System

AI-powered supply chain risk analysis with a **LangGraph multi-agent network**, PyTorch TGN risk scoring, real-time external APIs, and a React dashboard.

## Features

* **LangGraph Agent Network**: Fan-out/fan-in graph of specialized agent nodes
* **Real-time External APIs**: LLM (Groq/Gemini), SERP, Weather, Mappls geocoding
* **TGN Model Integration**: PyTorch risk prediction with weighted-score fallback
* **Parallel Feature Agents**: Trade, News, Weather, Political, and GSCPI run concurrently
* **Interactive Dashboard**: React UI with live risk scoring and route context

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  React Frontend │◄──►│ FastAPI Backend │◄──►│  PyTorch TGN    │
│   (Port 5175)   │    │   (Port 8007)   │    │ (tgn_model.pth) │
└─────────────────┘    └────────┬────────┘    └─────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  LangGraph Network    │
                    │  (StateGraph)         │
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
        ┌──────────┐     ┌──────────┐     ┌──────────┐
        │   LLM    │     │ SERP API │     │ Weather  │
        └──────────┘     └──────────┘     └──────────┘
```

### LangGraph Topology

```
START → geocode
          ├─ trade ──────┐
          ├─ news ───────┤
          ├─ weather ────┼→ normalize → tgn → report → END
          ├─ political ──┤
          └─ gscpi ──────┘
```

Feature agents fan out in parallel after geocoding. LangGraph joins them before normalize → TGN → report.

## Agent Network Nodes

1. **Geocode** — resolve seller/import coordinates
2. **Trade** — LLM trade-flow / inventory estimates
3. **News** — SERP search → scrape → LLM sentiment features
4. **Weather** — weather APIs → anomaly detection
5. **Political** — LLM geopolitical / sanctions risk
6. **GSCPI** — LLM global supply-chain pressure index
7. **Normalizer** — assemble + scale feature vector
8. **TGN** — trained risk prediction (or weighted fallback)
9. **Reporter** — concise + comprehensive risk reports

## Prerequisites

* Python 3.12+
* Node.js 16+
* API keys (see `backend/.env.example`):
  - `MAPPLS_API_KEY` (geocoding)
  - `SERP_API_KEY`
  - `WEATHER_API_KEY`
  - `GROQ_API_KEY` (LLM)

## Installation & Setup

### 1. Clone

```bash
git clone https://github.com/iareARiES/LAMDAAnalytics.git
cd LAMDAAnalytics
```

### 2. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your API keys

python -m uvicorn main:app --host 127.0.0.1 --port 8007
```

Backend: `http://127.0.0.1:8007`  
OpenAPI docs: `http://127.0.0.1:8007/docs`

### 3. Frontend

```bash
cd project2
npm install
npm run dev
```

Frontend: `http://localhost:5175`

### 4. Quick Start Helper

```bash
# Backend
python start_backend.py
```

## API Endpoints

* `POST /analyze` — full LangGraph analysis
* `GET /model/info` — TGN model info
* `GET /health` — health check
* `GET /monitoring/alerts` — alerts
* `GET /analytics/overview` — dashboard overview

## Usage

1. Open `http://localhost:5175/dashboard`
2. Set component type, seller location, import location, seller name
3. Run analysis and review risk score, factors, and mitigation strategies

## Input Features (normalized 0–1)

* `inventory_days`, `past_delay_days`
* `news_vol_7d`, `neg_tone_frac_3d`, `strike_flag_7d`
* `weather_anomaly_7d`
* `global_risk`

## Testing

```bash
# LangGraph network smoke test (mocked agents, no external APIs)
cd backend
PYTHONPATH=. .venv/bin/python test_langgraph_network.py

# Full live pipeline (requires valid .env keys)
cd ..
python test_full_orchestrator.py

# API connectivity
python test_google_api.py
python test_simple_orchestrator.py
```

### Curl

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

## Project Structure

```
LAMDAAnalytics/
├── backend/
│   ├── main.py
│   ├── requirements.txt          # includes langgraph, langchain-core
│   ├── .env.example
│   ├── test_langgraph_network.py # LangGraph smoke test
│   ├── config/settings.py
│   ├── models/tgn_model.py
│   └── orchestrator/
│       ├── orchestrator.py       # invokes LangGraph ainvoke()
│       ├── graph/                # LangGraph StateGraph
│       │   ├── state.py          # AnalysisState
│       │   ├── nodes.py          # node wrappers
│       │   └── network.py        # graph compile + topology
│       ├── agents/               # agent implementations
│       └── utils/
├── project2/                     # React frontend
├── ORCHESTRATOR_README.md
├── AI_AGENT_PIPELINE.md
├── AI_AGENT_PIPELINE_DOCUMENTATION.md
├── API_DOCUMENTATION.md
└── DEPLOYMENT_GUIDE.md
```

## Performance

* Feature agents run in parallel via LangGraph fan-out
* Graph timeout controlled by `AGENT_TIMEOUT_SECONDS`
* News/weather caching with TTL
* Graceful mock fallback in API if external calls fail

## Troubleshooting

* Install deps: `pip install -r backend/requirements.txt` (needs `langgraph`)
* Ensure `backend/.env` has required keys from `.env.example`
* TGN works without torch via weighted fallback; torch is optional for `.pth` load
* Ports: backend `8007`, frontend `5175`

## License

Part of the LAMDA Analytics system for supply chain risk management.

## Team

* **Devansh Behl**: Full Stack Development
* **Mayan Sharma**: AI/ML Engineering
* **Aditya Takuli**: Data Engineering & Analytics
* **Lay Gupta**: Product & Business Model

## Support

* API docs: `http://127.0.0.1:8007/docs`
* GitHub Issues: [Create an issue](https://github.com/iareARiES/LAMDAAnalytics/issues)

---

**Status**: Operational — LangGraph multi-agent network  
**Version**: 3.0 — LangGraph Agent Network
