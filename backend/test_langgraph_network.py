"""Smoke test: LangGraph network with mocked feature agents (no external APIs)."""
from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

# Ensure backend is on path when run from repo root
sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("MAPPLS_API_KEY", "test")
os.environ.setdefault("SERP_API_KEY", "test")
os.environ.setdefault("WEATHER_API_KEY", "test")
os.environ.setdefault("GROQ_API_KEY", "test")

from orchestrator.utils.schema import (  # noqa: E402
    AnalyzeRequest,
    GSCPIFeatures,
    NewsFeatures,
    PoliticalFeatures,
    TradeFeatures,
    WeatherFeatures,
)


async def _run():
    from orchestrator.graph.network import build_analysis_graph
    from orchestrator.utils.schema import AnalyzeResponse

    trade = TradeFeatures(inventory_days=28.0, past_delay_days=4.0, edges=[])
    news = NewsFeatures(news_vol_7d=3, neg_tone_frac_3d=0.4, strike_flag_7d=0, sources=[])
    weather = WeatherFeatures(weather_anomaly_7d=0, details={})
    political = PoliticalFeatures(sanction_flag=0, political_risk_score=0.25, notes="ok")
    gscpi = GSCPIFeatures(global_risk=0.3, timestamp="2026-08")

    with (
        patch("orchestrator.graph.nodes.resolve_pair", new=AsyncMock(return_value=((12.9, 77.6), (1.3, 103.8)))),
        patch("orchestrator.graph.nodes.fetch_trade_features", new=AsyncMock(return_value=trade)),
        patch("orchestrator.graph.nodes.analyze_news", new=AsyncMock(return_value=news)),
        patch("orchestrator.graph.nodes.weather_features", new=AsyncMock(return_value=weather)),
        patch("orchestrator.graph.nodes.political_features", new=AsyncMock(return_value=political)),
        patch("orchestrator.graph.nodes.gscpi_features", new=AsyncMock(return_value=gscpi)),
    ):
        graph = build_analysis_graph()
        req = AnalyzeRequest(
            component_type="semiconductor",
            seller_location="Bangalore, India",
            import_location="Singapore",
            seller_name="Demo Fab",
        )
        now = datetime.now(timezone.utc)
        state = await graph.ainvoke(
            {
                "request": req,
                "request_id": "test-langgraph-1",
                "created_at": now.isoformat(),
                "errors": [],
            }
        )

    response = state["response"]
    assert isinstance(response, AnalyzeResponse)
    assert response.request_id == "test-langgraph-1"
    assert response.tgn_result.risk_label in {"Low", "Medium", "High"}
    assert 0.0 <= response.tgn_result.risk_score <= 1.0
    assert "inventory_days" in response.features
    assert len(response.concise) > 0
    print("PASS LangGraph network")
    print(f"  risk_score={response.tgn_result.risk_score:.3f}")
    print(f"  risk_label={response.tgn_result.risk_label}")
    print(f"  features={list(response.features.keys())}")
    print(f"  nodes_ran=geocode+5 agents+normalize+tgn+report")


if __name__ == "__main__":
    asyncio.run(_run())
