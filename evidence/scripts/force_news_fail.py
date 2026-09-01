#!/usr/bin/env python3
"""Prove a single agent can fail closed while the LangGraph still returns a body."""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

os.environ.setdefault("MAPPLS_API_KEY", "test")
os.environ.setdefault("SERP_API_KEY", "test")
os.environ.setdefault("WEATHER_API_KEY", "test")
os.environ.setdefault("GROQ_API_KEY", "test")
os.environ["LAMDA_FORCE_NEWS_FAIL"] = "1"

from orchestrator.utils.schema import (  # noqa: E402
    AnalyzeRequest,
    GSCPIFeatures,
    PoliticalFeatures,
    TradeFeatures,
    WeatherFeatures,
)


async def _run() -> None:
    from orchestrator.agents.news_agent import analyze_news
    from orchestrator.graph.network import build_analysis_graph
    from orchestrator.utils.schema import AnalyzeResponse

    news = await analyze_news("Semiconductor", "Hsinchu, Taiwan", "Los Angeles, USA", "TSMC")
    print(f"news_forced={news.model_dump()}")
    assert news.news_vol_7d == 0
    assert "LAMDA_FORCE_NEWS_FAIL" in news.sources

    trade = TradeFeatures(inventory_days=28.0, past_delay_days=4.0, edges=[])
    weather = WeatherFeatures(weather_anomaly_7d=0, details={})
    political = PoliticalFeatures(sanction_flag=0, political_risk_score=0.25, notes="ok")
    gscpi = GSCPIFeatures(global_risk=0.3, timestamp="2026-09")

    with (
        patch("orchestrator.graph.nodes.resolve_pair", new=AsyncMock(return_value=((24.8, 120.9), (34.0, -118.2)))),
        patch("orchestrator.graph.nodes.fetch_trade_features", new=AsyncMock(return_value=trade)),
        patch("orchestrator.graph.nodes.weather_features", new=AsyncMock(return_value=weather)),
        patch("orchestrator.graph.nodes.political_features", new=AsyncMock(return_value=political)),
        patch("orchestrator.graph.nodes.gscpi_features", new=AsyncMock(return_value=gscpi)),
    ):
        graph = build_analysis_graph()
        req = AnalyzeRequest(
            component_type="Semiconductor",
            seller_location="Hsinchu, Taiwan",
            import_location="Los Angeles, USA",
            seller_name="TSMC",
        )
        now = datetime.now(timezone.utc)
        state = await graph.ainvoke(
            {
                "request": req,
                "request_id": "force-news-fail-1",
                "created_at": now.isoformat(),
                "errors": [],
            }
        )

    response = state["response"]
    assert isinstance(response, AnalyzeResponse)
    print(f"risk_score={response.tgn_result.risk_score:.3f}")
    print(f"risk_label={response.tgn_result.risk_label}")
    print(f"news_vol_7d_normalized={response.features.get('news_vol_7d')}")
    print("response_json=")
    print(response.model_dump_json(indent=2))
    print("PASS forced news failure: graph returned a valid AnalyzeResponse")


if __name__ == "__main__":
    asyncio.run(_run())
