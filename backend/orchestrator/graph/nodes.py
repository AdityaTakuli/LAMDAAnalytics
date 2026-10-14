from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from config.settings import settings

from ..agents.gscpi_agent import gscpi_features
from ..agents.news_agent import analyze_news
from ..agents.normalizer_agent import normalize_all
from ..agents.political_agent import political_features
from ..agents.reporter_agent import concise_from_contrib, comprehensive, label_from_score
from ..agents.trade_agent import fetch_trade_features
from ..agents.weather_agent import weather_features
from ..utils.geocoding import resolve_pair
from ..utils.schema import (
    AnalyzeResponse,
    GSCPIFeatures,
    NewsFeatures,
    PoliticalFeatures,
    TGNResult,
    TradeFeatures,
    WeatherFeatures,
)
from .state import AnalysisState

# Conservative defaults written when an agent fails or exceeds its own timeout.
# They match the in-agent fallbacks, so a degraded field looks the same whether
# the failure happened inside the agent or around it.
_DEFAULTS: dict[str, Callable[[], Any]] = {
    "trade": lambda: TradeFeatures(inventory_days=30.0, past_delay_days=5.0, edges=[]),
    "news": lambda: NewsFeatures(news_vol_7d=0, neg_tone_frac_3d=0.0, strike_flag_7d=0, sources=[]),
    "weather": lambda: WeatherFeatures(weather_anomaly_7d=0, details={}),
    "political": lambda: PoliticalFeatures(sanction_flag=0, political_risk_score=0.3, notes=None),
    "gscpi": lambda: GSCPIFeatures(global_risk=0.2, timestamp=None),
}


async def _guarded(key: str, call: Awaitable[Any]) -> dict[str, Any]:
    """Run one feature agent under its own timeout; never raise into the graph."""
    try:
        value = await asyncio.wait_for(call, timeout=settings.agent_node_timeout_seconds)
        status = "ok"
    except asyncio.TimeoutError:
        value, status = _DEFAULTS[key](), "timeout"
    except Exception as exc:  # noqa: BLE001 - isolation is the point
        value, status = _DEFAULTS[key](), f"error: {type(exc).__name__}"
    return {key: value, "agent_status": {key: status}}


async def geocode_node(state: AnalysisState) -> dict[str, Any]:
    req = state["request"]
    seller_latlon, importer_latlon = await resolve_pair(req.seller_location, req.import_location)
    return {
        "seller_latlon": seller_latlon,
        "importer_latlon": importer_latlon,
    }


async def trade_node(state: AnalysisState) -> dict[str, Any]:
    req = state["request"]
    return await _guarded(
        "trade", fetch_trade_features(req.component_type, req.seller_location, req.import_location)
    )


async def news_node(state: AnalysisState) -> dict[str, Any]:
    req = state["request"]
    return await _guarded(
        "news",
        analyze_news(req.component_type, req.seller_location, req.import_location, req.seller_name),
    )


async def weather_node(state: AnalysisState) -> dict[str, Any]:
    seller_latlon = state.get("seller_latlon") or (0.0, 0.0)
    return await _guarded("weather", weather_features(*seller_latlon))


async def political_node(state: AnalysisState) -> dict[str, Any]:
    req = state["request"]
    return await _guarded(
        "political",
        political_features(req.component_type, req.seller_location, req.import_location, req.seller_name),
    )


async def gscpi_node(state: AnalysisState) -> dict[str, Any]:
    return await _guarded("gscpi", gscpi_features())


async def normalize_node(state: AnalysisState) -> dict[str, Any]:
    created_at = state.get("created_at") or datetime.now(timezone.utc).isoformat()
    normalized = normalize_all(
        created_at,
        state["trade"],
        state["news"],
        state["weather"],
        state["political"],
        state["gscpi"],
    )
    return {
        "normalized": normalized,
        "features": normalized.features,
    }


async def tgn_node(state: AnalysisState) -> dict[str, Any]:
    from models.tgn_model import tgn

    risk_score, contrib = tgn.predict(state["features"])
    label = label_from_score(risk_score)
    tgn_out = TGNResult(
        risk_score=risk_score,
        risk_label=label,
        risk_components=contrib,
    )
    return {"tgn_result": tgn_out}


async def report_node(state: AnalysisState) -> dict[str, Any]:
    tgn_out = state["tgn_result"]
    concise = concise_from_contrib(tgn_out.risk_components, tgn_out.risk_score)
    comp = comprehensive(tgn_out.risk_components)
    created_at = datetime.fromisoformat(state["created_at"])
    response = AnalyzeResponse(
        request_id=state["request_id"],
        created_at=created_at,
        inputs=state["request"],
        features=state["features"],
        tgn_result=tgn_out,
        concise=concise,
        comprehensive=comp,
    )
    return {
        "concise": concise,
        "comprehensive": comp,
        "response": response,
    }
