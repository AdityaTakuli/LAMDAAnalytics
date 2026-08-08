from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..agents.gscpi_agent import gscpi_features
from ..agents.news_agent import analyze_news
from ..agents.normalizer_agent import normalize_all
from ..agents.political_agent import political_features
from ..agents.reporter_agent import concise_from_contrib, comprehensive, label_from_score
from ..agents.trade_agent import fetch_trade_features
from ..agents.weather_agent import weather_features
from ..utils.geocoding import resolve_pair
from ..utils.schema import AnalyzeResponse, TGNResult
from .state import AnalysisState


async def geocode_node(state: AnalysisState) -> dict[str, Any]:
    req = state["request"]
    seller_latlon, importer_latlon = await resolve_pair(req.seller_location, req.import_location)
    return {
        "seller_latlon": seller_latlon,
        "importer_latlon": importer_latlon,
    }


async def trade_node(state: AnalysisState) -> dict[str, Any]:
    req = state["request"]
    trade = await fetch_trade_features(req.component_type, req.seller_location, req.import_location)
    return {"trade": trade}


async def news_node(state: AnalysisState) -> dict[str, Any]:
    req = state["request"]
    news = await analyze_news(
        req.component_type,
        req.seller_location,
        req.import_location,
        req.seller_name,
    )
    return {"news": news}


async def weather_node(state: AnalysisState) -> dict[str, Any]:
    seller_latlon = state.get("seller_latlon")
    if seller_latlon:
        weather = await weather_features(*seller_latlon)
    else:
        weather = await weather_features(0.0, 0.0)
    return {"weather": weather}


async def political_node(state: AnalysisState) -> dict[str, Any]:
    req = state["request"]
    political = await political_features(
        req.component_type,
        req.seller_location,
        req.import_location,
        req.seller_name,
    )
    return {"political": political}


async def gscpi_node(state: AnalysisState) -> dict[str, Any]:
    gscpi = await gscpi_features()
    return {"gscpi": gscpi}


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
