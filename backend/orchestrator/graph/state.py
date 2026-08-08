from __future__ import annotations

from typing import Dict, List, Optional, Tuple, TypedDict

from ..utils.schema import (
    AnalyzeRequest,
    AnalyzeResponse,
    ComprehensiveReport,
    GSCPIFeatures,
    NewsFeatures,
    NormalizedFeatureVector,
    PoliticalFeatures,
    RiskFactorReport,
    TGNResult,
    TradeFeatures,
    WeatherFeatures,
)


class AnalysisState(TypedDict, total=False):
    """Shared LangGraph state for the multi-agent risk analysis network."""

    # Inputs / metadata
    request: AnalyzeRequest
    request_id: str
    created_at: str  # ISO timestamp

    # Geocoding
    seller_latlon: Optional[Tuple[float, float]]
    importer_latlon: Optional[Tuple[float, float]]

    # Parallel agent outputs (each node writes a distinct key)
    trade: TradeFeatures
    news: NewsFeatures
    weather: WeatherFeatures
    political: PoliticalFeatures
    gscpi: GSCPIFeatures

    # Downstream pipeline
    normalized: NormalizedFeatureVector
    features: Dict[str, float]
    tgn_result: TGNResult
    concise: List[RiskFactorReport]
    comprehensive: ComprehensiveReport
    response: AnalyzeResponse

    # Soft error channel (optional)
    errors: List[str]
