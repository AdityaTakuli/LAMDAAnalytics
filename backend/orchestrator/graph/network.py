"""LangGraph multi-agent network for supply-chain risk analysis.

Topology
--------
START
  └─► geocode
        ├─► trade ──────┐
        ├─► news ───────┤
        ├─► weather ────┼─► normalize ─► tgn ─► report ─► END
        ├─► political ──┤
        └─► gscpi ──────┘

The five feature agents run in parallel after geocoding; LangGraph
fan-in waits for all of them before normalize runs.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    geocode_node,
    gscpi_node,
    news_node,
    normalize_node,
    political_node,
    report_node,
    tgn_node,
    trade_node,
    weather_node,
)
from .state import AnalysisState

FEATURE_AGENTS = ("trade", "news", "weather", "political", "gscpi")


def build_analysis_graph():
    graph = StateGraph(AnalysisState)

    graph.add_node("geocode", geocode_node)
    graph.add_node("trade", trade_node)
    graph.add_node("news", news_node)
    graph.add_node("weather", weather_node)
    graph.add_node("political", political_node)
    graph.add_node("gscpi", gscpi_node)
    graph.add_node("normalize", normalize_node)
    graph.add_node("tgn", tgn_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "geocode")
    for agent in FEATURE_AGENTS:
        graph.add_edge("geocode", agent)
        graph.add_edge(agent, "normalize")

    graph.add_edge("normalize", "tgn")
    graph.add_edge("tgn", "report")
    graph.add_edge("report", END)

    return graph.compile()


# Compiled singleton used by the orchestrator
analysis_graph = build_analysis_graph()
