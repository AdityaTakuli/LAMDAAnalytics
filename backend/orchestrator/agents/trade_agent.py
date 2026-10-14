from ..utils.schema import TradeFeatures, TradeEdge
from ..utils.api_clients import get_gemini
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config.settings import settings

async def fetch_trade_features(component_type: str, seller_loc: str, import_loc: str) -> TradeFeatures:
    """
    Without a direct Comtrade API key, use Gemini to reason and approximate:
    - inventory_days (imports vs baseline)
    - past_delay_days (recent drops)
    - edges (exporter->importer with rough trade_value_usd)
    """
    prompt = f"""
You are a supply-chain analyst. Estimate recent semiconductor (HS 8541, 8542) trade signals between:
Seller Location: "{seller_loc}"
Import Location: "{import_loc}"
Return JSON with fields:
{{
  "inventory_days": float,  // 5-90 typical
  "past_delay_days": float, // 0-60 typical
  "edges": [
    {{"exporter": string, "importer": string, "trade_value_usd": float, "timestamp": "YYYY-MM"}}
  ]
}}
Be conservative; cite month strings in last 1-2 months if possible. If unsure, provide plausible conservative values.
"""
    model = get_gemini()
    data = {}
    try:
        # The LLM call sits inside the try so a provider error degrades this
        # agent to its defaults instead of failing the whole graph.
        resp = await model.agenerate_content(prompt)
        import json, re
        txt = resp.text or "{}"
        txt = re.sub(r"```json|```", "", txt).strip()
        data = json.loads(txt)
    except Exception:
        data = {}

    inv = float(data.get("inventory_days", 30.0))
    delay = float(data.get("past_delay_days", 5.0))
    edges_raw = data.get("edges", [])
    edges = []
    for e in edges_raw:
        try:
            edges.append(TradeEdge(**e))
        except Exception:
            continue

    return TradeFeatures(inventory_days=inv, past_delay_days=delay, edges=edges)
