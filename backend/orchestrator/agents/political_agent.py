import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from orchestrator.utils.api_clients import get_gemini
from ..utils.schema import PoliticalFeatures

async def political_features(component_type: str, seller_loc: str, import_loc: str, seller_name: str | None) -> PoliticalFeatures:
    prompt = f"""
You are a geopolitics analyst. Considering semiconductor/electronics shipments for:
component="{component_type}", seller="{seller_name or 'unknown'}", seller_loc="{seller_loc}", import_loc="{import_loc}".
Assess current sanctions and political risk in the past 30 days relevant to this lane.
Return strict JSON: {{"sanction_flag": 0 or 1, "political_risk_score": float (0-1), "notes": "short"}}
Be conservative; if uncertain, sanction_flag=0 and risk_score near 0.3.
"""
    model = get_gemini()
    try:
        resp = await model.agenerate_content(prompt)
        import json, re
        s = (resp.text or "{}")
        s = re.sub(r"```json|```", "", s).strip()
        data = json.loads(s)
        return PoliticalFeatures(
            sanction_flag=int(data.get("sanction_flag", 0)),
            political_risk_score=float(data.get("political_risk_score", 0.3)),
            notes=data.get("notes", None)
        )
    except Exception:
        return PoliticalFeatures(sanction_flag=0, political_risk_score=0.3, notes=None)
