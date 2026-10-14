"""GSCPI feature: read the published NY Fed series, not a language-model recall.

The offline corpus reads ``global_risk`` from the New York Fed's published
GSCPI table (dataset/ingest_gscpi.py -> processed/gscpi_monthly.csv). The live
agent now reads the same table, so the served value and the trained value come
from one series on one scale (GSCPI standard-deviation units). The previous
behaviour, asking the LLM to recall the index on a 0-1 scale, is kept only as a
last-resort fallback when no published table is reachable, and is labelled as
such in ``timestamp``.
"""
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from orchestrator.utils.api_clients import get_gemini
from ..utils.schema import GSCPIFeatures
from ..utils.timeutils import utc_now_iso

_DEFAULT_SERIES = (
    Path(__file__).resolve().parents[3]
    / "dataset" / "data" / "four_year_2021_2024" / "processed" / "gscpi_monthly.csv"
)


def published_gscpi(as_of: datetime | None = None, path: str | os.PathLike | None = None) -> GSCPIFeatures | None:
    """Latest published GSCPI month whose ``available_at`` is not after ``as_of``.

    This is the same causal rule fuse_dataset._causal_global applies offline.
    """
    series = Path(path or os.environ.get("GSCPI_SERIES_PATH") or _DEFAULT_SERIES)
    if not series.exists():
        return None
    as_of = as_of or datetime.now(timezone.utc)
    best = None
    with series.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                available = datetime.fromisoformat(row["available_at"])
                value = float(row["global_risk"])
            except (KeyError, ValueError):
                continue
            if available <= as_of and (best is None or row["month"] > best[0]):
                best = (row["month"], value)
    if best is None:
        return None
    return GSCPIFeatures(global_risk=best[1], timestamp=f"{best[0]} (published NY Fed series)")


async def gscpi_features() -> GSCPIFeatures:
    published = published_gscpi()
    if published is not None:
        return published
    prompt = """
Fetch the latest NY Fed Global Supply Chain Pressure Index (GSCPI) value (monthly).
Return strict JSON: {"global_risk": float, "timestamp": "YYYY-MM"}.
If you cannot fetch today, return last known recent value.
"""
    model = get_gemini()
    try:
        resp = await model.agenerate_content(prompt)
        import json, re
        s = (resp.text or "{}")
        s = re.sub(r"```json|```", "", s).strip()
        data = json.loads(s)
        return GSCPIFeatures(
            global_risk=float(data.get("global_risk", 0.2)),
            timestamp=f"{data.get('timestamp', utc_now_iso()[:7])} (LLM prior; published series unavailable)",
        )
    except Exception:
        return GSCPIFeatures(global_risk=0.2, timestamp=utc_now_iso()[:7])
