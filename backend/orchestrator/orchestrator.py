import asyncio
import os
import uuid
from datetime import datetime, timezone

from config.settings import settings
from .graph import analysis_graph
from .utils.schema import AnalyzeRequest, AnalyzeResponse


async def run_analysis(inp: AnalyzeRequest) -> AnalyzeResponse:
    """Run the LangGraph multi-agent analysis network."""
    request_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    initial_state = {
        "request": inp,
        "request_id": request_id,
        "created_at": now.isoformat(),
        "errors": [],
    }

    async def _invoke():
        # Evidence flag: hang until AGENT_TIMEOUT_SECONDS fires (see evidence/REPRODUCE.md).
        if os.environ.get("LAMDA_FORCE_TIMEOUT") == "1":
            await asyncio.sleep(10_000)
        return await analysis_graph.ainvoke(initial_state)

    final_state = await asyncio.wait_for(
        _invoke(),
        timeout=settings.agent_timeout_seconds,
    )

    response = final_state.get("response")
    if response is None:
        raise RuntimeError("LangGraph analysis completed without a response payload")
    return response
