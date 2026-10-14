#!/usr/bin/env python3
"""what does the multi-agent decomposition buy?

Compares five ways of producing the same seven-feature vector, on the real
backend code, with the LLM replaced by an instrumented stub so that latency and
faults are controlled and the comparison isolates orchestration:

  A  LangGraph, as submitted     git HEAD snapshot of backend/ (agents await a
                                 blocking LLM call; whole-graph timeout only)
  B  LangGraph, revised          working-tree backend/ (non-blocking LLM calls,
                                 per-agent timeouts, guarded agents)
  C  asyncio.gather, no LangGraph   revised agents and guards, plain asyncio
  D  sequential, no LangGraph       revised agents and guards, one after another
  E  monolithic                  one LLM call returns all LLM-derived fields;
                                 weather computed as in the other arms

Stub: each LLM call sleeps U(0.5, 2.5) s (matching the 2.4-10.6 s end-to-end
latency of archived live /analyze calls with four LLM calls) and returns
schema-valid JSON. Faults: "error" raises with probability p per call; "hang"
blocks for 60 s (the Groq client's default request timeout) with probability p.
Weather retrieval is stubbed with a 0.3-0.8 s async fetch. All times are run at
SCALE x real time and reported back in real-equivalent seconds. The published
GSCPI table is hidden in every arm so that all arms make the same four LLM
calls (trade, news, political, GSCPI).

What the stub cannot measure: extraction accuracy and the latency effect of a
longer monolithic prompt (E's single call uses the same latency distribution,
so E's latency is a lower bound).

Usage: backend/.venv/bin/python evidence/agent_architecture_study.py
Writes results/benchmark/agent_architecture_study.{csv,json}.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import tarfile
import io
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark"
SCALE = 0.05
GRAPH_TIMEOUT = 90.0
NODE_TIMEOUT = 30.0
HANG_SECONDS = 60.0
N_REQUESTS = 40
CONCURRENT = 8
FEATURE_CHANNELS = {"trade": 2, "news": 3, "weather": 1, "gscpi": 1}  # political is not in the vector

STUB_JSON = {
    "news": {"news_vol_7d": 3, "neg_tone_frac_3d": 0.4, "strike_flag_7d": 0},
    "trade": {"inventory_days": 28.0, "past_delay_days": 4.0, "edges": []},
    "political": {"sanction_flag": 0, "political_risk_score": 0.3, "notes": "stub"},
    "gscpi": {"global_risk": 0.1, "timestamp": "2026-09"},
}
MONO_PROMPT = "MONOLITHIC: return news_vol_7d, neg_tone_frac_3d, strike_flag_7d, inventory_days, past_delay_days, sanction_flag, political_risk_score, global_risk"


# --------------------------------------------------------------------------- #
# worker (runs inside a subprocess with one backend version on sys.path)
# --------------------------------------------------------------------------- #
class Stub:
    def __init__(self, seed: int, fault: str, p: float):
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.fault, self.p = fault, p
        self.log: list[tuple[int, str, str]] = []
        self.request = 0

    def channel(self, prompt: str) -> str:
        if prompt.startswith("MONOLITHIC"):
            return "mono"
        if "news snippets" in prompt:
            return "news"
        if "geopolitics" in prompt:
            return "political"
        if "GSCPI" in prompt:
            return "gscpi"
        return "trade"

    def __call__(self, prompt: str) -> str:
        ch = self.channel(prompt)
        with self.lock:
            latency = self.rng.uniform(0.5, 2.5)
            draw = self.rng.random()
            req = self.request
        if self.fault == "hang" and draw < self.p:
            time.sleep(HANG_SECONDS * SCALE)
            with self.lock:
                self.log.append((req, ch, "hang"))
            raise TimeoutError("stub: provider timeout")
        time.sleep(latency * SCALE)
        if self.fault == "error" and draw < self.p:
            with self.lock:
                self.log.append((req, ch, "error"))
            raise RuntimeError("stub: provider error")
        with self.lock:
            self.log.append((req, ch, "ok"))
        if ch == "mono":
            merged = {}
            for part in STUB_JSON.values():
                merged.update(part)
            return json.dumps(merged)
        return json.dumps(STUB_JSON[ch])


def worker(args) -> None:
    sys.path.insert(0, args.backend)
    os.chdir(args.backend)
    os.environ.update({
        "MAPPLS_API_KEY": "PASTE_YOUR_KEY", "SERP_API_KEY": "PASTE_YOUR_KEY",
        "WEATHER_API_KEY": "PASTE_YOUR_KEY", "GROQ_API_KEY": "stub",
        "AGENT_TIMEOUT_SECONDS": "90",
        "SCORING_STATE_PATH": str(Path(args.scratch) / f"state_{args.arm}_{args.fault}.json"),
        "GSCPI_SERIES_PATH": str(Path(args.scratch) / "no_such_series.csv"),
    })
    from config.settings import settings
    # scaled timeouts (the settings field is an int, so assign after load)
    settings.agent_timeout_seconds = GRAPH_TIMEOUT * SCALE
    if "agent_node_timeout_seconds" in type(settings).model_fields:  # absent in the HEAD snapshot
        settings.agent_node_timeout_seconds = NODE_TIMEOUT * SCALE
    from orchestrator.utils import api_clients
    stub = Stub(seed=args.seed, fault=args.fault, p=args.p)
    api_clients.llm_generate = stub

    async def fake_weather(lat, lon):
        await asyncio.sleep(random.uniform(0.3, 0.8) * SCALE)
        return {"daily": [{"temp": {"day": 20.0 + i % 3}} for i in range(7)]}
    api_clients.fetch_weather = fake_weather
    import orchestrator.agents.weather_agent as wa
    wa.fetch_weather = fake_weather
    if hasattr(wa, "weather_cache"):
        wa.weather_cache.clear()
    import orchestrator.agents.news_agent as na
    na.news_cache.clear()

    from orchestrator.utils.schema import AnalyzeRequest
    req = AnalyzeRequest(component_type="Semiconductor", seller_location="Hsinchu, Taiwan",
                         import_location="Los Angeles, USA", seller_name="TSMC")

    async def run_graph():
        from orchestrator.orchestrator import run_analysis
        try:
            await run_analysis(req)
            return "ok"
        except Exception as exc:  # main.py turns this into the mock body
            return f"mock ({type(exc).__name__})"

    async def guarded(coro):
        try:
            return await asyncio.wait_for(coro, timeout=NODE_TIMEOUT * SCALE), True
        except Exception:
            return None, False

    async def agents_plain(mode: str):
        from orchestrator.agents.gscpi_agent import gscpi_features
        from orchestrator.agents.news_agent import analyze_news
        from orchestrator.agents.political_agent import political_features
        from orchestrator.agents.trade_agent import fetch_trade_features
        from orchestrator.agents.weather_agent import weather_features
        from orchestrator.utils.geocoding import resolve_pair
        s, _ = await resolve_pair(req.seller_location, req.import_location)
        calls = [
            lambda: fetch_trade_features(req.component_type, req.seller_location, req.import_location),
            lambda: analyze_news(req.component_type, req.seller_location, req.import_location, req.seller_name),
            lambda: weather_features(*(s or (0.0, 0.0))),
            lambda: political_features(req.component_type, req.seller_location, req.import_location, req.seller_name),
            lambda: gscpi_features(),
        ]
        if mode == "gather":
            await asyncio.gather(*(guarded(c()) for c in calls))
        else:
            for c in calls:
                await guarded(c())
        return "ok"

    async def monolithic():
        from orchestrator.agents.weather_agent import weather_features
        from orchestrator.utils.geocoding import resolve_pair
        s, _ = await resolve_pair(req.seller_location, req.import_location)

        async def one_call():
            return await asyncio.to_thread(api_clients.llm_generate, MONO_PROMPT)
        await asyncio.gather(guarded(one_call()), guarded(weather_features(*(s or (0.0, 0.0)))))
        return "ok"

    runner = {"A": run_graph, "B": run_graph, "C": lambda: agents_plain("gather"),
              "D": lambda: agents_plain("sequential"), "E": monolithic}[args.arm]

    def clear_caches():
        na.news_cache.clear()
        if hasattr(wa, "weather_cache"):
            wa.weather_cache.clear()

    async def one(i):
        clear_caches()
        stub.request = i
        t0 = time.perf_counter()
        status = await runner()
        return (time.perf_counter() - t0) / SCALE, status

    async def main_async():
        stub.request = -1
        await runner()  # warm-up: imports, graph compile, scorer state
        rows = []
        for i in range(args.n):
            rows.append(await one(i))
        clear_caches()
        stub.request = -2
        t0 = time.perf_counter()
        await asyncio.gather(*(runner() for _ in range(args.concurrent)))
        burst = (time.perf_counter() - t0) / SCALE
        return rows, burst

    rows, burst = asyncio.run(main_async())
    # features extracted per request: a channel counts if its LLM call (or the
    # weather fetch) succeeded and the request did not fall back to the mock.
    per_req = []
    for i, (lat, status) in enumerate(rows):
        events = {(ch, st) for r, ch, st in stub.log if r == i}
        if status != "ok":
            got = 0
        elif args.arm == "E":
            got = (6 if ("mono", "ok") in events else 0) + 1
        else:
            got = 1 + sum(n for ch, n in FEATURE_CHANNELS.items() if ch != "weather" and (ch, "ok") in events)
        per_req.append({"latency_s": lat, "status": status, "features_extracted": got})
    print(json.dumps({"rows": per_req, "burst_s": burst}))


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def snapshot_head(scratch: Path) -> Path:
    dest = scratch / "deployed_backend"
    if not dest.exists():
        blob = subprocess.run(["git", "-C", str(ROOT), "archive", "HEAD", "backend"],
                              check=True, capture_output=True).stdout
        with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
            tar.extractall(scratch, filter="data")
        (scratch / "backend").rename(dest)
    return dest


def summarise(arm: str, fault: str, p: float, payload: dict) -> dict:
    rows = payload["rows"]
    lat = sorted(r["latency_s"] for r in rows)
    feats = [r["features_extracted"] for r in rows]
    return {
        "arm": arm, "fault": fault, "p_fault_per_call": p, "requests": len(rows),
        "latency_median_s": statistics.median(lat), "latency_p95_s": lat[int(0.95 * (len(lat) - 1))],
        "mock_fallback_rate": sum(r["status"] != "ok" for r in rows) / len(rows),
        "mean_features_extracted_of_7": statistics.mean(feats),
        "requests_with_all_7": sum(f == 7 for f in feats) / len(feats),
        f"burst_{CONCURRENT}_concurrent_s": payload["burst_s"],
    }


ARMS = {"A": "LangGraph, as submitted (HEAD)", "B": "LangGraph, revised",
        "C": "asyncio.gather, no LangGraph", "D": "sequential, no LangGraph", "E": "monolithic single call"}


def driver() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="lamda_arch_"))
    head = snapshot_head(scratch)
    revised = ROOT / "backend"
    results = []
    for fault, p in (("none", 0.0), ("error", 0.10), ("hang", 0.10)):
        for arm in "ABCDE":
            backend = head if arm == "A" else revised
            cmd = [sys.executable, __file__, "--worker", "--backend", str(backend), "--arm", arm,
                   "--fault", fault, "--p", str(p), "--n", str(N_REQUESTS), "--seed", "11",
                   "--concurrent", str(CONCURRENT), "--scratch", str(scratch)]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            line = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
            if proc.returncode != 0 or not line:
                print(proc.stderr[-2000:])
                raise SystemExit(f"worker failed: {arm} {fault}")
            rec = summarise(arm, fault, p, json.loads(line[-1]))
            rec["arm_label"] = ARMS[arm]
            results.append(rec)
            print(f"{fault:5s} {arm} {ARMS[arm]:32s} median {rec['latency_median_s']:6.2f}s  p95 {rec['latency_p95_s']:6.2f}s  "
                  f"mock {rec['mock_fallback_rate']:.2f}  feats {rec['mean_features_extracted_of_7']:.2f}  "
                  f"all7 {rec['requests_with_all_7']:.2f}  burst{CONCURRENT} {rec[f'burst_{CONCURRENT}_concurrent_s']:6.1f}s",
                  flush=True)
    import csv
    with (OUT / "agent_architecture_study.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    (OUT / "agent_architecture_study.json").write_text(json.dumps({
        "scale": SCALE, "graph_timeout_s": GRAPH_TIMEOUT, "node_timeout_s": NODE_TIMEOUT,
        "hang_s": HANG_SECONDS, "requests_per_cell": N_REQUESTS, "concurrent_burst": CONCURRENT,
        "llm_latency_s": "U(0.5, 2.5)", "results": results}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--backend")
    ap.add_argument("--arm")
    ap.add_argument("--fault", default="none")
    ap.add_argument("--p", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=N_REQUESTS)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--concurrent", type=int, default=CONCURRENT)
    ap.add_argument("--scratch")
    a = ap.parse_args()
    worker(a) if a.worker else driver()
