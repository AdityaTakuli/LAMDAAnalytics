#!/usr/bin/env python3
"""CLI client for POST /analyze. Print JSON + wall-clock to stdout (redirect to a .log)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx


def main() -> int:
    p = argparse.ArgumentParser(description="Timed POST /analyze; write transcript to stdout.")
    p.add_argument("--url", default="http://127.0.0.1:8007/analyze")
    p.add_argument("--payload", default="evidence/payloads/analyze_example.json")
    p.add_argument("--timeout", type=float, default=120.0)
    args = p.parse_args()

    payload_path = Path(args.payload)
    body = json.loads(payload_path.read_text(encoding="utf-8"))

    print(f"utc={datetime.now(timezone.utc).isoformat()}")
    print(f"url={args.url}")
    print(f"payload={payload_path.resolve()}")
    print(f"timeout_s={args.timeout}")
    print("--- request ---")
    print(json.dumps(body, indent=2))
    print("--- response ---")

    t0 = time.perf_counter()
    try:
        r = httpx.post(args.url, json=body, timeout=args.timeout)
        elapsed = time.perf_counter() - t0
        print(f"http_code={r.status_code}")
        print(f"time_total_s={elapsed:.3f}")
        try:
            payload = r.json()
            feats = payload.get("features") or {}
            score = (payload.get("tgn_result") or {}).get("risk_score")
            if (
                feats.get("inventory_days") == 0.65
                and feats.get("news_vol_7d") == 0.78
                and score == 0.45
            ):
                print("WARNING: body matches hardcoded mock fallback in backend/main.py")
            else:
                print("INFO: body does not match hardcoded mock fallback")
            print(json.dumps(payload, indent=2, default=str))
        except Exception:
            print(r.text)
        return 0 if r.status_code < 500 else 1
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"time_total_s={elapsed:.3f}")
        print(f"client_error={type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
