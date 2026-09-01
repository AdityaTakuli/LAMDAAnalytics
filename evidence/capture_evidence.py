#!/usr/bin/env python3
"""Capture reproducibility evidence: audit, timings, figure regen."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path(__file__).resolve().parent
RUNS = EVIDENCE / "runs"


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=cwd or ROOT, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr, time.perf_counter() - t0


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "timestamp_utc": stamp,
        "git_commit": git_hash(),
        "recommended_tag": "v1.0-paper",
        "steps": [],
    }

    py = ROOT / "backend/.venv/bin/python"
    if not py.exists():
        py = Path(sys.executable)

    # Mocked LangGraph E2E timing
    code, log, elapsed = run([str(py), "test_langgraph_network.py"], cwd=ROOT / "backend")
    (out_dir / "langgraph_mock_timing.txt").write_text(log, encoding="utf-8")
    manifest["steps"].append({"name": "langgraph_mock_e2e", "seconds": round(elapsed, 3), "exit": code})

    # Figure regeneration
    code, log, elapsed = run([str(py), "generate_results_figures.py"], cwd=ROOT / "Research Paper")
    (out_dir / "generate_results_figures.log").write_text(log, encoding="utf-8")
    manifest["steps"].append({"name": "generate_results_figures", "seconds": round(elapsed, 3), "exit": code})

    # Bootstrap CIs
    code, log, elapsed = run([str(py), str(EVIDENCE / "bootstrap_ci.py")])
    (out_dir / "bootstrap_ci.log").write_text(log, encoding="utf-8")
    manifest["steps"].append({"name": "bootstrap_ci", "seconds": round(elapsed, 3), "exit": code})

    # Six CLI transcripts (nested evidence/runs/<utc>/)
    code, log, elapsed = run([str(py), str(EVIDENCE / "scripts" / "capture_cli_transcripts.py")])
    (out_dir / "capture_cli_transcripts.log").write_text(log, encoding="utf-8")
    manifest["steps"].append({"name": "capture_cli_transcripts", "seconds": round(elapsed, 3), "exit": code})

    # Copy latest bootstrap + figure manifest
    for name in ("bootstrap_ci.json", "bootstrap_ci.md"):
        src = EVIDENCE / name
        if src.exists():
            (out_dir / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    fig_manifest = ROOT / "Research Paper/figures/manifest.json"
    if fig_manifest.exists():
        (out_dir / "figures_manifest.json").write_text(fig_manifest.read_text(encoding="utf-8"), encoding="utf-8")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Evidence run written to {out_dir}")


if __name__ == "__main__":
    main()
