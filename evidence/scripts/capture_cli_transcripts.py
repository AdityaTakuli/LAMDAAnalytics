#!/usr/bin/env python3
"""Write six CLI transcripts under evidence/runs/<utc>/. No screenshots."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence"
BACKEND = ROOT / "backend"
PAYLOAD = EVIDENCE / "payloads" / "analyze_example.json"
PY = ROOT / "backend/.venv/bin/python"
if not PY.exists():
    PY = Path(sys.executable)

DUMMY_ENV = {
    **os.environ,
    "MAPPLS_API_KEY": os.environ.get("MAPPLS_API_KEY") or "test",
    "SERP_API_KEY": os.environ.get("SERP_API_KEY") or "test",
    "WEATHER_API_KEY": os.environ.get("WEATHER_API_KEY") or "test",
    "GROQ_API_KEY": os.environ.get("GROQ_API_KEY") or "test",
}


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def header(title: str) -> str:
    return (
        f"=== {title} ===\n"
        f"utc={datetime.now(timezone.utc).isoformat()}\n"
        f"git={git_hash()}\n"
        f"python={PY}\n"
        f"cwd={ROOT}\n"
        "---\n"
    )


def run_cmd(cmd: list[str], cwd: Path, env: dict | None = None, timeout: float | None = 180) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    p = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env or DUMMY_ENV,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return p.returncode, (p.stdout or "") + (p.stderr or ""), time.perf_counter() - t0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_port(port: int, timeout: float = 20.0) -> None:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(0.3)
            try:
                s.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.15)
    raise RuntimeError(f"uvicorn did not bind 127.0.0.1:{port}")


def start_uvicorn(port: int, extra_env: dict | None = None) -> subprocess.Popen:
    env = {**DUMMY_ENV, **(extra_env or {})}
    proc = subprocess.Popen(
        [
            str(PY),
            "-m",
            "uvicorn",
            "main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(BACKEND),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_port(port)
    except Exception:
        out = proc.stdout.read() if proc.stdout else ""
        proc.kill()
        raise RuntimeError(f"uvicorn failed:\n{out}")
    return proc


def stop_uvicorn(proc: subprocess.Popen) -> str:
    server_log = ""
    try:
        proc.send_signal(signal.SIGINT)
        try:
            server_log, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            server_log, _ = proc.communicate(timeout=3)
    except Exception:
        proc.kill()
    return server_log or ""


def write(out_dir: Path, name: str, body: str) -> None:
    (out_dir / name).write_text(body, encoding="utf-8")


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = EVIDENCE / "runs" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = []

    # 1. Mocked LangGraph E2E
    code, log, elapsed = run_cmd(
        [str(PY), "test_langgraph_network.py"],
        cwd=BACKEND,
    )
    write(
        out_dir,
        "01_langgraph_mock_e2e.log",
        header("01 mocked LangGraph E2E (no external APIs)")
        + f"exit={code}\nelapsed_s={elapsed:.3f}\n--- stdout/stderr ---\n"
        + log,
    )
    steps.append({"name": "01_langgraph_mock_e2e", "seconds": round(elapsed, 3), "exit": code})

    # 2. generate_results_figures.py
    code, log, elapsed = run_cmd(
        [str(PY), "generate_results_figures.py"],
        cwd=ROOT / "Research Paper",
    )
    write(
        out_dir,
        "02_generate_results_figures.log",
        header("02 generate_results_figures.py")
        + f"exit={code}\nelapsed_s={elapsed:.3f}\n--- stdout/stderr ---\n"
        + log,
    )
    steps.append({"name": "02_generate_results_figures", "seconds": round(elapsed, 3), "exit": code})

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = start_uvicorn(
        port,
        extra_env={"AGENT_TIMEOUT_SECONDS": "8", "HTTP_TIMEOUT_SECONDS": "2"},
    )
    try:
        # 3. Fast endpoints
        t0 = time.perf_counter()
        chunks = [header("03 GET fast endpoints")]
        chunks.append(f"base={base}\n")
        for path in ("/model/info", "/analytics/overview", "/monitoring/alerts"):
            c, log, el = run_cmd(
                [
                    "curl",
                    "-sS",
                    "-w",
                    "\nhttp_code=%{http_code}\ntime_total_s=%{time_total}\n",
                    f"{base}{path}",
                ],
                cwd=ROOT,
            )
            chunks.append(f"\n=== GET {path} elapsed_s={el:.3f} exit={c} ===\n{log}")
        write(out_dir, "03_fast_endpoints.log", "".join(chunks) + f"\nwall_s={time.perf_counter()-t0:.3f}\n")
        steps.append({"name": "03_fast_endpoints", "seconds": round(time.perf_counter() - t0, 3), "exit": 0})

        # 4. Timed /analyze via python -m evidence.analyze_client
        t0 = time.perf_counter()
        code, log, elapsed = run_cmd(
            [
                str(PY),
                "-m",
                "evidence.analyze_client",
                "--url",
                f"{base}/analyze",
                "--payload",
                str(PAYLOAD),
                "--timeout",
                "55",
            ],
            cwd=ROOT,
            timeout=70,
        )
        note = (
            "NOTE: dummy/test keys yield a FastAPI mock body if agents raise. "
            "A live ~26s run needs a real backend/.env and bash evidence/scripts/time_analyze.sh.\n"
        )
        write(
            out_dir,
            "04_analyze_timed.log",
            header("04 timed POST /analyze")
            + note
            + f"exit={code}\nelapsed_s={elapsed:.3f}\n--- stdout/stderr ---\n"
            + log,
        )
        steps.append({"name": "04_analyze_timed", "seconds": round(elapsed, 3), "exit": code})
    finally:
        stop_uvicorn(proc)

    # 5. Forced timeout → mock body
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = start_uvicorn(
        port,
        extra_env={"AGENT_TIMEOUT_SECONDS": "1", "LAMDA_FORCE_TIMEOUT": "1"},
    )
    try:
        code, log, elapsed = run_cmd(
            [
                str(PY),
                "-m",
                "evidence.analyze_client",
                "--url",
                f"{base}/analyze",
                "--payload",
                str(PAYLOAD),
                "--timeout",
                "20",
            ],
            cwd=ROOT,
            timeout=25,
        )
        write(
            out_dir,
            "05_forced_timeout.log",
            header("05 forced timeout (LAMDA_FORCE_TIMEOUT=1, AGENT_TIMEOUT_SECONDS=1)")
            + "Expect ~1s and a structurally complete mock AnalyzeResponse.\n"
            + f"exit={code}\nelapsed_s={elapsed:.3f}\n--- stdout/stderr ---\n"
            + log,
        )
        steps.append({"name": "05_forced_timeout", "seconds": round(elapsed, 3), "exit": code})
    finally:
        stop_uvicorn(proc)

    # 6. Forced news-agent failure → zeros, graph still returns (mocked other agents)
    t0 = time.perf_counter()
    code, log, elapsed = run_cmd(
        [str(PY), str(EVIDENCE / "scripts" / "force_news_fail.py")],
        cwd=BACKEND,
    )
    write(
        out_dir,
        "06_forced_agent_failure.log",
        header("06 forced news failure (LAMDA_FORCE_NEWS_FAIL=1, other agents mocked)")
        + "Expect news_vol_7d=0 and sources containing LAMDA_FORCE_NEWS_FAIL; still a real graph response.\n"
        + f"exit={code}\nelapsed_s={elapsed:.3f}\n--- stdout/stderr ---\n"
        + log,
    )
    steps.append({"name": "06_forced_agent_failure", "seconds": round(elapsed, 3), "exit": code})

    manifest = {
        "timestamp_utc": stamp,
        "git_commit": git_hash(),
        "kind": "cli_transcripts",
        "steps": steps,
        "files": [
            "01_langgraph_mock_e2e.log",
            "02_generate_results_figures.log",
            "03_fast_endpoints.log",
            "04_analyze_timed.log",
            "05_forced_timeout.log",
            "06_forced_agent_failure.log",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"CLI transcripts written to {out_dir}")


if __name__ == "__main__":
    main()
