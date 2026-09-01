#!/usr/bin/env python3
"""Preflight check for model training.

Run this first on any new machine. It answers three questions and nothing else:

1. Are the Python packages the training scripts need installed, and at what versions?
2. Is a CUDA GPU visible to PyTorch, and does a real tensor operation work on it?
3. Do the fused tables named by a config exist, and can they support a
   chronological split?

It never trains, never downloads, and never writes into a data profile.

Usage::

    python check_training_env.py
    python check_training_env.py --config config.yaml
    python check_training_env.py --config config.yaml --device cuda --json

Exit codes
----------
0  everything required is present
1  a required package or the requested device is missing
2  the packages and device are fine but the data is not ready
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

EXIT_OK = 0
EXIT_ENVIRONMENT = 1
EXIT_DATA = 2

REQUIRED_PACKAGES = ("numpy", "pandas", "yaml", "torch", "sklearn")
OPTIONAL_PACKAGES = ("matplotlib", "dotenv", "pyarrow")


def _version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "installed"))


def check_packages() -> tuple[dict[str, str | None], list[str]]:
    found = {name: _version(name) for name in (*REQUIRED_PACKAGES, *OPTIONAL_PACKAGES)}
    missing = [name for name in REQUIRED_PACKAGES if found[name] is None]
    return found, missing


def check_device(requested: str) -> dict[str, object]:
    from training import runtime

    result: dict[str, object] = {"requested": requested}
    try:
        choice = runtime.resolve_device(requested)
    except runtime.EnvironmentError_ as error:
        result.update({"ok": False, "error": str(error)})
        return result

    result.update(runtime.environment_report(choice))
    result["ok"] = True

    # A visible GPU is not a working GPU; prove it with a real kernel.
    try:
        import torch

        left = torch.randn(256, 256, device=choice.device)
        right = torch.randn(256, 256, device=choice.device)
        product = (left @ right).sum().item()
        result["matmul_smoke_test"] = "passed" if product == product else "failed (NaN)"
        if choice.device.type == "cuda":
            torch.cuda.synchronize(choice.device)
            result["cuda_memory"] = runtime.cuda_memory_summary(choice.device)
    except Exception as error:  # pragma: no cover - hardware/driver dependent
        result["ok"] = False
        result["matmul_smoke_test"] = f"failed: {error}"
    return result


def check_data(config_path: str) -> dict[str, object]:
    from common import load_config
    from training import data as data_module

    result: dict[str, object] = {"config": config_path}
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        return {**result, "ok": False, "error": f"config not found: {config_path}"}

    locations = data_module.table_paths(config)
    result["tables"] = {name: str(path) for name, path in locations.items()}
    result["tables_present"] = {
        name: bool(path.exists() or path.with_suffix(".csv").exists())
        for name, path in locations.items()
    }
    try:
        nodes, edges = data_module.load_tables(config)
        validation = data_module.validate_tables(nodes, edges)
        settings = config.get("model_training") or {}
        frame, stats = data_module.build_targets(
            nodes,
            horizon=int((config.get("analysis") or {}).get("horizon_months", 1)),
            baseline_window=int(settings.get("baseline_window", 12)),
            baseline_min_periods=int(settings.get("baseline_min_periods", 1)),
            taus=[float(value) for value in (config.get("analysis") or {}).get("taus", [0.30, 0.35, 0.40])],
        )
        supervised = data_module.supervised_months(frame, validation["months"])
        result.update(
            {
                "ok": len(supervised) >= 3,
                "rows": validation["node_rows"],
                "months": validation["month_count"],
                "countries": validation["node_count"],
                "edge_rows": validation["edge_rows"],
                "valid_targets": stats["valid_targets"],
                "supervised_months": len(supervised),
                "positives_by_tau": stats["positives_by_tau"],
            }
        )
        if len(supervised) < 3:
            result["error"] = (
                f"only {len(supervised)} month(s) have an observable target; a chronological split "
                "needs at least three"
            )
    except Exception as error:
        result.update({"ok": False, "error": str(error)})
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="Config whose tables should be checked")
    parser.add_argument("--device", default="auto", help="auto (default), cpu, cuda, or cuda:<index>")
    parser.add_argument("--skip-data", action="store_true", help="Only check packages and the device")
    parser.add_argument("--json", action="store_true", help="Emit the full report as JSON")
    args = parser.parse_args(argv)

    packages, missing = check_packages()
    report: dict[str, object] = {"packages": packages, "missing_required": missing}

    if missing:
        report["device"] = {"ok": False, "error": "skipped; required packages are missing"}
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print("Required package(s) missing:", ", ".join(missing))
            print("Install them with:  pip install -r requirements-train.txt")
        return EXIT_ENVIRONMENT

    report["device"] = check_device(args.device)
    if not args.skip_data:
        report["data"] = check_data(args.config)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        from training import runtime

        print("Packages")
        for name, version in packages.items():
            marker = "ok " if version else ("MISSING" if name in REQUIRED_PACKAGES else "optional, absent")
            print(f"  {name:<12} {version or '-':<24} {marker}")
        print("\nDevice")
        device_report = report["device"]
        if device_report.get("ok"):
            for line in runtime.describe_environment(device_report).splitlines():  # type: ignore[arg-type]
                print(f"  {line}")
            print(f"  Matmul smoke test: {device_report.get('matmul_smoke_test')}")
        else:
            print(f"  FAILED: {device_report.get('error')}")
        if "data" in report:
            data_report = report["data"]
            print(f"\nData ({data_report['config']})")
            for name, present in data_report.get("tables_present", {}).items():  # type: ignore[union-attr]
                print(f"  {name:<8} {'found' if present else 'MISSING'}  {data_report['tables'][name]}")
            if data_report.get("ok"):
                print(
                    f"  {data_report['rows']} country-month rows, {data_report['months']} months, "
                    f"{data_report['countries']} countries, {data_report['edge_rows']} edges"
                )
                print(
                    f"  {data_report['valid_targets']} valid targets across "
                    f"{data_report['supervised_months']} supervised months"
                )
                print(f"  positives by tau: {data_report['positives_by_tau']}")
            else:
                print(f"  NOT READY: {data_report.get('error')}")

    if not report["device"].get("ok"):  # type: ignore[union-attr]
        return EXIT_ENVIRONMENT
    if "data" in report and not report["data"].get("ok"):  # type: ignore[union-attr]
        return EXIT_DATA
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
