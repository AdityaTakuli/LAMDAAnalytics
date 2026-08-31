"""Device selection, determinism, and environment reporting.

The training entry points must behave identically on a CPU-only laptop, a
Linux CUDA box, and a Windows CUDA box. This module is the single place where
that difference is resolved, so no other module needs to branch on hardware.
"""

from __future__ import annotations

import logging
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER = logging.getLogger("training.runtime")


class EnvironmentError_(RuntimeError):
    """Raised when the requested compute device cannot be provided."""


@dataclass(frozen=True)
class DeviceChoice:
    """The resolved device plus the reason it was chosen."""

    device: torch.device
    requested: str
    reason: str

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"


def _cuda_diagnostic() -> str:
    """Explain, in one actionable sentence, why CUDA is unavailable."""
    if not hasattr(torch, "cuda"):
        return "this PyTorch build has no CUDA module at all"
    if torch.version.cuda is None:  # type: ignore[attr-defined]
        return (
            "the installed PyTorch is a CPU-only build (torch.version.cuda is None); "
            "reinstall a CUDA wheel, for example "
            "pip install torch --index-url https://download.pytorch.org/whl/cu124"
        )
    return (
        f"PyTorch was built against CUDA {torch.version.cuda} but no usable GPU was "  # type: ignore[attr-defined]
        "detected; check that an NVIDIA driver is installed and that `nvidia-smi` works"
    )


def resolve_device(requested: str = "auto") -> DeviceChoice:
    """Turn ``auto`` / ``cpu`` / ``cuda`` / ``cuda:N`` into a real device.

    ``auto`` prefers CUDA and silently falls back to CPU. An explicit ``cuda``
    request never falls back: a run that was asked for a GPU and quietly used
    the CPU would misreport its own timings, so it fails loudly instead.
    """
    requested = (requested or "auto").strip().lower()
    available = bool(getattr(torch, "cuda", None)) and torch.cuda.is_available()

    if requested == "auto":
        if available:
            return DeviceChoice(torch.device("cuda:0"), requested, "CUDA is available")
        return DeviceChoice(torch.device("cpu"), requested, f"CUDA unavailable: {_cuda_diagnostic()}")

    if requested == "cpu":
        return DeviceChoice(torch.device("cpu"), requested, "CPU explicitly requested")

    if requested == "cuda" or requested.startswith("cuda:"):
        if not available:
            raise EnvironmentError_(
                f"--device {requested} was requested but CUDA is not usable: {_cuda_diagnostic()}. "
                "Use --device cpu to run on the processor instead."
            )
        index = 0
        if ":" in requested:
            suffix = requested.split(":", 1)[1]
            if not suffix.isdigit():
                raise EnvironmentError_(f"Invalid device string {requested!r}; expected cuda:<integer>")
            index = int(suffix)
        count = torch.cuda.device_count()
        if index >= count:
            raise EnvironmentError_(
                f"--device {requested} was requested but only {count} CUDA device(s) are visible "
                f"(valid indices 0..{max(count - 1, 0)})."
            )
        torch.cuda.set_device(index)
        return DeviceChoice(torch.device(f"cuda:{index}"), requested, "CUDA explicitly requested")

    raise EnvironmentError_(
        f"Unknown --device {requested!r}. Valid values are: auto, cpu, cuda, cuda:<index>."
    )


def set_determinism(seed: int, strict: bool = False) -> dict[str, Any]:
    """Seed every generator the training path can touch.

    ``strict`` additionally asks PyTorch for deterministic kernels. It is
    requested with ``warn_only=True`` on purpose: the GCN uses ``index_add_``,
    which has no deterministic CUDA implementation, and a hard error there
    would make ``--deterministic`` unusable on GPU. The warning is surfaced in
    the run manifest so the reduced guarantee is never silent.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if getattr(torch, "cuda", None) and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    report: dict[str, Any] = {"seed": seed, "strict_requested": bool(strict)}
    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            report["deterministic_algorithms"] = "enabled (warn_only)"
        except Exception as exc:  # pragma: no cover - depends on the torch build
            report["deterministic_algorithms"] = f"unavailable: {exc}"
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            report["cudnn"] = "deterministic"
        except Exception as exc:  # pragma: no cover
            report["cudnn"] = f"unavailable: {exc}"
        report["caveat"] = (
            "GCN mean aggregation uses index_add_, which has no deterministic CUDA "
            "kernel; bitwise reproducibility is guaranteed on CPU only."
        )
    return report


def _package_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception:
        return "not installed"
    return str(getattr(module, "__version__", "unknown"))


def environment_report(choice: DeviceChoice | None = None) -> dict[str, Any]:
    """Collect everything needed to reproduce or debug a run."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "torch": torch.__version__,
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "torch_cudnn": (
            torch.backends.cudnn.version()
            if getattr(torch.backends, "cudnn", None) and torch.backends.cudnn.is_available()
            else None
        ),
        "cuda_available": bool(getattr(torch, "cuda", None)) and torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if getattr(torch, "cuda", None) and torch.cuda.is_available() else 0,
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "sklearn": _package_version("sklearn"),
        "matplotlib": _package_version("matplotlib"),
        "yaml": _package_version("yaml"),
    }
    if report["cuda_available"]:
        devices = []
        for index in range(report["cuda_device_count"]):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "capability": f"{properties.major}.{properties.minor}",
                    "total_memory_gb": round(properties.total_memory / (1024 ** 3), 2),
                    "multi_processor_count": properties.multi_processor_count,
                }
            )
        report["cuda_devices"] = devices
    if choice is not None:
        report["requested_device"] = choice.requested
        report["selected_device"] = str(choice.device)
        report["device_reason"] = choice.reason
    return report


def describe_environment(report: dict[str, Any]) -> str:
    """Render the environment report as a short human-readable block."""
    lines = [
        f"Python           : {report['python']}",
        f"Platform         : {report['platform']}",
        f"PyTorch          : {report['torch']} (CUDA build: {report['torch_cuda_build'] or 'cpu-only'})",
        f"CUDA available   : {report['cuda_available']}",
    ]
    for device in report.get("cuda_devices", []):
        lines.append(
            f"  GPU {device['index']}          : {device['name']} "
            f"(sm_{device['capability'].replace('.', '')}, {device['total_memory_gb']} GB)"
        )
    if "selected_device" in report:
        lines.append(f"Selected device  : {report['selected_device']} ({report['device_reason']})")
    return "\n".join(lines)


def configure_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure root logging once, optionally mirroring to a file."""
    logger = logging.getLogger("training")
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def cuda_memory_summary(device: torch.device) -> dict[str, Any] | None:
    """Peak allocation for the run, or ``None`` on CPU."""
    if device.type != "cuda":
        return None
    return {
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024 ** 2), 2),
    }
