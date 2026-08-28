"""Shared utilities for the self-contained dataset pipeline.

All paths are resolved relative to this directory unless an absolute path is
given.  The helpers deliberately keep source pulls and derived tables on disk
so every stage can be re-run without repeatedly calling an external service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent


def load_config(path: str | Path = ROOT / "config.yaml") -> dict[str, Any]:
    """Load YAML configuration and resolve paths relative to ``dataset/``."""
    path = Path(path)
    if not path.is_absolute():
        cwd_path = Path.cwd() / path
        path = cwd_path if cwd_path.exists() else ROOT / path
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path)
    return config


def configured_path(config: Mapping[str, Any], key: str, default: str) -> Path:
    """Resolve a path stored in a config mapping relative to dataset/."""
    value = config.get(key, default)
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def nested(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def env_value(config: Mapping[str, Any], reference: str | None, default: str = "") -> str:
    """Resolve a config value that names an environment variable."""
    if not reference:
        return default
    return os.getenv(reference, default)


def parse_date(value: Any) -> date:
    parsed = pd.to_datetime(value, utc=True)
    if isinstance(parsed, pd.DatetimeIndex):
        raise ValueError(f"Expected one date, got {value!r}")
    return parsed.date()


def parse_timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def monthly_steps(config: Mapping[str, Any]) -> list[pd.Timestamp]:
    start = pd.Timestamp(nested(config, "analysis", "start_date"), tz="UTC")
    end = pd.Timestamp(nested(config, "analysis", "end_date"), tz="UTC")
    return list(pd.date_range(start=start, end=end, freq="MS", tz="UTC"))


def month_end(step: pd.Timestamp) -> pd.Timestamp:
    return step + pd.offsets.MonthEnd(0)


def vintage_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return text or "unknown"


def stable_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{slug(value)[:32]}_{digest}"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(payload: Any, path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read parquet where possible, otherwise CSV fallback."""
    if path.exists():
        if path.suffix.lower() in {".parquet", ".pq"}:
            try:
                return pd.read_parquet(path, **kwargs)
            except (ImportError, ModuleNotFoundError):
                csv_path = path.with_suffix(".csv")
                if csv_path.exists():
                    return pd.read_csv(csv_path, **kwargs)
                raise
        return pd.read_csv(path, **kwargs)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, **kwargs)
    raise FileNotFoundError(path)


def write_table(frame: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Write parquet and transparently fall back to CSV if parquet is unavailable."""
    ensure_parent(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            frame.to_parquet(path, index=index)
            return path
        except (ImportError, ModuleNotFoundError):
            csv_path = path.with_suffix(".csv")
            frame.to_csv(csv_path, index=index)
            return csv_path
    frame.to_csv(path, index=index)
    return path


def cache_path(config: Mapping[str, Any], source: str, filename: str) -> Path:
    base = configured_path(config, "cache_dir", "cache")
    return ensure_parent(base / source / filename)


def add_cache_metadata(frame: pd.DataFrame, source: str, vintage: str | None = None) -> pd.DataFrame:
    result = frame.copy()
    result["source"] = source
    result["vintage_date"] = vintage or vintage_now()
    return result


def canonical_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in frame.columns}
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]", "", candidate.lower())
        if key in normalized:
            return str(normalized[key])
    return None


def numeric(frame: pd.DataFrame, column: str | None, default: float = 0.0) -> pd.Series:
    if not column:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def add_common_cli(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="config.yaml", help="YAML config path")
    parser.add_argument("--force", action="store_true", help="Ignore cached output")


def request_headers() -> dict[str, str]:
    return {"User-Agent": "LAMDAAnalytics-dataset-pipeline/1.0"}


def atomic_download(url: str, destination: Path, timeout: int = 60) -> Path:
    """Download a URL atomically; never leave a partial cache file."""
    import requests

    ensure_parent(destination)
    if destination.exists():
        return destination
    response = requests.get(url, headers=request_headers(), timeout=timeout)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        handle.write(response.content)
        temporary = Path(handle.name)
    temporary.replace(destination)
    return destination


def iso_or_none(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return pd.Timestamp(value, tz="UTC").isoformat()
    except (TypeError, ValueError):
        return str(value)

