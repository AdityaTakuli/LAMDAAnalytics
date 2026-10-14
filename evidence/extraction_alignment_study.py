#!/usr/bin/env python3
"""Paper-21 revision: are the live LLM extractors right, and does decomposition help?

Ground truth is the fused panel built from published series
(dataset/.../processed/nodes_monthly.csv): Comtrade-derived trade proxies,
GDELT news/tone/strike features, and the NY Fed GSCPI. For each country-month
the deployed prompts are asked, with an explicit as-of month, for the fields
they produce in production, and the answers are scored against the panel.

Arms (same model, temperature, and JSON handling as the backend):
  per_agent    the deployed Trade, News (empty scrape, i.e. the no-retrieval
               path the archived live runs exercised) and GSCPI prompts, one
               call per agent
  monolithic   one call that returns all six LLM-derived fields at once

Scored:
  * alignment     Spearman rho (all fields), MAE (GSCPI, trade proxies),
                  agreement (strike flag) against the panel
  * validity      share of calls returning parseable, in-range JSON
  * stability     coefficient of variation over 3 repeats on 10 country-months
  * usefulness    Spearman rho of each extracted field with the next-month
                  log change T1, next to the same rho for the published field

Usage (needs a valid GROQ_API_KEY in backend/.env):
  backend/.venv/bin/python evidence/extraction_alignment_study.py [--limit N]
Writes results/paper21_revision/extraction_alignment_{calls.csv,summary.csv}.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.chdir(ROOT / "backend")

OUT = ROOT / "results" / "paper21_revision"
PANEL = ROOT / "dataset/data/four_year_2021_2024/processed/nodes_monthly.csv"
UNIVERSE = ROOT / "dataset/data/four_year_2021_2024/processed/country_universe.csv"
NODE = {"CHN": "country_cn", "KOR": "country_kr", "HKG": "country_344", "SGP": "country_sg",
        "VNM": "country_vn", "MYS": "country_my", "JPN": "country_jp", "THA": "country_th",
        "IRL": "country_372", "USA": "country_us", "MEX": "country_mx", "PHL": "country_ph",
        "DEU": "country_de", "GBR": "country_gb", "FRA": "country_fr", "ITA": "country_380",
        "CZE": "country_203", "IND": "country_in"}

TRADE_PROMPT = """
You are a supply-chain analyst. Estimate semiconductor (HS 8541, 8542) import signals for:
Importing country: "{country}"
As of month: {month}
Return JSON with fields:
{{
  "inventory_days": float,  // 5-90 typical; 30 means imports this month equal the median of the previous three months
  "past_delay_days": float  // 0-60 typical; 0 unless imports fell below that median
}}
Be conservative. If unsure, provide plausible conservative values.
"""
NEWS_PROMPT = """
You are analyzing news snippets about supply-chain disruptions in {country} for the last week of {month}. From the provided texts, compute:
- news_vol_7d: count of distinct relevant disruption mentions in last 7 days
- neg_tone_frac_3d: fraction [0..1] of negative-toned mentions in last 3 days
- strike_flag_7d: 1 if any strike/unrest detected in last 7 days else 0
Return JSON with fields exactly: {{"news_vol_7d": int, "neg_tone_frac_3d": float, "strike_flag_7d": int}}
Texts:
[]
"""
GSCPI_PROMPT = """
What was the NY Fed Global Supply Chain Pressure Index (GSCPI) value for {month}?
Return strict JSON: {{"global_risk": float, "timestamp": "YYYY-MM"}}.
"""
MONO_PROMPT = """
You are a supply-chain risk analyst. For importing country "{country}", month {month}, semiconductor
(HS 8541, 8542) imports, return one JSON object with exactly these fields:
{{"inventory_days": float (5-90; 30 = imports equal the prior three-month median),
  "past_delay_days": float (0-60),
  "news_vol_7d": int (disruption-related news events in the last week of the month),
  "neg_tone_frac_3d": float (0-1, negative share of that coverage, last 3 days),
  "strike_flag_7d": int (0/1, strike or labour unrest in the last week),
  "global_risk": float (NY Fed GSCPI value for that month)}}
"""


def parse(text: str) -> dict | None:
    try:
        return json.loads(re.sub(r"```json|```", "", text or "").strip())
    except Exception:
        match = re.search(r"\{.*\}", text or "", re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return None
    return None


MIN_INTERVAL_S = 3.0  # free tier: 8k tokens/min at ~300 tokens/call
_last_call = [0.0]


def call(llm, prompt: str) -> tuple[dict | None, float, str]:
    last = "unknown"
    for attempt in range(6):
        wait = _last_call[0] + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()
        t0 = time.perf_counter()
        try:
            text = llm(prompt)
            return parse(text), time.perf_counter() - t0, "ok"
        except Exception as exc:  # rate limits: back off and retry
            last = type(exc).__name__
            if "Authentication" in last:
                raise
            time.sleep(20 * (attempt + 1) if "RateLimit" in last else 5)
    return None, float("nan"), f"error: {last}"


def load_panel() -> pd.DataFrame:
    panel = pd.read_csv(PANEL)
    panel["month"] = panel["month"].astype(str)
    panel = panel.sort_values(["host_country_id", "month"])
    v = panel.groupby("host_country_id")["inbound_flow_usd"]
    with np.errstate(divide="ignore", invalid="ignore"):
        panel["T1"] = np.log(v.shift(-1) / panel["inbound_flow_usd"].where(panel["inbound_flow_usd"] > 0))
    return panel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap country-months (debug)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--summarise-only", action="store_true",
                    help="rebuild the summary from the saved calls without calling the LLM")
    args = ap.parse_args()
    if args.summarise_only:
        summarise(pd.read_csv(OUT / "extraction_alignment_calls.csv"), load_panel())
        return 0

    from orchestrator.utils.api_clients import GROQ_MODEL, llm_generate
    panel = load_panel()
    names = pd.read_csv(UNIVERSE).set_index("iso3")["country_name"].to_dict()
    node_name = {NODE[k]: names[k] for k in NODE}

    # 18 countries x six 2024 months (free-tier token budget: ~115k tokens)
    months = ["2024-01", "2024-03", "2024-05", "2024-07", "2024-09", "2024-11"]
    cells = panel[panel["month"].isin(months)][["host_country_id", "month"]].values.tolist()
    if args.limit:
        cells = cells[: args.limit]
    rng = np.random.default_rng(3)
    stab = [tuple(c) for c in rng.permutation(np.array(cells, dtype=object))[:10]]

    rows = []
    try:
        for i, (node, month) in enumerate(cells):
            country = node_name[node]
            reps = args.repeats if (node, month) in stab else 1
            for rep in range(reps):
                t, lt, st = call(llm_generate, TRADE_PROMPT.format(country=country, month=month))
                n, ln, sn = call(llm_generate, NEWS_PROMPT.format(country=country, month=month))
                m, lm, sm = call(llm_generate, MONO_PROMPT.format(country=country, month=month))
                rows.append({"node_id": node, "month": month, "rep": rep, "arm": "per_agent",
                             "latency_s": lt + ln, "status": f"{st}|{sn}", **(t or {}), **(n or {}),
                             "valid": t is not None and n is not None})
                rows.append({"node_id": node, "month": month, "rep": rep, "arm": "monolithic",
                             "latency_s": lm, "status": sm, **(m or {}), "valid": m is not None})
            print(f"{i + 1}/{len(cells)} {country} {month}", flush=True)
        for month in sorted(panel["month"].unique()):
            g, lg, sg = call(llm_generate, GSCPI_PROMPT.format(month=month))
            rows.append({"node_id": "global", "month": month, "rep": 0, "arm": "gscpi_agent",
                         "latency_s": lg, "status": sg, **(g or {}), "valid": g is not None})
    except Exception as exc:
        if "Authentication" in type(exc).__name__:
            print("GROQ_API_KEY rejected (401). Put a valid key in backend/.env and rerun.")
            return 2
        raise
    calls = pd.DataFrame(rows)
    calls["model"] = GROQ_MODEL
    calls.to_csv(OUT / "extraction_alignment_calls.csv", index=False)
    summarise(calls, panel)
    return 0


def summarise(calls: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    calls = calls.copy()
    calls["month"] = calls["month"].astype(str)
    stab = set(map(tuple, calls.loc[calls["rep"] > 0, ["node_id", "month"]].drop_duplicates().to_numpy()))
    fields = {"inventory_days": "inventory_days_proxy", "past_delay_days": "trade_delay_proxy",
              "news_vol_7d": "news_vol_7d", "neg_tone_frac_3d": "neg_tone_frac_3d",
              "strike_flag_7d": "strike_flag_7d", "global_risk": "global_risk"}
    truth = panel.set_index(["host_country_id", "month"])[[*fields.values(), "T1"]].add_suffix("_pub")
    truth.index.names = ["node_id", "month"]
    summary = []
    first = calls[calls["rep"] == 0]
    for arm in ("per_agent", "monolithic"):
        sub = first[(first["arm"] == arm) & (first["node_id"] != "global")].set_index(["node_id", "month"])
        joined = sub.join(truth, how="inner")
        for f_llm, f_pub in fields.items():
            if f_llm not in joined or joined[f_llm].isna().all():
                continue
            pair = joined[[f_llm, f"{f_pub}_pub", "T1_pub"]].apply(pd.to_numeric, errors="coerce")
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
            if len(pair) < 5:
                continue
            x, y_pub, t1 = pair[f_llm], pair[f"{f_pub}_pub"], pair["T1_pub"]
            rec = {"arm": arm, "field": f_llm, "n": len(pair),
                   "distinct_values": int(x.nunique()),
                   "spearman_vs_published": spearmanr(x, y_pub).statistic if x.nunique() > 1 else np.nan,
                   "mae_vs_published": float((x - y_pub).abs().mean()),
                   "spearman_llm_vs_T1": spearmanr(x, t1).statistic if x.nunique() > 1 else np.nan,
                   "spearman_published_vs_T1": spearmanr(y_pub, t1).statistic,
                   "llm_mean": float(x.mean()), "published_mean": float(y_pub.mean())}
            if f_llm == "strike_flag_7d":
                rec["agreement"] = float((x.round() == y_pub).mean())
            # alignment without requiring a valid T1, split into the part that
            # tracks which country it is and the part that tracks the month
            full = joined[[f_llm, f"{f_pub}_pub"]].apply(pd.to_numeric, errors="coerce").dropna()
            full.columns = ["llm", "pub"]
            country = [i[0] for i in full.index]
            means = full.groupby(country).mean()
            dev = full - full.groupby(country).transform("mean")
            rec["n_alignment"] = len(full)
            rec["spearman_all_rows"] = spearmanr(full["llm"], full["pub"]).statistic if full["llm"].nunique() > 1 else np.nan
            rec["spearman_between_country"] = spearmanr(means["llm"], means["pub"]).statistic if means["llm"].nunique() > 1 else np.nan
            rec["spearman_within_country"] = spearmanr(dev["llm"], dev["pub"]).statistic if dev["llm"].abs().sum() > 0 else np.nan
            summary.append(rec)
        rep = calls[(calls["arm"] == arm) & pd.Series(list(zip(calls["node_id"], calls["month"]))).isin(stab).to_numpy()]
        for f_llm in fields:
            if f_llm in rep and rep[f_llm].notna().any():
                vals = rep.groupby(["node_id", "month"])[f_llm].agg(["mean", "std"]).dropna()
                cv = (vals["std"] / vals["mean"].abs().replace(0, np.nan)).dropna()
                summary.append({"arm": arm, "field": f_llm,
                                "stability_cv_median": float(cv.median()) if len(cv) else np.nan,
                                "stability_identical_share": float((vals["std"] < 1e-12).mean()) if len(vals) else np.nan,
                                "stability_cells": len(vals)})
        arm_rows = first[first["arm"] == arm]
        summary.append({"arm": arm, "field": "_all", "calls": len(arm_rows),
                        "valid_json_rate": float(arm_rows["valid"].astype(bool).mean()),
                        "latency_median_s": float(arm_rows["latency_s"].median())})
    g = calls[calls["arm"] == "gscpi_agent"].set_index("month")
    gt = panel.drop_duplicates("month").set_index("month")["global_risk"]
    llm = pd.to_numeric(g["global_risk"], errors="coerce") if "global_risk" in g else pd.Series(dtype=float)
    pair = pd.concat([llm, gt], axis=1, keys=["llm", "pub"]).dropna()
    rec = {"arm": "gscpi_agent", "field": "global_risk", "calls": int(len(g)),
           "valid_json_rate": float(g["valid"].astype(bool).mean()) if len(g) else np.nan,
           "numeric_answers": int(llm.notna().sum()), "n": len(pair)}
    if len(pair) >= 5:
        rec.update({"spearman_vs_published": spearmanr(pair["llm"], pair["pub"]).statistic,
                    "pearson_vs_published": float(np.corrcoef(pair["llm"], pair["pub"])[0, 1]),
                    "mae_vs_published": float((pair["llm"] - pair["pub"]).abs().mean())})
    summary.append(rec)
    out = pd.DataFrame(summary)
    out.to_csv(OUT / "extraction_alignment_summary.csv", index=False)
    print(out.round(3).to_string(index=False))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
