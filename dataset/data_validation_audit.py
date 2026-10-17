#!/usr/bin/env python3
"""Missing versus zero: is the rule for unreported months applied in every panel, basket and year?

Every panel stores a reporter-month that Comtrade has not published as zero inbound value; the
analysis treats it as missing. This audit checks the rule against the raw Comtrade replies, which
were requested for all thirteen HS codes (both baskets) and all twenty partners at once:

  1. Every stored zero, in every panel (original; consumer, 18 countries; 41 countries 2021-2024;
     41 countries 2019-2025) and basket, is classified from the raw replies as
       unreported  - no record for that reporter-month in any HS code from any requested partner;
       true zero   - the reporter has records that month (other basket or other partners), but none
                     in this basket from the panel partners.
  2. Partial reporting: positive months whose number of records in the basket is below half the
     reporter's median, with the month's log change.
  3. Scoring: no scored row of any table has an undefined target filled in (the log change must be
     NaN, never +-inf, on a gap), and no saved model prediction on the log change touches a gap.
     Predictions on the overlapping target that touch a gap are counted; they are the artefact rows
     that the corrected scoring removes. The bilateral pair table is checked the same way.

Writes results/benchmark/data_validation/{stored_zeros,coverage,partial_months,predictions,pair_table,ten_block_loader}.csv
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extended_rolling_blocks as pda  # noqa: E402
import tabular_analyses as tab  # noqa: E402

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "four_year_2021_2024"
CACHE = DATA / "cache" / "comtrade"
REV = BASE.parent / "results" / "benchmark"
OUT = REV / "data_validation"
SEMI = {"8541", "8542"}
PANELS = [
    ("original, 18 countries, 2021-2024", "processed", "semiconductor"),
    ("consumer basket, 18 countries, 2021-2024", "processed_v2/consumer", "consumer"),
    ("41 countries, 2021-2024", "processed_v3/semiconductor", "semiconductor"),
    ("41 countries, 2021-2024", "processed_v3/consumer", "consumer"),
    ("41 countries, 2019-2025", "processed_v4/semiconductor", "semiconductor"),
    ("41 countries, 2019-2025", "processed_v4/consumer", "consumer"),
]


def node_codes() -> dict[str, int]:
    cfg = yaml.safe_load(open(BASE / "config_extended_2019_2025.yaml"))
    found = {}

    def walk(x):
        if isinstance(x, dict):
            if "node_id" in x and "comtrade_code" in x:
                found[x["node_id"]] = int(x["comtrade_code"])
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(cfg)
    return found


def raw_index() -> tuple[dict, dict]:
    """(reporter, 'YYYY-MM') -> record counts by basket, and the set of requested reporter-months."""
    counts = defaultdict(lambda: {"semiconductor": 0, "consumer": 0})
    requested = defaultdict(set)
    for path in sorted(CACHE.glob("*/reporter_*.json")):
        j = json.load(open(path))
        if "request" not in j:
            continue
        rq = j["request"]
        rep = int(rq["reporter_code"])
        periods = rq.get("periods") or [rq.get("period")]
        codes = set(rq.get("commodity_codes") or ["8541", "8542"])
        for per in periods:
            m = f"{per[:4]}-{per[4:6]}"
            for b in ("semiconductor", "consumer"):
                if (b == "semiconductor" and codes & SEMI) or (b == "consumer" and codes - SEMI):
                    requested[(rep, m)].add(b)
        for r in j["response"].get("data") or []:
            m = f"{r['period'][:4]}-{r['period'][4:6]}"
            counts[(rep, m)]["semiconductor" if r["cmdCode"] in SEMI else "consumer"] += 1
    return counts, requested


def target_frame(rel: str) -> pd.DataFrame:
    """The panel as the analysis harness loads it (tabular_analyses.load), with its log change."""
    tab.NODES = DATA / rel / "nodes_monthly.csv"
    f = tab.load()[["month", "host_country_id", "inbound_flow_usd", "future_inbound_flow_usd", "T1"]].copy()
    f["month"] = f["month"].astype(str).str[:7]
    return f.reset_index(drop=True)


def seven_year_frames() -> pd.DataFrame:
    """The seven-year panel as the ten-block evaluation loads it (extended_rolling_blocks.frames)."""
    fr = pda.frames()
    f = pd.concat(fr.values(), ignore_index=True)
    f["month"] = f["month"].astype(str).str[:7]
    return f


def audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    codes = node_codes()
    counts, requested = raw_index()
    zeros, cover, partial = [], [], []
    for panel, rel, basket in PANELS:
        f = target_frame(rel)
        f["reporter"] = f["host_country_id"].map(codes)
        missing_code = sorted(set(f.loc[f["reporter"].isna(), "host_country_id"]))
        if missing_code:
            raise SystemExit(f"{panel}: no Comtrade code for {missing_code}")
        f["zero"] = f["inbound_flow_usd"] <= 0
        f["n_this"] = [counts[(int(r), m)][basket] if (int(r), m) in counts else 0 for r, m in zip(f["reporter"], f["month"])]
        f["n_any"] = [sum(counts[(int(r), m)].values()) if (int(r), m) in counts else 0 for r, m in zip(f["reporter"], f["month"])]
        f["asked"] = [basket in requested.get((int(r), m), set()) for r, m in zip(f["reporter"], f["month"])]
        usual = f[~f["zero"]].groupby("host_country_id")["n_any"].median()
        for _, r in f[f["zero"]].iterrows():
            kind = "unreported" if r["n_any"] == 0 else "true zero"
            zeros.append({"panel": panel, "basket": basket, "country": r["host_country_id"], "reporter": int(r["reporter"]),
                          "month": r["month"], "records_this_basket": int(r["n_this"]), "records_any_basket": int(r["n_any"]),
                          "requested": bool(r["asked"]), "median_records_in_reported_months": float(usual.get(r["host_country_id"], np.nan)),
                          "classification": kind})
        # partial reporting among positive months
        med = f[~f["zero"]].groupby("host_country_id")["n_this"].median()
        pos = f[~f["zero"]].copy()
        pos["median_records"] = pos["host_country_id"].map(med)
        low = pos[pos["n_this"] < 0.5 * pos["median_records"]]
        for _, r in low.iterrows():
            partial.append({"panel": panel, "basket": basket, "country": r["host_country_id"], "month": r["month"],
                            "records": int(r["n_this"]), "median_records": float(r["median_records"]),
                            "inbound_usd": float(r["inbound_flow_usd"]), "log_change_next": float(r["T1"])})
        # coverage and scoring checks by year
        nxt = (pd.PeriodIndex(f["month"], freq="M") + 1).astype(str)
        gap = set(zip(f.loc[f["zero"], "host_country_id"], f.loc[f["zero"], "month"]))
        f["gap_next"] = [(c, m) in gap for c, m in zip(f["host_country_id"], nxt)]
        touched = f["zero"] | f["gap_next"]
        for year, g in f.groupby(f["month"].str[:4]):
            t = touched[g.index]
            cover.append({"panel": panel, "basket": basket, "year": year, "country_months": len(g),
                          "countries": g["host_country_id"].nunique(),
                          "stored_zeros": int(g["zero"].sum()),
                          "unreported": int((g["zero"] & (g["n_any"] == 0)).sum()),
                          "true_zeros": int((g["zero"] & (g["n_any"] > 0)).sum()),
                          "rows_touching_a_gap": int(t.sum()),
                          "log_change_finite_on_gap_rows": int(np.isfinite(g.loc[t, "T1"]).sum()),
                          "log_change_infinite_anywhere": int(np.isinf(g["T1"]).sum()),
                          "partial_months": int(sum(1 for p in partial if p["panel"] == panel and p["basket"] == basket and p["month"][:4] == year))})
        print(f"{panel} / {basket}: {int(f['zero'].sum())} stored zeros, "
              f"{int((f['zero'] & (f['n_any'] == 0)).sum())} unreported, {len(low)} partial-looking months", flush=True)
    return pd.DataFrame(zeros), pd.DataFrame(cover), pd.DataFrame(partial)


def gaps_by_panel() -> dict[str, set]:
    out = {}
    for _, rel, _ in PANELS:
        f = target_frame(rel)
        out[rel] = set(zip(f.loc[f["inbound_flow_usd"] <= 0, "host_country_id"], f.loc[f["inbound_flow_usd"] <= 0, "month"]))
    return out


def predictions() -> pd.DataFrame:
    """Every saved prediction file: rows (test and validation) whose month or next month is a gap."""
    gaps = gaps_by_panel()
    sets = [  # results folder -> panel whose gaps apply
        ("benchmark_default", "processed", "T0"), ("graph_t1", "processed", "T1"),
        ("graph_rolling/T0", "processed", "T0"), ("graph_rolling/T1", "processed", "T1"),
        ("consumer/benchmark_default", "processed_v2/consumer", "T0"), ("consumer/graph_t1", "processed_v2/consumer", "T1"),
        ("consumer/graph_rolling/T0", "processed_v2/consumer", "T0"), ("consumer/graph_rolling/T1", "processed_v2/consumer", "T1"),
    ]
    for b in ("semiconductor", "consumer"):
        sets += [(f"extended_2021_2024/{b}/benchmark_default", f"processed_v3/{b}", "T0"),
                 (f"extended_2021_2024/{b}/graph_t1", f"processed_v3/{b}", "T1"),
                 (f"extended_2019_2025/{b}/graph_t1", f"processed_v4/{b}", "T1"),
                 (f"extended_2019_2025/{b}/graph_tune", f"processed_v4/{b}", "T1")]
    rows = []
    for folder, rel, target in sets:
        files = sorted((REV / folder).glob("**/predictions/*.csv"))
        if target == "T0":
            files = [p for p in files if "/classification/" not in str(p)]
        gap = gaps[rel]
        n = hit = 0
        for p in files:
            d = pd.read_csv(p, usecols=["node_id", "month", "split"])
            d = d[d["split"].isin(["test", "validation"])]
            m = d["month"].astype(str).str[:7]
            nxt = (pd.PeriodIndex(m, freq="M") + 1).astype(str)
            touch = [(c, a) in gap or (c, b) in gap for c, a, b in zip(d["node_id"], m, nxt)]
            n += len(d)
            hit += int(np.sum(touch))
        rows.append({"predictions": folder, "panel": rel, "target": target, "files": len(files),
                     "scored_rows": n, "rows_touching_a_gap": hit})
    return pd.DataFrame(rows)


def pair_check() -> pd.DataFrame:
    """The bilateral (pair-month) table: valid rows whose importer has an unreported current or next month."""
    import bilateral_diagnostics as bd
    f = bd.build()
    f["month"] = f["month"].astype(str).str[:7]
    nodes = target_frame("processed")
    gap = set(zip(nodes.loc[nodes["inbound_flow_usd"] <= 0, "host_country_id"], nodes.loc[nodes["inbound_flow_usd"] <= 0, "month"]))
    nxt = (pd.PeriodIndex(f["month"], freq="M") + 1).astype(str)
    touch = np.array([(d, m) in gap or (d, n) in gap for d, m, n in zip(f["destination"], f["month"], nxt)])
    rows = []
    for split, (a, b) in (("train", bd.TRAIN), ("validation", ("2023-01", "2023-12")), ("test", bd.TEST)):
        m = f["month"].between(a, b)
        rows.append({"split": split, "pair_rows": int(m.sum()), "valid_rows": int((m & f["valid"]).sum()),
                     "rows_touching_an_unreported_importer_month": int((m & touch).sum()),
                     "valid_rows_touching_one": int((m & touch & f["valid"]).sum()),
                     "pair_log_change_defined_on_them": int((m & touch & f["pair_T1"].notna()).sum())})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_rows", 300)
    zeros, cover, partial = audit()
    zeros.to_csv(OUT / "stored_zeros.csv", index=False)
    cover.to_csv(OUT / "coverage.csv", index=False)
    partial.to_csv(OUT / "partial_months.csv", index=False)
    sy = seven_year_frames()
    gap = set(zip(sy.loc[sy["inbound_flow_usd"] <= 0, "host_country_id"], sy.loc[sy["inbound_flow_usd"] <= 0, "month"]))
    nxt = (pd.PeriodIndex(sy["month"], freq="M") + 1).astype(str)
    touch = np.array([(c, m) in gap or (c, n) in gap for c, m, n in zip(sy["host_country_id"], sy["month"], nxt)])
    scored = sy["T1"].notna() & np.isfinite(sy["T1"])
    ten = {"loader": "ten half-year blocks (extended_rolling_blocks.frames)", "rows": len(sy), "rows_touching_a_gap": int(touch.sum()),
           "scored_rows_touching_a_gap": int((scored & touch).sum()), "infinite_targets": int(np.isinf(sy["T1"]).sum())}
    pd.DataFrame([ten]).to_csv(OUT / "ten_block_loader.csv", index=False)
    print(ten)
    pairs = pair_check()
    pairs.to_csv(OUT / "pair_table.csv", index=False)
    print(pairs.to_string(index=False))
    pred = predictions()
    pred.to_csv(OUT / "predictions.csv", index=False)
    print(zeros.groupby(["panel", "basket", "classification"]).size().to_string())
    print(cover.to_string(index=False))
    print(partial.to_string(index=False))
    print(pred.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
