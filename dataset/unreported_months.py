#!/usr/bin/env python3
"""The impact of missing data.

Comtrade has no monthly report for some reporter-months (Vietnam 2024-2025, China 2025,
Cambodia 2025, Estonia 2019-12, Philippines 2025-10). The fusion stores such a month as
zero inbound value. Consequences:
  * T0 (contraction against the 12-month median): a row whose NEXT month is unreported gets
    c = -1, a full collapse, and label 1 at every tau. These are artefacts, not disruptions.
  * T1 (one-step log change) is undefined when either month is zero, so those rows drop out.
  * Trade proxies of an unreported month hit their clip bounds (inventory 5, delay 30).

Part 1, original 18-country panel, test 2024 (the headline benchmark): every model, scored on
all test rows and on the rows without artefacts. Tabular models are refit (the training months
contain no unreported month); graph and tree models use the saved five-seed predictions, so
only the scoring changes.
Part 2, extended panels: counts of affected rows, and the main T1 contrasts of the ten-block
evaluation under alternative treatments of the gaps.

Writes results/benchmark/corrected/missing_*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extended_rolling_blocks as pda  # noqa: E402
import tabular_analyses as tab  # noqa: E402
from dm_tests import wild_cluster_boot  # noqa: E402

BASE = Path(__file__).resolve().parent
REV = BASE.parent / "results" / "benchmark"
OUT = REV / "corrected"


def unreported(f: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, f.loc[f["inbound_flow_usd"] <= 0, ["host_country_id", "month"]].to_numpy()))


def flags(f: pd.DataFrame) -> pd.DataFrame:
    gap = unreported(f)
    nxt = (pd.PeriodIndex(f["month"], freq="M") + 1).astype(str)
    f = f.copy()
    f["gap_now"] = [(c, m) in gap for c, m in zip(f["host_country_id"], f["month"])]
    f["gap_next"] = [(c, m) in gap for c, m in zip(f["host_country_id"], nxt)]
    f["artefact"] = f["gap_now"] | f["gap_next"]
    return f


def ensemble(folder_glob: str, split: str = "test") -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted((REV / "benchmark_default").glob(folder_glob))]
    allp = pd.concat(frames)
    allp = allp[allp["split"] == split]
    allp["month"] = allp["month"].astype(str)
    ens = allp.groupby(["model", "node_id", "month"], as_index=False).agg(y=("target", "first"), score=("score", "mean"))
    wide = ens.pivot_table(index=["node_id", "month"], columns="model", values="score")
    wide["y"] = ens.groupby(["node_id", "month"])["y"].first()
    return wide.reset_index()


def part1() -> pd.DataFrame:
    tab.NODES = BASE / "data/four_year_2021_2024/processed/nodes_monthly.csv"
    f = flags(tab.load())
    rows = []
    te_all = tab.months_between(f, tab.TEST) & f["T0"].notna()
    clean = te_all & ~f["artefact"]
    tr = tab.months_between(f, tab.TRAIN) & f["T0"].notna()
    print(f"original panel: unreported months {sorted(unreported(f))[:3]}... ({len(unreported(f))}); "
          f"test rows {int(te_all.sum())}, artefact rows {int((te_all & f['artefact']).sum())}, "
          f"positives {int(f.loc[te_all, 'label'].sum())} of which artefacts {int(f.loc[te_all & f['artefact'], 'label'].sum())}", flush=True)
    sets = {"Persistence": None, "Own lag c_{T-1} (AR(1))": ["T0_lag1"], "Trade proxies only": tab.TRADE,
            "Exogenous only": tab.EXO, "All seven features (ridge)": tab.ALL7}
    for name, feats in sets.items():
        for scope, te in (("all test rows", te_all), ("without artefacts", clean)):
            y = f.loc[te, "T0"].to_numpy(float)
            p = f.loc[te, "T0_lag1"].fillna(0.0).to_numpy(float) if feats is None else tab.fit_predict(f, feats, "T0", tr, te)
            pr = np.nan
            if feats is not None:
                pr = tab.pr_auc(f, feats, tr & f["label"].notna(), te)
            rows.append({"part": "original panel, test 2024", "target": "T0", "model": name, "scope": scope,
                         "n": int(te.sum()), "positives": int(f.loc[te, "label"].sum()), "r2": r2_score(y, p), "pr_auc": pr})
    art = f.loc[f["artefact"], ["host_country_id", "month"]].assign(artefact=True).rename(columns={"host_country_id": "node_id"})
    for task, metric in (("regression", "r2"), ("classification", "pr_auc")):
        w = ensemble(f"seed_*/{task}/predictions/*.csv").merge(art, on=["node_id", "month"], how="left")
        w["artefact"] = w["artefact"].fillna(False).astype(bool)
        models = [c for c in w.columns if c not in ("node_id", "month", "y", "artefact")]
        for scope, q in (("all test rows", w), ("without artefacts", w[~w["artefact"]])):
            for m in models:
                s = q[["y", m]].dropna()
                val = r2_score(s["y"], s[m]) if task == "regression" else average_precision_score(s["y"].astype(int), s[m])
                rows.append({"part": "original panel, test 2024", "target": f"T0 {task} (saved 5-seed predictions)", "model": m,
                             "scope": scope, "n": len(s), "positives": int(s["y"].sum()) if task == "classification" else np.nan,
                             metric: val})
    t1 = tab.months_between(f, tab.TEST)
    rows.append({"part": "original panel, test 2024", "target": "T1", "model": "(row count)", "scope": "T1 defined",
                 "n": int((t1 & f["T1"].notna()).sum()), "positives": np.nan})
    rows.append({"part": "original panel, test 2024", "target": "T1", "model": "(row count)", "scope": "T1 defined and an artefact row",
                 "n": int((t1 & f["T1"].notna() & f["artefact"]).sum()), "positives": np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    p1 = part1()
    p1.to_csv(OUT / "missing_original_panel.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 300)
    print(p1.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
