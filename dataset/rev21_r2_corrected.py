#!/usr/bin/env python3
"""Round-2 correction: the original benchmark re-scored without the unreported-month artefacts.

Vietnam reported no monthly imports to Comtrade for 2024; the fusion stored those months as
zero inbound value (see rev21_r2_missing.py). On the overlapping target c (T0) seven rows are
artefacts: Vietnam 2023-12 (validation) and 2024-01..06 (test), each with c = -1 because the
next month is "zero". Their true value is unobservable, so the corrected protocol drops them,
exactly as the trainer already drops every other row whose target cannot be observed.

Nothing is retrained. The tabular models are refit by the same harness (their training
months, 2021-12..2022-12, contain no unreported month, so the fits are unchanged); graph and
tree models keep their saved predictions and are re-scored. Every number below is therefore
the published model scored on the rows that have a target.

Outputs (results/paper21_revision/round2/):
  corrected_reg.csv, corrected_clf.csv, corrected_boot.csv       Tables reg, clf, boot
  ensemble_check.csv, seed_spread.csv, pairwise_tests.csv        Table signif (rev21_significance)
  corrected_ablation.csv, corrected_rolling_blocks.csv,
  corrected_rolling_pooled.csv, corrected_refit.csv,
  corrected_overlap.csv                                          Tables ablation, nonoverlap, rolling
  corrected_graph_rolling.csv                                    graph rows of Table rolling
  corrected_generalisation.csv                                   Table general
  corrected_misc.csv                                             target summary, validation year
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score, r2_score,
                             recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rev21_r2_missing as rm  # noqa: E402
import rev21_tabular as tab  # noqa: E402

BASE = Path(__file__).resolve().parent
REV = BASE.parent / "results" / "paper21_revision"
OUT = REV / "round2"
SEEDS = (7, 13, 21, 42, 101)
GRAPH = ("gcn", "tgn", "tgn_no_memory")
tab.NODES = BASE / "data/four_year_2021_2024/processed/nodes_monthly.csv"


def artefacts() -> set[tuple[str, str]]:
    f = rm.flags(tab.load())
    return set(map(tuple, f.loc[f["artefact"] & f["T0"].notna(), ["host_country_id", "month"]].to_numpy()))


ART = artefacts()


def clean_load() -> pd.DataFrame:
    f = tab.load()
    bad = [(c, m) in ART for c, m in zip(f["host_country_id"], f["month"])]
    f.loc[bad, ["T0", "label"]] = np.nan
    return f


def is_art(df: pd.DataFrame, node: str = "node_id") -> np.ndarray:
    return np.array([(c, m) in ART for c, m in zip(df[node], df["month"].astype(str))])


def saved(task: str, split: str = "test") -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        for p in (REV / "benchmark_default" / f"seed_{seed}" / task / "predictions").glob("*.csv"):
            d = pd.read_csv(p)
            d = d[d["split"] == split].copy()
            d["seed"] = seed
            frames.append(d)
    allp = pd.concat(frames, ignore_index=True)
    allp["month"] = allp["month"].astype(str)
    allp["artefact"] = is_art(allp)
    return allp


def ens(allp: pd.DataFrame) -> pd.DataFrame:
    e = allp.groupby(["model", "node_id", "month"], as_index=False).agg(y=("target", "first"), s=("score", "mean"),
                                                                       artefact=("artefact", "first"))
    return e


def reg_table() -> pd.DataFrame:
    e = ens(saved("regression"))
    rows = []
    for scope, q in (("as published", e), ("corrected", e[~e["artefact"]])):
        for m, g in q.groupby("model"):
            y, p = g["y"].to_numpy(float), g["s"].to_numpy(float)
            rows.append({"scope": scope, "model": m, "n": len(g), "r2": r2_score(y, p),
                         "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "mae": float(np.mean(np.abs(y - p))),
                         "pearson": float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else np.nan,
                         "direction_acc": float(np.mean(np.sign(p) == np.sign(y)))})
    return pd.DataFrame(rows)


def clf_table() -> pd.DataFrame:
    e = ens(saved("classification"))
    rows = []
    for scope, q in (("as published", e), ("corrected", e[~e["artefact"]])):
        for m, g in q.groupby("model"):
            y, s = g["y"].astype(int).to_numpy(), g["s"].to_numpy(float)
            hit = (s >= 0.5).astype(int)
            rows.append({"scope": scope, "model": m, "n": len(g), "positives": int(y.sum()), "prevalence": y.mean(),
                         "pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s) if s.std() > 0 else 0.5,
                         "f1": f1_score(y, hit, zero_division=0), "precision": precision_score(y, hit, zero_division=np.nan),
                         "recall": recall_score(y, hit, zero_division=0), "accuracy_always_negative": 1 - y.mean()})
    return pd.DataFrame(rows)


def boot_table(reps: int = 5000) -> pd.DataFrame:
    """Row bootstrap as evidence/bootstrap_ci.py (5,000 resamples, seed 42), on the ensembles."""
    rows = []
    for task, models, metric in (("regression", ("ridge_regression", "gcn", "tgn"), r2_score),
                                 ("classification", ("tgn", "gcn", "logistic_regression"), average_precision_score)):
        e = ens(saved(task))
        for scope, q in (("as published", e), ("corrected", e[~e["artefact"]])):
            for m in models:
                g = q[q["model"] == m]
                y, s = g["y"].to_numpy(float), g["s"].to_numpy(float)
                if task == "classification":
                    y = y.astype(int)
                rng = np.random.default_rng(42)
                vals = []
                for _ in range(reps):
                    i = rng.integers(0, len(y), len(y))
                    if task == "classification" and y[i].sum() == 0:
                        continue
                    vals.append(metric(y[i], s[i]))
                rows.append({"task": task, "scope": scope, "model": m, "n": len(y), "point": metric(y, s),
                             "ci_lo": np.percentile(vals, 2.5), "ci_hi": np.percentile(vals, 97.5)})
    return pd.DataFrame(rows)


def significance() -> None:
    import rev21_significance as sig
    original = sig.read_task

    def read_clean(task: str) -> pd.DataFrame:
        p = original(task)
        return p[~is_art(p)].reset_index(drop=True)

    sig.read_task = read_clean
    sig.load = clean_load
    sig.OUT = OUT
    sig.main()


def graph_rolling() -> pd.DataFrame:
    rows = []
    for scope in ("as published", "corrected"):
        ys, ps = [], {g: [] for g in GRAPH}
        for block in ("2023H1", "2023H2", "2024H1", "2024H2"):
            frames = []
            for seed in (7, 13, 21):
                for g in GRAPH:
                    p = REV / "graph_rolling" / "T0" / block / f"seed_{seed}" / "predictions" / f"{g}.csv"
                    if p.exists():
                        d = pd.read_csv(p)
                        frames.append(d[d["split"] == "test"])
            allp = pd.concat(frames)
            allp["month"] = allp["month"].astype(str)
            if scope == "corrected":
                allp = allp[~is_art(allp)]
            w = allp.groupby(["model", "node_id", "month"]).agg(y=("target", "first"), s=("score", "mean")).reset_index()
            w = w.pivot_table(index=["node_id", "month"], columns="model", values="s").join(
                w.groupby(["node_id", "month"])["y"].first()).dropna()
            rec = {"scope": scope, "block": block, "n": len(w)}
            for g in GRAPH:
                rec[g] = r2_score(w["y"], w[g])
                ps[g].append(w[g].to_numpy())
            ys.append(w["y"].to_numpy())
            rows.append(rec)
        y = np.concatenate(ys)
        rows.append({"scope": scope, "block": "pooled", "n": len(y), **{g: r2_score(y, np.concatenate(ps[g])) for g in GRAPH}})
    return pd.DataFrame(rows)


def generalisation() -> pd.DataFrame:
    import rev21_generalisation as gen
    original = gen.basket_frame

    def clean_frame(nodes, inbound):
        f = original(nodes, inbound)
        f.loc[[(c, m) in ART for c, m in zip(f["host_country_id"], f["month"])], "T0"] = np.nan
        g = f.groupby("host_country_id", sort=False)
        f["T0_lag1"], f["T0_lag12"] = g["T0"].shift(1), g["T0"].shift(12)
        return f

    gen.basket_frame = clean_frame
    gen.OUT = OUT
    gen.main()
    return pd.read_csv(OUT / "generalisation_by_hs.csv")


def misc() -> pd.DataFrame:
    rows = []
    f0, f1 = tab.load(), clean_load()
    for scope, f in (("as published", f0), ("corrected", f1)):
        v = f["T0"].dropna()
        rows.append({"item": "T0 over all valid rows", "scope": scope, "n": len(v), "median": v.median(), "sd": v.std()})
        te = tab.months_between(f, tab.TEST) & f["T0"].notna()
        rows.append({"item": "test positives (tau 0.20)", "scope": scope, "n": int(te.sum()), "positives": int(f.loc[te, "label"].sum()),
                     "prevalence": f.loc[te, "label"].mean()})
        va = tab.months_between(f, tab.VALID) & f["T0"].notna()
        rows.append({"item": "validation positives", "scope": scope, "n": int(va.sum()), "prevalence": f.loc[va, "label"].mean()})
    for scope in ("as published", "corrected"):
        for task in ("regression",):
            e = ens(saved(task, "validation"))
            if scope == "corrected":
                e = e[~e["artefact"]]
            for m, g in e.groupby("model"):
                rows.append({"item": f"validation 2023 R2, {m}", "scope": scope, "n": len(g), "value": r2_score(g["y"], g["s"])})
    return pd.DataFrame(rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("artefact rows:", sorted(ART), flush=True)
    pd.set_option("display.width", 250, "display.max_rows", 400)
    for name, fn in (("corrected_reg", reg_table), ("corrected_clf", clf_table), ("corrected_misc", misc),
                     ("corrected_graph_rolling", graph_rolling), ("corrected_boot", boot_table)):
        t = fn()
        t.to_csv(OUT / f"{name}.csv", index=False)
        print(f"--- {name}\n{t.round(3).to_string(index=False)}", flush=True)
    f = clean_load()
    a = tab.part_a(f)
    a.to_csv(OUT / "corrected_ablation.csv", index=False)
    blocks, pooled = tab.part_b(f)
    blocks.to_csv(OUT / "corrected_rolling_blocks.csv", index=False)
    pooled.to_csv(OUT / "corrected_rolling_pooled.csv", index=False)
    refit = tab.part_refit(f)
    refit.to_csv(OUT / "corrected_refit.csv", index=False)
    ov = tab.part_c(f)
    ov.to_csv(OUT / "corrected_overlap.csv", index=False)
    print("--- ablation\n", a.round(3).to_string(index=False))
    print("--- rolling blocks\n", blocks.round(3).T.to_string())
    print("--- rolling pooled\n", pooled.round(3).T.to_string())
    print("--- refit\n", refit.round(3).to_string(index=False))
    print("--- overlap\n", ov.round(3).to_string(index=False), flush=True)
    print("--- significance", flush=True)
    significance()
    print("--- generalisation", flush=True)
    g = generalisation()
    g.to_csv(OUT / "corrected_generalisation.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
