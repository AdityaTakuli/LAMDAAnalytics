"""Phase 0.3 - graph models (GCN, TGN, TGN-no-memory) under the overlap-free T1 target.

Mirrors graph_benchmark.py but predicts T1 = log(V_{T+1}/V_T) for regression and a
prevalence-matched log-return label for classification, so the T0 graph rankings
can be checked for preservation vs inversion. Improved feature set (flow_ratio,
contraction_lag1, flow_ratio_lag1). Same discipline: fit on train, select epoch on
2023 validation, report once on 2024 test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_gcn import SnapshotGCN  # noqa: E402
from model_tgn import TemporalGraphNetwork  # noqa: E402

DATA = Path(__file__).parent / "data/four_year_2021_2024/processed"
TRAIN, VALID, TEST = ("2021-12", "2022-12"), ("2023-01", "2023-12"), ("2024-01", "2024-11")
PREVALENCE = 0.104
DEVICE = torch.device("cpu")
EPOCHS, PATIENCE, SEED = 40, 10, 0
FEATURES = ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"]
RESULTS = Path(__file__).parent / "graph_benchmark_t1_results.txt"


def load():
    n = pd.read_csv(DATA / "nodes_monthly.csv")
    n["month"] = n["month"].astype(str)
    n["node_id"] = n["node_id"].astype(str)
    n = n.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    v = n.groupby("host_country_id", sort=False)["inbound_flow_usd"]
    fut = v.shift(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = n["inbound_flow_usd"] > 0
        n["T1"] = np.log(fut / n["inbound_flow_usd"].where(pos))
        n.loc[fut <= 0, "T1"] = np.nan
    n["flow_ratio"] = n["inventory_days_proxy"] / 30.0
    grp = n.groupby("host_country_id", sort=False)
    n["flow_ratio_lag1"] = grp["flow_ratio"].shift(1)
    n["contraction_lag1"] = grp["contraction"].shift(1)
    thr = float(n.loc[n["month"].between(*TRAIN) & n["T1"].notna(), "T1"].quantile(PREVALENCE))
    n["t1_label"] = (n["T1"] < thr).astype(float)
    n["t1_valid"] = n["T1"].notna() & n[FEATURES].notna().all(axis=1)
    e = pd.read_csv(DATA / "edges_monthly.csv")
    for c in ("month", "source", "destination"):
        e[c] = e[c].astype(str)
    return n, e


def in_window(s, b):
    return s.between(b[0], b[1])


def build(nodes, edges):
    node_ids = sorted(nodes["node_id"].unique())
    index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    months = sorted(nodes["month"].unique())
    tr = nodes[in_window(nodes["month"], TRAIN)]
    mean = tr[FEATURES].astype(float).mean().to_numpy()
    scale = tr[FEATURES].astype(float).std().replace(0.0, 1.0).fillna(1.0).to_numpy()
    escale = 1.0
    te = edges[in_window(edges["month"], TRAIN)]
    if len(te):
        escale = max(float(np.log1p(pd.to_numeric(te["trade_value_usd"], errors="coerce").fillna(0).clip(lower=0)).max()), 1.0)

    bn = {m: g for m, g in nodes.groupby("month")}
    be = {m: g for m, g in edges.groupby("month")}
    batches = {}
    for m in months:
        rows = bn.get(m).drop_duplicates("node_id").set_index("node_id").reindex(node_ids)
        feats = np.nan_to_num((rows[FEATURES].astype(float).to_numpy() - mean) / scale)
        t1 = pd.to_numeric(rows["T1"], errors="coerce").to_numpy(float)
        lab = pd.to_numeric(rows["t1_label"], errors="coerce").to_numpy(float)
        valid = pd.to_numeric(rows["t1_valid"], errors="coerce").fillna(0).to_numpy(bool).copy()
        valid = valid & ~np.isnan(t1)
        src, dst, events = [], [], []
        me = be.get(m)
        if me is not None:
            trade = np.log1p(pd.to_numeric(me["trade_value_usd"], errors="coerce").fillna(0).clip(lower=0)) / escale
            vol = np.log1p(pd.to_numeric(me.get("flow_volume", 0.0), errors="coerce").fillna(0).clip(lower=0))
            for k, (s, d) in enumerate(zip(me["source"], me["destination"])):
                si, di = index.get(s), index.get(d)
                if si is None or di is None:
                    continue
                src.append(si); dst.append(di)
                events.append({"source_index": si, "destination_index": di,
                               "time": float(pd.Period(m, freq="M").ordinal), "time_delta": 1.0,
                               "edge_features": [float(trade.iloc[k]), float(vol.iloc[k])]})
        if src:
            fwd = torch.tensor([src, dst], dtype=torch.long)
            ei = torch.cat([fwd, fwd.flip(0)], dim=1)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)
        batches[m] = {"features": torch.tensor(feats, dtype=torch.float32),
                      "edge_index": ei, "events": events, "time": float(pd.Period(m, freq="M").ordinal),
                      "target": torch.tensor(np.nan_to_num(t1), dtype=torch.float32),
                      "labels": torch.tensor(np.nan_to_num(lab), dtype=torch.float32),
                      "mask": torch.tensor(valid, dtype=torch.bool)}
    return batches, months, n


def split(months):
    return ([m for m in months if TRAIN[0] <= m <= TRAIN[1]],
            [m for m in months if VALID[0] <= m <= VALID[1]],
            [m for m in months if TEST[0] <= m <= TEST[1]])


def pos_weight(batches, tr):
    p = t = 0
    for m in tr:
        b = batches[m]; mk = b["mask"]
        p += float(b["labels"][mk].sum()); t += int(mk.sum())
    prev = p / t if t else 0
    return min(1.0 / prev, 1e4) if prev > 0 else 1.0


def gather(batches, ms, preds):
    ys, ps = [], []
    for m in ms:
        b = batches[m]; mk = b["mask"].numpy()
        ys.append(np.where(mk, b["target"].numpy(), np.nan))
        ps.append(np.where(mk, preds[m].numpy(), np.nan))
    y = np.concatenate(ys); p = np.concatenate(ps); ok = ~np.isnan(y)
    return y[ok], p[ok]


def gather_lab(batches, ms, preds):
    ys, ps = [], []
    for m in ms:
        b = batches[m]; mk = b["mask"].numpy()
        ys.append(np.where(mk, b["labels"].numpy(), np.nan))
        ps.append(np.where(mk, preds[m].numpy(), np.nan))
    y = np.concatenate(ys); p = np.concatenate(ps); ok = ~np.isnan(y)
    return y[ok], p[ok]


def run(kind, task, batches, months, n):
    torch.manual_seed(SEED); np.random.seed(SEED)
    act = "linear" if task == "regression" else "sigmoid"
    model = (SnapshotGCN(feature_dim=len(FEATURES), hidden_dim=32, layers=2, output_activation=act)
             if kind == "gcn" else
             TemporalGraphNetwork(feature_dim=len(FEATURES), memory_dim=32, edge_dim=2,
                                  use_memory=(kind == "tgn"), output_activation=act))
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    trm, vam, tem = split(months)
    pw = torch.tensor(pos_weight(batches, trm))

    def full_pass():
        preds = {}
        if kind != "gcn":
            model.reset_memory(n, DEVICE)
        for m in months:
            b = batches[m]
            preds[m] = model(b["features"], b["edge_index"]) if kind == "gcn" else model(b["features"], b["events"], b["time"])
        return preds

    best_val, best_test, wait = -np.inf, float("nan"), 0
    for _ in range(EPOCHS):
        model.train()
        if kind != "gcn":
            model.reset_memory(n, DEVICE)
        opt.zero_grad(); losses = []
        for m in months:
            b = batches[m]
            out = model(b["features"], b["edge_index"]) if kind == "gcn" else model(b["features"], b["events"], b["time"])
            if m not in trm:
                continue
            mk = b["mask"]
            if mk.sum() == 0:
                continue
            if task == "regression":
                losses.append(((out[mk] - b["target"][mk]) ** 2).mean())
            else:
                losses.append(torch.nn.functional.binary_cross_entropy(
                    out[mk].clamp(1e-6, 1 - 1e-6), b["labels"][mk],
                    weight=torch.where(b["labels"][mk] > 0, pw, torch.tensor(1.0))))
        if not losses:
            break
        torch.stack(losses).mean().backward(); opt.step()
        model.eval()
        with torch.no_grad():
            preds = full_pass()
        if task == "regression":
            vy, vp = gather(batches, vam, preds); ty, tp = gather(batches, tem, preds)
            vs = r2_score(vy, vp) if np.std(vy) > 1e-9 else -np.inf
            ts = r2_score(ty, tp) if np.std(ty) > 1e-9 else float("nan")
        else:
            vy, vp = gather_lab(batches, vam, preds); ty, tp = gather_lab(batches, tem, preds)
            vs = average_precision_score(vy, vp) if vy.max() > 0 else -np.inf
            ts = average_precision_score(ty, tp) if ty.max() > 0 else float("nan")
        if vs > best_val:
            best_val, best_test, wait = vs, ts, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best_val, best_test


def emit(line):
    print(line, flush=True)
    with open(RESULTS, "a") as h:
        h.write(line + "\n")


def main():
    open(RESULTS, "w").close()
    nodes, edges = load()
    batches, months, n = build(nodes, edges)
    for task in ("regression", "classification"):
        unit = "R2" if task == "regression" else "PR-AUC"
        emit(f"\n=== GRAPH MODELS under T1 (log return): {task.upper()} (test {unit}) ===")
        emit(f"{'model':16s} | {'val':>7s} {'test':>7s}")
        emit("-" * 34)
        for kind, label in (("gcn", "GCN"), ("tgn", "TGN"), ("tgn_no_mem", "TGN no-memory")):
            val, test = run(kind, task, batches, months, n)
            emit(f"{label:16s} | {val:7.3f} {test:7.3f}")
        emit("-" * 34)


if __name__ == "__main__":
    main()
