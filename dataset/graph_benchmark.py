"""Graph models (GCN, TGN, TGN-no-memory) under current vs improved preprocessing.

Reuses the repository's own ``SnapshotGCN`` and ``TemporalGraphNetwork`` so the
comparison is against the same architectures the paper reports, not a reimplementation.

Discipline matches tuned_benchmark.py:
  * fit on train months (2021-12..2022-12);
  * pick the training epoch by 2023 validation metric (early-stopping selection);
  * report the selected epoch once on 2024 test.

Feature sets:
  * paper    : the seven Table-'tab:features' fields (protective inventory flipped).
  * improved : flow ratio + past-only causal lags (contraction_lag1, flow_ratio_lag1).
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
TAU = 0.20
PAPER = ["inventory_days_proxy", "trade_delay_proxy", "news_vol_7d", "neg_tone_frac_3d",
         "strike_flag_7d", "weather_anomaly_7d", "global_risk"]
PROTECTIVE = "inventory_days_proxy"
DEVICE = torch.device("cpu")
EPOCHS, PATIENCE, SEED = 40, 10, 0
RESULTS = Path(__file__).parent / "graph_benchmark_results.txt"


def load():
    nodes = pd.read_csv(DATA / "nodes_monthly.csv")
    nodes["month"] = nodes["month"].astype(str)
    nodes["node_id"] = nodes["node_id"].astype(str)
    edges = pd.read_csv(DATA / "edges_monthly.csv")
    edges["month"] = edges["month"].astype(str)
    edges["source"] = edges["source"].astype(str)
    edges["destination"] = edges["destination"].astype(str)
    return nodes.sort_values(["host_country_id", "month"]).reset_index(drop=True), edges


def add_improved(nodes: pd.DataFrame) -> pd.DataFrame:
    out = nodes.copy()
    out["flow_ratio"] = out["inventory_days_proxy"] / 30.0
    grouped = out.groupby("host_country_id", sort=False)
    out["contraction_lag1"] = grouped["contraction"].shift(1).fillna(0.0)
    out["flow_ratio_lag1"] = grouped["flow_ratio"].shift(1).fillna(1.0)
    return out


def in_window(series, bounds):
    return series.between(bounds[0], bounds[1])


def build_batches(nodes, edges, features):
    node_ids = sorted(nodes["node_id"].unique())
    index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    months = sorted(nodes["month"].unique())

    train_rows = nodes[in_window(nodes["month"], TRAIN)]
    matrix = train_rows[features].astype(float).replace([np.inf, -np.inf], np.nan)
    if PROTECTIVE in features:
        matrix[PROTECTIVE] = -matrix[PROTECTIVE]
    mean = matrix.mean().to_numpy()
    scale = matrix.std().replace(0.0, 1.0).fillna(1.0).to_numpy()

    edge_scale = 1.0
    tr_edges = edges[in_window(edges["month"], TRAIN)]
    if len(tr_edges):
        edge_scale = max(float(np.log1p(pd.to_numeric(tr_edges["trade_value_usd"], errors="coerce")
                                        .fillna(0).clip(lower=0)).max()), 1.0)

    by_month_nodes = {m: g for m, g in nodes.groupby("month")}
    by_month_edges = {m: g for m, g in edges.groupby("month")}

    batches = {}
    for m in months:
        rows = by_month_nodes.get(m)
        ordered = rows.drop_duplicates("node_id").set_index("node_id").reindex(node_ids)
        feats = ordered[features].astype(float).to_numpy()
        if PROTECTIVE in features:
            feats[:, features.index(PROTECTIVE)] *= -1.0
        feats = np.nan_to_num((feats - mean) / scale)
        contraction = pd.to_numeric(ordered["contraction"], errors="coerce").to_numpy(float)
        if "target_valid" in ordered:
            valid = pd.to_numeric(ordered["target_valid"], errors="coerce").fillna(0).to_numpy(bool).copy()
        else:
            valid = ~np.isnan(contraction)
        valid = valid & ~np.isnan(contraction)

        events, src, dst = [], [], []
        me = by_month_edges.get(m)
        if me is not None:
            trade = np.log1p(pd.to_numeric(me["trade_value_usd"], errors="coerce").fillna(0).clip(lower=0)) / edge_scale
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
            fwd = torch.tensor([src, dst], dtype=torch.long, device=DEVICE)
            edge_index = torch.cat([fwd, fwd.flip(0)], dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=DEVICE)

        batches[m] = {
            "features": torch.tensor(feats, dtype=torch.float32, device=DEVICE),
            "edge_index": edge_index,
            "events": events,
            "time": float(pd.Period(m, freq="M").ordinal),
            "target": torch.tensor(np.nan_to_num(contraction), dtype=torch.float32, device=DEVICE),
            "labels": torch.tensor(np.nan_to_num((contraction < -TAU).astype(float)), dtype=torch.float32, device=DEVICE),
            "mask": torch.tensor(valid, dtype=torch.bool, device=DEVICE),
        }
    return batches, months, n


def split_months(months):
    return ([m for m in months if TRAIN[0] <= m <= TRAIN[1]],
            [m for m in months if VALID[0] <= m <= VALID[1]],
            [m for m in months if TEST[0] <= m <= TEST[1]])


def positive_weight(batches, train_months):
    pos = tot = 0
    for m in train_months:
        b = batches[m]; mask = b["mask"]
        pos += float(b["labels"][mask].sum()); tot += int(mask.sum())
    prev = pos / tot if tot else 0.0
    return min(1.0 / prev, 1e4) if prev > 0 else 1.0


def run_model(kind, task, batches, months, n, feat_dim):
    torch.manual_seed(SEED); np.random.seed(SEED)
    activation = "linear" if task == "regression" else "sigmoid"
    if kind == "gcn":
        model = SnapshotGCN(feature_dim=feat_dim, hidden_dim=32, layers=2, output_activation=activation)
    else:
        model = TemporalGraphNetwork(feature_dim=feat_dim, memory_dim=32, edge_dim=2,
                                     use_memory=(kind == "tgn"), output_activation=activation)
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    train_m, val_m, test_m = split_months(months)
    pw = torch.tensor(positive_weight(batches, train_m), device=DEVICE)

    def predict_pass(collect):
        preds = {}
        if kind != "gcn":
            model.reset_memory(n, DEVICE)
        for m in months:
            b = batches[m]
            if kind == "gcn":
                out = model(b["features"], b["edge_index"])
            else:
                out = model(b["features"], b["events"], b["time"])
            if m in collect:
                preds[m] = out
        return preds

    best_val, best_state, best_test, wait = -np.inf, None, None, 0
    for _ in range(EPOCHS):
        model.train()
        if kind != "gcn":
            model.reset_memory(n, DEVICE)
        opt.zero_grad()
        losses = []
        for m in months:  # chronological; loss only on train months
            b = batches[m]
            out = model(b["features"], b["edge_index"]) if kind == "gcn" \
                else model(b["features"], b["events"], b["time"])
            if m not in train_m:
                continue
            mask = b["mask"]
            if mask.sum() == 0:
                continue
            if task == "regression":
                losses.append(((out[mask] - b["target"][mask]) ** 2).mean())
            else:
                losses.append(torch.nn.functional.binary_cross_entropy(
                    out[mask].clamp(1e-6, 1 - 1e-6), b["labels"][mask],
                    weight=torch.where(b["labels"][mask] > 0, pw, torch.tensor(1.0))))
        if not losses:
            break
        torch.stack(losses).mean().backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            preds = predict_pass(set(val_m) | set(test_m))
        vy, vp = gather(batches, val_m, preds)
        ty, tp = gather(batches, test_m, preds)
        score = metric(task, vy, vp)
        if score > best_val:
            best_val, best_test, wait = score, metric(task, ty, tp), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best_val, best_test


def gather(batches, month_list, preds):
    ys, ps = [], []
    for m in month_list:
        b = batches[m]; mask = b["mask"].cpu().numpy()
        target = (b["target"] if True else None)
        ys.append(np.where(mask, b["target"].cpu().numpy(), np.nan))
        ps.append(np.where(mask, preds[m].cpu().numpy(), np.nan))
    y = np.concatenate(ys); p = np.concatenate(ps)
    ok = ~np.isnan(y)
    return y[ok], p[ok]


def metric(task, y, p):
    if len(y) == 0:
        return -np.inf
    if task == "regression":
        return r2_score(y, p)
    c = (y < -TAU).astype(int)
    if c.max() == 0:
        return -np.inf
    return average_precision_score(c, p)


def emit(line: str):
    print(line, flush=True)
    with open(RESULTS, "a") as handle:
        handle.write(line + "\n")


def main():
    open(RESULTS, "w").close()
    nodes, edges = load()
    variants = {"paper": (nodes, PAPER),
                "improved": (add_improved(nodes), ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"])}
    for task in ("regression", "classification"):
        unit = "R2" if task == "regression" else "PR-AUC"
        emit(f"\n=========== GRAPH MODELS: {task.upper()} (test {unit}) ===========")
        emit(f"{'model':16s} {'features':9s} | {'val':>7s} {'test':>7s}")
        emit("-" * 46)
        for fname, (frame, feats) in variants.items():
            batches, months, n = build_batches(frame, edges, feats)
            for kind, label in (("gcn", "GCN"), ("tgn", "TGN"), ("tgn_no_mem", "TGN no-memory")):
                val, test = run_model(kind, task, batches, months, n, len(feats))
                emit(f"{label:16s} {fname:9s} | {val:7.3f} {test:7.3f}")
            emit("-" * 46)


if __name__ == "__main__":
    main()
