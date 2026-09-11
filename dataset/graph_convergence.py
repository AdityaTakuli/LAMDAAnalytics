"""Phase 0.7 C.1/C.2 - canonical converged graph benchmark, one target per invocation.

Usage: graph_convergence.py {T0|T1}

Matched budget across GCN / TGN / TGN-no-memory: 80 epochs, patience 15, 5 seeds,
best-val-epoch test R2, improved feature set. This is the SINGLE SOURCE for every
graph number in the paper (both targets), superseding the 40-epoch/single-seed rows
in changes.md 0.3.

Outputs (results/phase07/):
  graph_convergence_{target}.csv    mean, sd, per-seed test R2
  graph_preds_{target}.csv          long format: model, seed, host_country_id, month, y, pred
  (figures/graph_convergence_{target}.pdf/png)   loss + val curves, seed 0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_gcn import SnapshotGCN  # noqa: E402
from model_tgn import TemporalGraphNetwork  # noqa: E402

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).parent / "data/four_year_2021_2024/processed"
TRAIN, VALID, TEST = ("2021-12", "2022-12"), ("2023-01", "2023-12"), ("2024-01", "2024-11")
EPOCHS, PATIENCE, SEEDS = 80, 15, [0, 1, 2, 3, 4]
FEATURES = ["flow_ratio", "contraction_lag1", "flow_ratio_lag1"]
KINDS = (("gcn", "GCN"), ("tgn", "TGN"), ("tgn_no_mem", "TGN no-memory"))


def load():
    n = pd.read_csv(DATA / "nodes_monthly.csv")
    n["month"] = n["month"].astype(str); n["node_id"] = n["node_id"].astype(str)
    n = n.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    v = n.groupby("host_country_id", sort=False)["inbound_flow_usd"]
    fut = v.shift(-1)
    med12 = v.transform(lambda s: s.rolling(12, min_periods=1).median())
    n["T0"] = (fut - med12) / med12.replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = n["inbound_flow_usd"] > 0
        n["T1"] = np.log(fut / n["inbound_flow_usd"].where(pos))
        n.loc[fut <= 0, "T1"] = np.nan
    n["flow_ratio"] = n["inventory_days_proxy"] / 30.0
    g = n.groupby("host_country_id", sort=False)
    n["flow_ratio_lag1"] = g["flow_ratio"].shift(1)
    n["contraction_lag1"] = g["T0"].shift(1)
    e = pd.read_csv(DATA / "edges_monthly.csv")
    for c in ("month", "source", "destination"):
        e[c] = e[c].astype(str)
    return n, e


def build(nodes, edges, target):
    node_ids = sorted(nodes["node_id"].unique())
    index = {nid: i for i, nid in enumerate(node_ids)}
    nmap = nodes.drop_duplicates("node_id").set_index("node_id")["host_country_id"]
    n = len(node_ids)
    months = sorted(nodes["month"].unique())
    tr = nodes[nodes["month"].between(*TRAIN)]
    mean = tr[FEATURES].astype(float).mean().to_numpy()
    scale = tr[FEATURES].astype(float).std().replace(0.0, 1.0).fillna(1.0).to_numpy()
    te = edges[edges["month"].between(*TRAIN)]
    escale = max(float(np.log1p(pd.to_numeric(te["trade_value_usd"], errors="coerce").fillna(0).clip(lower=0)).max()), 1.0) if len(te) else 1.0
    bn = {m: g for m, g in nodes.groupby("month")}
    be = {m: g for m, g in edges.groupby("month")}
    batches = {}
    for m in months:
        rows = bn.get(m).drop_duplicates("node_id").set_index("node_id").reindex(node_ids)
        fx = np.nan_to_num((rows[FEATURES].astype(float).to_numpy() - mean) / scale)
        tgt = pd.to_numeric(rows[target], errors="coerce").to_numpy(float)
        valid = ((~np.isnan(tgt)) & (~np.isnan(rows[FEATURES].astype(float).to_numpy()).any(axis=1))).copy()
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
        batches[m] = {"features": torch.tensor(fx, dtype=torch.float32), "edge_index": ei,
                      "events": events, "time": float(pd.Period(m, freq="M").ordinal),
                      "target": torch.tensor(np.nan_to_num(tgt), dtype=torch.float32),
                      "mask": torch.tensor(valid, dtype=torch.bool)}
    return batches, months, n, node_ids, nmap


def make(kind):
    if kind == "gcn":
        return SnapshotGCN(feature_dim=len(FEATURES), hidden_dim=32, layers=2, output_activation="linear")
    return TemporalGraphNetwork(feature_dim=len(FEATURES), memory_dim=32, edge_dim=2,
                                use_memory=(kind == "tgn"), output_activation="linear")


def gather(batches, ms, preds):
    ys, ps = [], []
    for m in ms:
        b = batches[m]; mk = b["mask"].numpy()
        ys.append(b["target"].numpy()[mk]); ps.append(preds[m].detach().numpy()[mk])
    return np.concatenate(ys), np.concatenate(ps)


def train_seed(kind, seed, batches, months, n, node_ids, nmap):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make(kind)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    trm = [m for m in months if TRAIN[0] <= m <= TRAIN[1]]
    vam = [m for m in months if VALID[0] <= m <= VALID[1]]
    tem = [m for m in months if TEST[0] <= m <= TEST[1]]

    def full_pass():
        preds = {}
        if kind != "gcn":
            model.reset_memory(n, torch.device("cpu"))
        for m in months:
            b = batches[m]
            preds[m] = model(b["features"], b["edge_index"]) if kind == "gcn" else model(b["features"], b["events"], b["time"])
        return preds

    loss_curve, val_curve = [], []
    best_val, best_test, best_preds, wait = -np.inf, float("nan"), None, 0
    for _ in range(EPOCHS):
        model.train()
        if kind != "gcn":
            model.reset_memory(n, torch.device("cpu"))
        opt.zero_grad(); losses = []
        for m in months:
            b = batches[m]
            out = model(b["features"], b["edge_index"]) if kind == "gcn" else model(b["features"], b["events"], b["time"])
            if m in trm and b["mask"].sum() > 0:
                losses.append(((out[b["mask"]] - b["target"][b["mask"]]) ** 2).mean())
        if not losses:
            break
        loss = torch.stack(losses).mean(); loss.backward(); opt.step()
        loss_curve.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            preds = full_pass()
        vy, vp = gather(batches, vam, preds); ty, tp = gather(batches, tem, preds)
        vs = r2_score(vy, vp) if np.std(vy) > 1e-9 else -np.inf
        ts = r2_score(ty, tp) if np.std(ty) > 1e-9 else float("nan")
        val_curve.append(vs)
        if vs > best_val:
            best_val, best_test, wait = vs, ts, 0
            best_preds = {m: preds[m].detach().numpy().copy() for m in tem}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    recs = []
    for m in tem:
        mk = batches[m]["mask"].numpy()
        for i, ok in enumerate(mk):
            if ok:
                recs.append((nmap[node_ids[i]], m, float(batches[m]["target"].numpy()[i]), float(best_preds[m][i])))
    return best_test, loss_curve, val_curve, recs


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "T1"
    assert target in ("T0", "T1")
    (ROOT / "results/phase07").mkdir(parents=True, exist_ok=True)
    nodes, edges = load()
    batches, months, n, node_ids, nmap = build(nodes, edges, target)

    summary, curves, pred_rows = [], {}, []
    for kind, label in KINDS:
        tests = []
        for s in SEEDS:
            t, lc, vc, recs = train_seed(kind, s, batches, months, n, node_ids, nmap)
            tests.append(t)
            for (hc, mo, y, p) in recs:
                pred_rows.append({"target": target, "model": kind, "seed": s,
                                  "host_country_id": hc, "month": mo, "y": y, "pred": p})
            if s == 0:
                curves[label] = (lc, vc)
        arr = np.array(tests, float)
        summary.append({"target": target, "model": kind, "mean": arr.mean(), "sd": arr.std(),
                        "epochs": EPOCHS, "n_seeds": len(SEEDS),
                        "per_seed": " ".join(f"{x:.4f}" for x in tests)})
        print(f"{label:16s} | mean {arr.mean():.3f} sd {arr.std():.3f} | "
              + " ".join(f"{x:+.3f}" for x in tests), flush=True)

    pd.DataFrame(summary).to_csv(ROOT / f"results/phase07/graph_convergence_{target}.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(ROOT / f"results/phase07/graph_preds_{target}.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        for label, (lc, vc) in curves.items():
            a1.plot(range(1, len(lc) + 1), lc, label=label)
            a2.plot(range(1, len(vc) + 1), vc, label=label)
        a1.set_title(f"Train MSE ({target}, seed 0)"); a1.set_xlabel("epoch"); a1.set_ylabel("MSE"); a1.legend()
        a2.set_title(f"Validation $R^2$ ({target}, seed 0)"); a2.set_xlabel("epoch"); a2.set_ylabel("val $R^2$"); a2.legend()
        fig.tight_layout()
        fig.savefig(ROOT / f"figures/graph_convergence_{target}.pdf")
        fig.savefig(ROOT / f"figures/graph_convergence_{target}.png", dpi=150)
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)
    print(f"DONE {target}", flush=True)


if __name__ == "__main__":
    main()
