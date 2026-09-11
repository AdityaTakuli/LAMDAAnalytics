"""Phase 0.6 C.4 - converged graph benchmark under T1 (regression), to power the
negative claim that graph models do not beat a 3-feature linear AR on the clean target.

Matched compute budget across all three graph classes: 80 epochs, patience 15,
5 seeds. Reports mean +/- sd test R2 (best-val epoch) and saves train-loss / val-R2
convergence curves as evidence that the models were trained to convergence.

Reference: AR ridge R2 = 0.138 on the same T1 test observations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graph_benchmark_t1 import build, load, split  # noqa: E402
from model_gcn import SnapshotGCN  # noqa: E402
from model_tgn import TemporalGraphNetwork  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
EPOCHS, PATIENCE, SEEDS = 80, 15, [0, 1, 2, 3, 4]
FEATURE_DIM = 3
RESULTS = Path(__file__).parent / "graph_convergence_t1_results.txt"


def make(kind):
    if kind == "gcn":
        return SnapshotGCN(feature_dim=FEATURE_DIM, hidden_dim=32, layers=2, output_activation="linear")
    return TemporalGraphNetwork(feature_dim=FEATURE_DIM, memory_dim=32, edge_dim=2,
                                use_memory=(kind == "tgn"), output_activation="linear")


def gather(batches, ms, preds):
    ys, ps = [], []
    for m in ms:
        b = batches[m]; mk = b["mask"].numpy()
        ys.append(b["target"].numpy()[mk]); ps.append(preds[m].detach().numpy()[mk])
    y = np.concatenate(ys); p = np.concatenate(ps)
    return y, p


def train_seed(kind, seed, batches, months, n):
    torch.manual_seed(seed); np.random.seed(seed)
    model = make(kind)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    trm, vam, tem = split(months)

    def full_pass():
        preds = {}
        if kind != "gcn":
            model.reset_memory(n, torch.device("cpu"))
        for m in months:
            b = batches[m]
            preds[m] = model(b["features"], b["edge_index"]) if kind == "gcn" else model(b["features"], b["events"], b["time"])
        return preds

    loss_curve, val_curve = [], []
    best_val, best_test, wait = -np.inf, float("nan"), 0
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
        loss = torch.stack(losses).mean()
        loss.backward(); opt.step()
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
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    return best_test, loss_curve, val_curve


def emit(line):
    print(line, flush=True)
    with open(RESULTS, "a") as h:
        h.write(line + "\n")


def main():
    open(RESULTS, "w").close()
    nodes, edges = load()
    batches, months, n = build(nodes, edges)
    curves = {}
    emit("=== Converged graph models under T1 (regression, test R2) ===")
    emit(f"budget: {EPOCHS} epochs, patience {PATIENCE}, seeds {SEEDS}; AR ridge reference = 0.138")
    emit(f"{'model':16s} | {'mean':>7s} {'sd':>7s} | per-seed")
    emit("-" * 60)
    for kind, label in (("gcn", "GCN"), ("tgn", "TGN"), ("tgn_no_mem", "TGN no-memory")):
        tests = []
        best_curve = None
        for s in SEEDS:
            t, lc, vc = train_seed(kind, s, batches, months, n)
            tests.append(t)
            if s == 0:
                best_curve = (lc, vc)
        curves[label] = best_curve
        arr = np.array(tests, float)
        emit(f"{label:16s} | {arr.mean():7.3f} {arr.std():7.3f} | "
             + " ".join(f"{x:+.3f}" for x in tests))
    emit("-" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
        for label, (lc, vc) in curves.items():
            a1.plot(range(1, len(lc) + 1), lc, label=label)
            a2.plot(range(1, len(vc) + 1), vc, label=label)
        a1.set_title("Train MSE (seed 0)"); a1.set_xlabel("epoch"); a1.set_ylabel("MSE"); a1.legend()
        a2.axhline(0.138, color="k", ls="--", lw=1, label="AR ridge test $R^2$")
        a2.set_title("Validation $R^2$ (seed 0)"); a2.set_xlabel("epoch"); a2.set_ylabel("val $R^2$"); a2.legend()
        fig.tight_layout()
        (ROOT / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(ROOT / "figures/graph_convergence_t1.pdf")
        fig.savefig(ROOT / "figures/graph_convergence_t1.png", dpi=150)
        emit("figure -> figures/graph_convergence_t1.pdf (+ .png)")
    except Exception as e:  # noqa: BLE001
        emit(f"figure skipped: {e}")


if __name__ == "__main__":
    main()
