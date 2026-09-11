"""Phase 0.6 C.5 - per-origin persistence R^2 across expanding windows, extended
through 2024-11 and covering T0, T1, T4.

Shows that a zero-parameter persistence rule's apparent skill is regime-dependent
(single-split evaluation reports regime, not skill), for every target construction.
Produces Fig 2 (figures/per_origin_persistence.pdf) and the CSV.

(Standalone rather than editing preprocessing_robustness.py, which is coupled to the
candidate-vs-baseline ablation on the T0 'contraction' target; this needs T1/T4.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).parent / "data/four_year_2021_2024/processed/nodes_monthly.csv"
MIN_TRAIN, HORIZON = 13, 6


def load():
    f = pd.read_csv(DATA)
    f["month"] = f["month"].astype(str)
    f = f.sort_values(["host_country_id", "month"]).reset_index(drop=True)
    v = f.groupby("host_country_id", sort=False)["inbound_flow_usd"]
    fut = v.shift(-1)
    med12 = v.transform(lambda s: s.rolling(12, min_periods=1).median())
    v_lag11 = v.shift(11)
    f["T0"] = (fut - med12) / med12.replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = f["inbound_flow_usd"] > 0
        f["T1"] = np.log(fut / f["inbound_flow_usd"].where(pos))
        f.loc[fut <= 0, "T1"] = np.nan
        f["T4"] = np.log(fut / v_lag11.where(v_lag11 > 0))
        f.loc[fut <= 0, "T4"] = np.nan
    return f


def persistence_r2(f, target, eval_months):
    lag = f.groupby("host_country_id", sort=False)[target].shift(1)
    mask = f["month"].isin(eval_months) & f[target].notna() & lag.notna()
    y = f.loc[mask, target].to_numpy(float)
    p = lag[mask].to_numpy(float)
    if len(y) <= 2 or np.std(y) <= 1e-12:
        return float("nan"), int(mask.sum())
    return r2_score(y, p), int(mask.sum())


def main():
    f = load()
    months = sorted(f["month"].unique())
    rows = []
    print(f"{'eval window':22s} {'n':>4s} {'T0':>8s} {'T1':>8s} {'T4':>8s}")
    print("-" * 54)
    for start in range(MIN_TRAIN, len(months), HORIZON):
        ev = months[start:start + HORIZON]
        if len(ev) < 2:
            continue
        rec = {"eval": f"{ev[0]}..{ev[-1]}"}
        for t in ("T0", "T1", "T4"):
            r2, n = persistence_r2(f, t, ev)
            rec[t] = r2
            rec["n"] = n
        rows.append(rec)
        print(f"{rec['eval']:22s} {rec['n']:4d} {rec['T0']:8.3f} {rec['T1']:8.3f} {rec['T4']:8.3f}")
    table = pd.DataFrame(rows)
    print("-" * 54)
    print(f"{'range':22s} {'':4s} "
          f"{table['T0'].min():.3f}..{table['T0'].max():.3f}  "
          f"{table['T1'].min():.3f}..{table['T1'].max():.3f}  "
          f"{table['T4'].min():.3f}..{table['T4'].max():.3f}")
    (ROOT / "results/phase06").mkdir(parents=True, exist_ok=True)
    table.to_csv(ROOT / "results/phase06/per_origin_persistence.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(table))
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for t, c in (("T0", "#d62728"), ("T1", "#1f77b4"), ("T4", "#2ca02c")):
            ax.plot(x, table[t], "o-", color=c, label=t)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([e.replace("..", "\n..") for e in table["eval"]], fontsize=7)
        ax.set_ylabel("persistence $R^2$")
        ax.set_title("Persistence skill is regime-dependent (zero parameters)")
        ax.legend()
        fig.tight_layout()
        (ROOT / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(ROOT / "figures/per_origin_persistence.pdf")
        fig.savefig(ROOT / "figures/per_origin_persistence.png", dpi=150)
        print("\nfigure -> figures/per_origin_persistence.pdf (+ .png)")
    except Exception as e:  # noqa: BLE001
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
