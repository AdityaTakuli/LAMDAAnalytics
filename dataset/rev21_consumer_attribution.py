#!/usr/bin/env python3
"""Phase A: what does each feature group add on top of the target's own lags?

For each basket and target, start from AR(1,12) (own lags 1 and 12) and add one
group at a time: trade proxies, news (3), weather, GSCPI, all exogenous (5),
all seven. Scored on test 2024 and on the pooled four-origin rolling
evaluation (2023-2024), with a wild-cluster-bootstrap p-value for
"AR(1,12)+group vs AR(1,12)" on the pooled rolling predictions.

Writes results/paper21_revision/consumer/attribution_over_ar.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rev21_tabular as tab  # noqa: E402
from dm_tests import wild_cluster_boot  # noqa: E402
from rev21_consumer_analyses import BASKETS, OUT  # noqa: E402

GROUPS = {
    "+ trade proxies": tab.TRADE,
    "+ news (3)": tab.NEWS,
    "+ weather": ["weather_anomaly_7d"],
    "+ GSCPI": ["global_risk"],
    "+ all exogenous (5)": tab.EXO,
    "+ all seven": tab.ALL7,
}


def rolling_preds(f, feats, target):
    ok = f[target].notna() & np.isfinite(f[target])
    ys, ps, cl = [], [], []
    for start, end in tab.BLOCKS:
        end_train = f"{int(start[:4]) - 1}-12" if start[5:] == "01" else f"{start[:4]}-06"
        tr = f["month"].between(tab.TRAIN[0], end_train) & ok
        te = f["month"].between(start, end) & ok
        ys.append(f.loc[te, target].to_numpy(float))
        ps.append(tab.fit_predict(f, feats, target, tr, te))
        cl.append(f.loc[te, "host_country_id"].to_numpy())
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(cl)


def main() -> int:
    rows = []
    for basket, path in BASKETS.items():
        tab.NODES = path
        f = tab.load()
        for target in ("T0", "T1"):
            base = [f"{target}_lag1", f"{target}_lag12"]
            ok = f[target].notna() & np.isfinite(f[target])
            tr = f["month"].between(*tab.TRAIN) & ok
            te = f["month"].between(*tab.TEST) & ok
            y_te = f.loc[te, target].to_numpy(float)
            y_r, p_base, clusters = rolling_preds(f, base, target)
            rec0 = {"basket": basket, "target": target, "model": "AR(1,12)",
                    "test_r2": r2_score(y_te, tab.fit_predict(f, base, target, tr, te)),
                    "rolling_pooled_r2": r2_score(y_r, p_base)}
            rows.append(rec0)
            for name, group in GROUPS.items():
                feats = base + [g for g in group if g not in base]
                _, p, _ = rolling_preds(f, feats, target)
                d = (y_r - p) ** 2 - (y_r - p_base) ** 2
                wild_p, _ = wild_cluster_boot(d, clusters)
                rows.append({"basket": basket, "target": target, "model": f"AR(1,12) {name}",
                             "test_r2": r2_score(y_te, tab.fit_predict(f, feats, target, tr, te)),
                             "rolling_pooled_r2": r2_score(y_r, p),
                             "gain_vs_ar_rolling": r2_score(y_r, p) - rec0["rolling_pooled_r2"],
                             "wild_p_vs_ar_rolling": wild_p})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "attribution_over_ar.csv", index=False)
    print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
