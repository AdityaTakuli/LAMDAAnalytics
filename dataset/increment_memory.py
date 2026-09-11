"""Phase 0.9 C.4 — lag-2 and lag-3 autocorrelation of T1 increments.

If both are positive, the MA(1) underprediction at large k is explained by
longer-lag memory in the increments (measured, not assumed).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phase0_t1_ladder import load

ROOT = Path(__file__).resolve().parent.parent


def pooled_ac(series, lag):
    lagged = series.groupby(level=0).shift(lag)
    ok = series.notna() & lagged.notna()
    return float(np.corrcoef(series[ok], lagged[ok])[0, 1]), int(ok.sum())


def main():
    f = load()
    s = f.set_index("host_country_id")["T1"]
    print("T1 increment autocorrelations (within-country, pooled)")
    for h in (1, 2, 3, 4, 6, 11, 12):
        ac, n = pooled_ac(s, h)
        sign = "POSITIVE" if ac > 0 else "negative"
        print(f"  lag-{h:2d}: {ac:+.3f}  n={n}  ({sign})")


if __name__ == "__main__":
    main()
