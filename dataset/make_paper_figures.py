"""Build the three result figures the paper needs beyond the two frozen ones.

Reads only frozen CSVs (results/canonical_numbers.csv and
results/phase07/graph_convergence_{T0,T1}.csv); retrains nothing.

  figures/rank_inversion.pdf       F2  T0 -> T1 ranking flip
  figures/feature_degradation.pdf  F3/F4  nothing beats AR(1) on T1
  figures/graph_seeds.pdf          F3  per-seed graph spread vs AR baselines
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"
CANON = ROOT / "results/canonical_numbers.csv"


def canon() -> pd.DataFrame:
    df = pd.read_csv(CANON)
    return df[(df["metric"] == "R2") & (df["split"] == "test2024")]


def value(df: pd.DataFrame, target: str, model: str) -> float:
    hit = df[(df["target"] == target) & (df["model"] == model)]["value"]
    if len(hit) != 1:
        raise KeyError(f"{target}/{model}: {len(hit)} rows in canonical_numbers.csv")
    return float(hit.iloc[0])


def fmt3(v: float, sign: bool = False) -> str:
    """Three decimals, matching the paper's tables.

    canonical_numbers.csv stores six decimals, so ordinary rounding here gives the
    same three-decimal value as rounding the unrounded result (no double rounding).
    """
    d = Decimal(repr(float(v))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return f"{d:+}" if sign else f"{d}"


def spread(ys: list[float], gap: float, lo: float) -> list[float]:
    """Label y-positions at least `gap` apart, kept as close to `ys` as possible."""
    order = sorted(range(len(ys)), key=lambda i: -ys[i])
    pos = [ys[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = min(pos[k], pos[k - 1] - gap)
    if pos and pos[-1] < lo:
        pos[-1] = lo
        for k in range(len(pos) - 2, -1, -1):
            pos[k] = max(pos[k], pos[k + 1] + gap)
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


LADDER = [
    ("persistence", "Persistence $c_{T-1}$"),
    ("seasonal_T-11", "Seasonal $c_{T-11}$"),
    ("ridge_paper7", "Ridge, 7 feat."),
    ("ar_ridge", "AR ridge, 3 feat."),
    ("ar1_only", "Univariate AR(1)"),
    ("random_forest", "Random forest"),
    ("lightgbm", "LightGBM"),
    ("per_country_median", "Per-country median"),
    ("global_median", "Global median"),
    ("zero", "Zero"),
]


def rank_inversion(df: pd.DataFrame) -> None:
    """F2: slope chart of every ladder model, overlapping T0 -> overlap-free T1."""
    rows = [(lab, value(df, "T0", m), value(df, "T1", m)) for m, lab in LADDER]
    lo, hi = -0.70, 0.78

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.axhline(0, color="gray", lw=0.6, zorder=0)

    y1s = [max(t1, lo) for _, _, t1 in rows]
    right = spread(y1s, gap=0.055, lo=lo + 0.02)
    left = spread([t0 for _, t0, _ in rows], gap=0.045, lo=lo + 0.02)

    for (lab, t0, t1), y1, ly1, ly0 in zip(rows, y1s, right, left):
        off = t1 < lo
        crosses = (t0 > 0.4) and (t1 < 0.05)
        color = "#d62728" if crosses else "#4c72b0"
        lw = 2.0 if crosses else 1.1
        ax.plot([0, 1], [t0, y1], "-", color=color, lw=lw, alpha=0.9, zorder=2)
        ax.plot([0], [t0], "o", color=color, ms=5, zorder=3)
        ax.plot([1], [y1], "v" if off else "o", color=color, ms=7 if off else 5,
                clip_on=False, zorder=3)
        text = f"{lab}  {fmt3(t1)}" + (" (off scale)" if off else "")
        ax.annotate(text, xy=(1.0, y1), xytext=(1.08, ly1), fontsize=6.5, color=color,
                    va="center", arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5))
        ax.annotate(fmt3(t0), xy=(0.0, t0), xytext=(-0.08, ly0), fontsize=6.5, color=color,
                    va="center", ha="right",
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.5))

    ax.set_xlim(-0.42, 1.95)
    ax.set_ylim(lo, hi)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["T0\n(overlap $\\approx 11/12$)", "T1\n(overlap $0$)"])
    ax.set_ylabel("test-2024 $R^2$")
    ax.set_title("Removing window overlap inverts the ranking", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, "rank_inversion")


DEGRADE = [
    ("ar1_only", "AR(1), lag only"),
    ("ar1_plus_flow", "AR(1) + flow level"),
    ("exo_+weather", "AR ridge + weather"),
    ("ar_ridge", "AR ridge, 3 feat."),
    ("exo_+news", "AR ridge + news"),
    ("exo_+gscpi", "AR ridge + GSCPI"),
    ("exo_+all", "AR ridge + all exog."),
]


def feature_degradation(df: pd.DataFrame) -> None:
    """F3/F4: adding channels to the T1 AR baseline only ever costs skill."""
    rows = [(lab, value(df, "T1", m)) for m, lab in DEGRADE]
    rows.sort(key=lambda r: r[1])
    ref = value(df, "T1", "ar1_only")

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    y = np.arange(len(rows))
    vals = [v for _, v in rows]
    colors = ["#d62728" if v < 0 else "#4c72b0" for v in vals]
    ax.barh(y, vals, color=colors, height=0.62)
    ax.axvline(0, color="gray", lw=0.8)
    ax.axvline(ref, color="#2ca02c", lw=1.2, ls="--",
               label=f"univariate AR(1) = {fmt3(ref)}")

    for yi, v in zip(y, vals):
        ax.annotate(fmt3(v, sign=True), xy=(v, yi), xytext=(4 if v >= 0 else -4, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v >= 0 else "right", fontsize=7)

    ax.set_yticks(y)
    ax.set_yticklabels([lab for lab, _ in rows], fontsize=8)
    ax.set_xlim(-0.26, 0.26)
    ax.set_xlabel("test-2024 $R^2$ on T1 (overlap $0$)")
    ax.set_title("No exogenous channel clears the target's own lag", fontsize=10)
    ax.legend(loc="lower left", fontsize=7, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    save(fig, "feature_degradation")


GRAPHS = [("gcn", "GCN"), ("tgn", "TGN"), ("tgn_no_mem", "TGN,\nno memory")]


def graph_seeds(df: pd.DataFrame) -> None:
    """F3: five matched-budget seeds per architecture against the AR baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 3.2))

    for ax, target, ylim in zip(axes, ("T0", "T1"), ((0.30, 0.75), (-0.05, 0.25))):
        seeds = pd.read_csv(ROOT / f"results/phase07/graph_convergence_{target}.csv")
        seeds = seeds.set_index("model")
        for i, (m, lab) in enumerate(GRAPHS):
            vals = [float(v) for v in str(seeds.loc[m, "per_seed"]).split()]
            mean = float(seeds.loc[m, "mean"])
            if not np.isclose(np.mean(vals), mean, atol=5e-3):
                raise ValueError(f"{target}/{m}: per_seed mean {np.mean(vals):.4f} "
                                 f"!= stored mean {mean:.4f}")
            ax.plot(np.full(len(vals), i), vals, "o", color="#4c72b0", ms=5,
                    alpha=0.7, zorder=3)
            ax.plot([i - 0.22, i + 0.22], [mean, mean], "-", color="#d62728",
                    lw=1.8, zorder=4)

        ar1 = value(df, target, "ar1_only")
        ar112 = value(df, target, "ar1_12")
        ridge = value(df, target, "ar_ridge")
        ax.axhline(ar112, color="#9467bd", ls="-.", lw=1.1,
                   label=f"AR(1,12) = {fmt3(ar112)}")
        ax.axhline(ar1, color="#2ca02c", ls="--", lw=1.1,
                   label=f"AR(1) = {fmt3(ar1)}")
        ax.axhline(ridge, color="#8c564b", ls=":", lw=1.1,
                   label=f"AR ridge = {fmt3(ridge)}")

        if target == "T1":
            pers = value(df, "T1", "persistence")
            ax.annotate(f"persistence {fmt3(pers)}\n(off scale)",
                        xy=(0.02, 0.03), xycoords="axes fraction",
                        fontsize=6.5, color="gray")
            ax.axhline(0, color="gray", lw=0.6, zorder=0)

        ax.set_xticks(range(len(GRAPHS)))
        ax.set_xticklabels([lab for _, lab in GRAPHS], fontsize=7.5)
        ax.set_xlim(-0.5, len(GRAPHS) - 0.5)
        ax.set_ylim(*ylim)
        ax.set_title(f"{target} (overlap "
                     f"{'$\\approx 11/12$' if target == 'T0' else '$0$'})",
                     fontsize=9)
        ax.legend(loc="lower right", fontsize=6.5, frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    axes[0].set_ylabel("test-2024 $R^2$")
    fig.suptitle("Five seeds at matched budget; spread is small, level is not competitive",
                 fontsize=9.5)
    fig.tight_layout()
    save(fig, "graph_seeds")


def save(fig, stem: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{stem}.pdf")
    fig.savefig(FIGS / f"{stem}.png", dpi=150)
    plt.close(fig)
    print(f"-> figures/{stem}.pdf (+ .png)")


def main() -> None:
    df = canon()
    rank_inversion(df)
    feature_degradation(df)
    graph_seeds(df)


if __name__ == "__main__":
    main()
