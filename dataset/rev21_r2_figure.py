#!/usr/bin/env python3
"""Round 2: corrected version of the benchmark figure (results_country_month_panels), scored on the
187 test rows with an observable target. Same layout and colours as the first version; the own-lag
ridge (Table ablation) is added because it is the only model with positive skill.
Writes results/paper21_revision/round2/figures/results_country_month_panels_r2.{pdf,png}, and the
manuscript's figures folder too when it is present."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

R2 = Path(__file__).resolve().parents[1] / "results" / "paper21_revision" / "round2"
FIG = R2 / "figures"
PAPER_FIG = Path(__file__).resolve().parents[1] / "Research Paper" / "paper2" / "figures"
COL = {"Graph": "#2563eb", "Linear": "#059669", "Tree": "#d97706", "Baseline": "#6b7280"}
plt.rcParams.update({"font.family": "serif", "font.size": 10})

reg = pd.read_csv(R2 / "corrected_reg.csv").query("scope == 'corrected'").set_index("model")["r2"]
clf = pd.read_csv(R2 / "corrected_clf.csv").query("scope == 'corrected'").set_index("model")
abl = pd.read_csv(R2 / "corrected_ablation.csv").query("target == 'T0'").set_index("features")
own = abl.loc["Own lag only (AR(1))"]
a = [("Own lag $c_{T-1}$ (ridge)", own["test_r2"], "Linear"), ("Ridge", reg["ridge_regression"], "Linear"),
     ("GCN", reg["gcn"], "Graph"), ("TGN (no mem.)", reg["tgn_no_memory"], "Graph"), ("TGN", reg["tgn"], "Graph"),
     ("LightGBM+lags", reg["lightgbm_lags"], "Tree"), ("Random forest", reg["random_forest"], "Tree"),
     ("LightGBM", reg["lightgbm"], "Tree"), ("Train median", reg["train_median"], "Baseline")]
b = [("Own lag $c_{T-1}$ (logistic)", own["test_pr_auc"], "Linear"), ("TGN", clf.loc["tgn", "pr_auc"], "Graph"),
     ("Hand-weighted", clf.loc["hand_weighted_linear", "pr_auc"], "Linear"), ("GCN", clf.loc["gcn", "pr_auc"], "Graph"),
     ("TGN (no mem.)", clf.loc["tgn_no_memory", "pr_auc"], "Graph"), ("Logistic", clf.loc["logistic_regression", "pr_auc"], "Linear"),
     ("LightGBM", clf.loc["lightgbm", "pr_auc"], "Tree"), ("LightGBM+lags", clf.loc["lightgbm_lags", "pr_auc"], "Tree"),
     ("Random forest", clf.loc["random_forest", "pr_auc"], "Tree"), ("Prevalence", clf.loc["constant_prevalence", "pr_auc"], "Baseline")]
prev = clf.loc["constant_prevalence", "prevalence"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
for ax, rows, title, xlab, ref, reflab in (
        (axes[0], a, "(a) Regression", "$R^2$ (higher is better)", 0.0, "zero"),
        (axes[1], b, r"(b) Classification, $\tau=0.20$", "PR-AUC (higher is better)", prev, f"prev. ({prev:.3f})")):
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [COL[r[2]] for r in rows][::-1]
    ax.barh(names, vals, color=cols, height=0.65)
    for i, v in enumerate(vals):
        ax.text(v + (0.008 if v >= 0 else -0.008), i, f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.axvline(ref, color="#dc2626", ls="--", lw=1.4)
    ax.text(ref + 0.01, -1.05, reflab, color="#dc2626", fontsize=8.5)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(xlab)
    ax.grid(axis="x", ls="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_xlim(-0.45, 0.5)
axes[1].set_xlim(0, 0.42)
fig.suptitle("Country-month benchmark, corrected (post-repair, $n=187$ rows with an observable target)", fontweight="bold")
fig.legend(handles=[Patch(color=c, label=k) for k, c in COL.items()], loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
FIG.mkdir(parents=True, exist_ok=True)
for out in (FIG, PAPER_FIG) if PAPER_FIG.is_dir() else (FIG,):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"results_country_month_panels_r2.{ext}", dpi=300)
print("written")
