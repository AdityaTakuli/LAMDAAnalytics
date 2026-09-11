# Agent instructions (LAMDAAnalytics)

**Read `CODEBASE.md` before editing.** That file is the product/codebase PRD: two products in one repo, frozen paper numbers, directory map, and “if asked to X, open Y”.

## What this repo is right now

TMLR **methods / negative-result paper** on overlapping-window trade-disruption targets.
The FastAPI + LangGraph + React app is a **demo**, not a paper contribution.

## Non-negotiable

1. Paper numbers are frozen in `results/canonical_numbers.csv` and `results/phase07/`. Do not invent experiments unless the user asks.
2. Do not treat live `/analyze` or LangGraph extractors as validated skill.
3. Overlap bias is **imported** from financial econometrics, not discovered here.
4. Primary forecast target is **T1** (one-step log return). **T0** is the overlapping demonstration target.
5. Best clean-target model is the target's own lags: AR(1,12) at **R² 0.215** (univariate AR(1) **0.171** is the reference). Graphs/exogenous do not beat either. AR(1,12) > AR(1) is wild p = 0.037, unadjusted (`results/phase08/`).
6. Python for paper scripts: `backend/.venv/bin/python`.
7. Current manuscript: `Research Paper/paper.tex` (TMLR). Ignore `Research Paper/README.md` for claims.
8. Do not claim a public dataset URL until one exists.

If `CODEBASE.md` and an older README disagree, follow `CODEBASE.md` and `paper.tex`.
