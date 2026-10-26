#!/usr/bin/env python3
"""
Post-hoc calibration of the classical baselines, fitted on the training subset
only. The test split is used only for evaluation.

For each N and seed:
  1. Rebuild the main experiment's fixed split (Config.DATA_SEED) and the paired
     training subset (qlr.data.subset_indices_for), as qlr/baselines.py does.
  2. For LogReg, SVM-RBF (CSVM) and XGBoost, weighted (grid setup) and
     unweighted (as in scripts/baselines_weighting_comparison_N1000.py):
     run the SAME GridSearchCV as qlr/baselines.py (grids, scoring, cv=3,
     random_state=seed) and keep the selected model ("none" = uncalibrated),
     then calibrate a clone of the selected model with
     CalibratedClassifierCV(cv=5) on the training subset only, with
     method="sigmoid" (Platt) and method="isotonic". cv=5 gives StratifiedKFold(5)
     for a classifier. The test split is never seen during fitting.
  3. Evaluate on the test split with the project's own metric code:
     qlr.aggregate.metrics_from_probs / metrics_at_fpr (PR-AUC, ROC-AUC,
     Recall@FPR1/5%, Brier, 15-bin equal-mass ECE, ...) and
     qlr.plots.compute_murphy_components (reliability, resolution, uncertainty).
  4. Save per-seed test predictions to results/calibrated_baselines_train/
     preds_N{N}_seed{s}.pkl (used by the bootstrap analysis), per-seed metrics to
     results/calibrated_baselines_train/per_seed_N{N}_seed{s}.csv, and the
     per-seed + mean ± SE table to tables/calibrated_baselines_train_N{N}.csv.

Consistency check: the uncalibrated WEIGHTED probabilities must equal the grid
cache caches/baselines_N{N}_seed{s}.pkl (if present) bit for bit.

Stability flags (column `flags`):
  few_pos_cal   : some calibration fold holds < 10 positives
  isotonic_steps: isotonic output has < 20 distinct values on the test set
  murphy_bins<15: calibration_curve returned < 15 quantile bins (tied
                  probabilities), so compute_murphy_components' reconstructed
                  bin weights are misaligned; use the *_aligned Murphy columns
                  (same bins, consistent weights) for these rows
  cache_mismatch: uncalibrated weighted probs differ from the grid cache

Usage:
  python scripts/posthoc_calibration_train.py --samples 200 500 1000 --seeds 0 1 2 --jobs 8
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import pickle
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.base import clone                                  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # noqa: E402
from sklearn.linear_model import LogisticRegression             # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.svm import SVC                                      # noqa: E402
from xgboost import XGBClassifier                                # noqa: E402

from qlr.config import Config                                   # noqa: E402
from qlr.data import load_htru2, HTRU2Preprocessor, subset_indices_for  # noqa: E402
from qlr.aggregate import metrics_from_probs, metrics_at_fpr     # noqa: E402
from qlr.plots import compute_murphy_components                  # noqa: E402
from qlr.paths import baseline_cache_path                        # noqa: E402

OUT_DIR = ROOT / "results" / "calibrated_baselines_train"
MODELS = ["LogReg", "CSVM", "XGBoost"]
METHODS = ["none", "sigmoid", "isotonic"]
METRIC_COLS = ["PR_AUC", "ROC_AUC", "Recall_FPR1", "Recall_FPR5", "Brier", "ECE",
               "F1", "Precision", "Recall", "Precision_FPR1", "FNR_FPR1",
               "Murphy_reliability", "Murphy_resolution", "Murphy_uncertainty",
               "Murphy_reliability_aligned", "Murphy_resolution_aligned"]


def grid_search(model: str, Xb, yb, seed: int, class_balanced: bool):
    """Identical estimators, grids, scoring and CV to qlr/baselines.py."""
    cw = "balanced" if class_balanced else None
    if model == "LogReg":
        est = LogisticRegression(solver="lbfgs", max_iter=2000, class_weight=cw, random_state=seed)
        grid = {'C': [0.1, 1.0, 10.0]}
    elif model == "CSVM":
        est = SVC(kernel="rbf", probability=True, class_weight=cw, random_state=seed)
        grid = {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']}
    else:
        pos, neg = float((yb == 1).sum()), float((yb == 0).sum())
        spw = ((neg / pos) if pos > 0 else 1.0) if class_balanced else 1.0
        est = XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=seed, n_jobs=1)
        grid = {'max_depth': [3, 5], 'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100]}
    gs = GridSearchCV(est, param_grid=grid, scoring='average_precision', cv=3, n_jobs=1)
    gs.fit(Xb, yb)
    return gs.best_estimator_, gs.best_params_


def evaluate(y, p) -> tuple[dict, int]:
    m = metrics_from_probs(y, p)
    r1, p1, f1, _ = metrics_at_fpr(y, p, 0.01)
    r5, _, _, _ = metrics_at_fpr(y, p, 0.05)
    m.update(Recall_FPR1=r1, Precision_FPR1=p1, FNR_FPR1=f1, Recall_FPR5=r5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = compute_murphy_components(np.asarray(y), np.asarray(p))
        n_bins = len(calibration_curve(y, p, n_bins=15, strategy="quantile")[0])
    m.update(Murphy_reliability=mu["reliability"], Murphy_resolution=mu["resolution"],
             Murphy_uncertainty=mu["uncertainty"])
    rel_a, res_a = murphy_aligned(np.asarray(y), np.asarray(p))
    m.update(Murphy_reliability_aligned=rel_a, Murphy_resolution_aligned=res_a)
    return m, n_bins


def murphy_aligned(y, p, n_bins: int = 15):
    """Same 15 quantile bins as sklearn calibration_curve(strategy="quantile")
    (edges = np.percentile, bin id = np.searchsorted(edges[1:-1], p)), with the
    bin weights taken from THE SAME bin assignment. qlr.plots.compute_murphy_components
    rebuilds the weights with a different edge rule and truncates them, so when
    probabilities are tied (isotonic, XGBoost) its weights and bin means refer to
    different bins. For the 135 QLR caches the two agree to < 2e-5."""
    edges = np.percentile(p, np.linspace(0, 100, n_bins + 1))
    ids = np.searchsorted(edges[1:-1], p)
    c = np.bincount(ids, minlength=n_bins)
    sp = np.bincount(ids, weights=p, minlength=n_bins)
    sy = np.bincount(ids, weights=y, minlength=n_bins)
    nz = c > 0
    w, pp, pt = c[nz] / c.sum(), sp[nz] / c[nz], sy[nz] / c[nz]
    return float(np.sum(w * (pp - pt) ** 2)), float(np.sum(w * (pt - y.mean()) ** 2))


def run_one(args) -> list[dict]:
    n, seed = args
    df = load_htru2(Config.DATA_PATH)
    X_train, X_test, y_train, y_test = HTRU2Preprocessor(
        test_size=Config.TEST_SIZE, random_state=Config.DATA_SEED).preprocess(df)
    idx = subset_indices_for(n, seed, X_train, y_train)
    Xb, yb = X_train[idx], y_train[idx]
    n_pos = int(yb.sum())
    fold_pos = [int(yb[te].sum()) for _, te in StratifiedKFold(5).split(Xb, yb)]

    grid_cache = None
    cp = baseline_cache_path(n, seed)
    if cp.exists() and cp.stat().st_size:
        with open(cp, "rb") as f:
            grid_cache = pickle.load(f)["probs"]

    rows, preds = [], {}
    for weighting, cb in [("weighted", True), ("unweighted", False)]:
        for model in MODELS:
            best, params = grid_search(model, Xb, yb, seed, cb)
            for method in METHODS:
                if method == "none":
                    p = best.predict_proba(X_test)[:, 1].astype(float)
                else:
                    cal = CalibratedClassifierCV(estimator=clone(best), method=method, cv=5)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        cal.fit(Xb, yb)
                    p = cal.predict_proba(X_test)[:, 1].astype(float)
                preds[(model, weighting, method)] = p
                m, n_bins = evaluate(y_test, p)
                flags = []
                if method != "none" and min(fold_pos) < 10:
                    flags.append("few_pos_cal")
                n_unique = int(np.unique(np.round(p, 12)).size)
                if method == "isotonic" and n_unique < 20:
                    flags.append("isotonic_steps")
                if n_bins < 15:
                    flags.append("murphy_bins<15")
                if method == "none" and weighting == "weighted" and grid_cache is not None:
                    if not np.array_equal(p, np.asarray(grid_cache[model], dtype=float)):
                        flags.append("cache_mismatch")
                rows.append(dict(row_type="per_seed", seed_set="", N=n, seed=seed, model=model,
                                 weighting=weighting, method=method, best_params=str(params),
                                 n_pos_train=n_pos, min_pos_per_cal_fold=min(fold_pos),
                                 n_unique_test_probs=n_unique, n_murphy_bins=n_bins,
                                 flags=";".join(flags), **{k: m[k] for k in METRIC_COLS}))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"preds_N{n}_seed{seed}.pkl", "wb") as f:
        pickle.dump({"N": n, "seed": seed, "y_test": y_test, "probs": preds,
                     "subset_idx": idx, "calibration": "CalibratedClassifierCV(cv=5) on training subset",
                     "keys": "(model, weighting, method)"}, f)
    pd.DataFrame(rows).to_csv(OUT_DIR / f"per_seed_N{n}_seed{seed}.csv", index=False)
    print(f"[calibration] N={n} seed={seed} done; flags: "
          f"{sorted({r['flags'] for r in rows if r['flags']})}", flush=True)
    return rows


def write_table(n: int) -> Path:
    """Per-seed rows for every seed available at this N + mean ± SE rows per seed set."""
    files = sorted(OUT_DIR.glob(f"per_seed_N{n}_seed*.csv"))
    per = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    per["flags"] = per["flags"].fillna("")
    seeds = sorted(per.seed.unique())
    sets = [("0-2", [0, 1, 2])] if set([0, 1, 2]) <= set(seeds) else []
    if len(seeds) > 3:
        sets.append((f"{min(seeds)}-{max(seeds)}", seeds))
    out = [per]
    for tag, ss in sets:
        sub = per[per.seed.isin(ss)]
        g = sub.groupby(["model", "weighting", "method"], sort=False)
        mean = g[METRIC_COLS].mean().reset_index()
        se = (g[METRIC_COLS].std(ddof=1) / np.sqrt(len(ss))).reset_index()
        fl = g["flags"].apply(lambda s: ";".join(sorted({x for v in s for x in v.split(";") if x}))).reset_index()
        for d, rt in [(mean, "mean"), (se, "se")]:
            d = d.merge(fl, on=["model", "weighting", "method"])
            d.insert(0, "row_type", rt); d.insert(1, "seed_set", tag); d.insert(2, "N", n); d.insert(3, "seed", "")
            d["n_seeds"] = len(ss)
            out.append(d)
    tab = pd.concat(out, ignore_index=True)
    path = ROOT / "tables" / f"calibrated_baselines_train_N{n}.csv"
    tab.to_csv(path, index=False)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, nargs="+", default=[200, 500, 1000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    tasks = [(n, s) for n in a.samples for s in a.seeds]
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            pool.map(run_one, tasks)
    else:
        for t in tasks:
            run_one(t)
    for n in a.samples:
        print(f"[calibration] wrote {write_table(n)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
