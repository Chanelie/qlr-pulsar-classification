# qlr/aggregate.py
# ------------------------------------------------------------
# Seed-paired aggregation of metrics and deltas vs best baseline.
#
# Outputs (mean ± standard error across seeds) for:
#   - QLR by (encoding, layers, n_samples)
#   - Baselines by (model, n_samples)
#   - Paired deltas: Δ(QLR − best_baseline) for selected metrics
#
# Also computes low-FPR operating metrics per seed:
#   - Recall_FPR1, Precision_FPR1, FNR_FPR1 (at FPR <= 1%)
#   - Recall_FPR5 (at FPR <= 5%)
#
# Runtime:
#   - Baselines: expects keys "<Model>_train_s" and "<Model>_pred_s"
#   - QLR: reads cached meta keys "train_time_s" and "pred_time_s"
# ------------------------------------------------------------

from __future__ import annotations

from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    brier_score_loss,
    roc_curve,
    precision_score,
    recall_score,
)

from .baselines import train_baselines_for_seed
from .cache import ArtifactManager
from .paths import baseline_cache_path


def ece_from_probs(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (ECE) with equal-mass (quantile) binning.
    Solves imbalance skew by ensuring each bin has the same number of samples.
    ECE = sum_k (|B_k|/n) * |acc(B_k) - conf(B_k)|
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_prob)

    # Sort probabilities to create equal-mass bins
    sort_idx = np.argsort(y_prob)
    y_true_sorted = y_true[sort_idx]
    y_prob_sorted = y_prob[sort_idx]

    # Split into n_bins (handles non-perfect divisibility automatically)
    bins_true = np.array_split(y_true_sorted, n_bins)
    bins_prob = np.array_split(y_prob_sorted, n_bins)

    ece = 0.0
    for bt, bp in zip(bins_true, bins_prob):
        if len(bp) == 0:
            continue
        acc = float(np.mean(bt))       # empirical accuracy within bin
        conf = float(np.mean(bp))      # average predicted confidence
        weight = len(bp) / n           # fraction of total samples
        ece += weight * abs(acc - conf)

    return float(ece)


def metrics_from_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    return dict(
        PR_AUC=average_precision_score(y_true, y_prob),
        ROC_AUC=roc_auc_score(y_true, y_prob),
        F1=f1_score(y_true, y_pred, zero_division=0),
        Precision=precision_score(y_true, y_pred, zero_division=0),
        Recall=recall_score(y_true, y_pred, zero_division=0),
        Brier=brier_score_loss(y_true, y_prob),
        ECE=ece_from_probs(y_true, y_prob, n_bins=15),
    )




def metrics_at_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fpr_target: float,
) -> Tuple[float, float, float, float]:
    fpr, tpr, thr = roc_curve(y_true, y_prob)

    mask = fpr <= fpr_target
    if not np.any(mask):
        print(f"[WARN] No threshold found for FPR <= {fpr_target}. Defaulting to 0.0 recall.") 
        return 0.0, 0.0, 1.0, np.inf

    i = int(np.argmax(tpr[mask]))
    recall = float(tpr[mask][i])
    thr_star = float(thr[mask][i])

    y_pred = (y_prob >= thr_star).astype(int)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    precision = float(TP / (TP + FP + 1e-12))

    fnr = float(1.0 - recall)
    return recall, precision, fnr, thr_star


def _get_baseline_time(times: Dict[str, float], model: str, kind: str) -> float:
    key = f"{model}_{kind}"  # e.g., "QSVM_train_s"
    v = times.get(key, np.nan)
    return float(v) if np.isfinite(v) else float(np.nan)




def aggregate_block(
    block_cfgs: List[dict],
    AM: ArtifactManager,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    ns_vals = sorted({cfg["n_samples"] for cfg in block_cfgs})
    if len(ns_vals) != 1:
        raise ValueError("aggregate_block expects a single n_samples value in block_cfgs")
    ns = ns_vals[0]

    seeds = sorted({cfg["seed"] for cfg in block_cfgs})

    rows_qlr: List[dict] = []
    rows_bl: List[dict] = []

    for sd in seeds:
        print(f"[AGG] start baselines seed={sd}")

        path = baseline_cache_path(ns, sd)

        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            bl_probs = data["probs"]
            bl_times = data["times"]
        else:
            print(f"[AGG] baseline cache missing for n={ns} seed={sd}; recomputing")
            bl_probs, bl_times = train_baselines_for_seed(ns, sd, X_train, y_train, X_test, y_test)

        print(f"[AGG] done baselines seed={sd}")

        for name, y_prob in bl_probs.items():
            base = metrics_from_probs(y_test, y_prob)
            rec1, prec1, fnr1, _ = metrics_at_fpr(y_test, y_prob, 0.01)
            rec5, _, _, _ = metrics_at_fpr(y_test, y_prob, 0.05)

            base.update(
                {
                    "Recall_FPR1": rec1,
                    "Precision_FPR1": prec1,
                    "FNR_FPR1": fnr1,
                    "Recall_FPR5": rec5,
                }
            )

            rows_bl.append(
                {
                    "model": name,
                    "seed": sd,
                    "n_samples": ns,
                    **base,
                    "Train_s": _get_baseline_time(bl_times, name, "train_s"),
                    "Pred_s": _get_baseline_time(bl_times, name, "pred_s"),
                }
            )

        for cfg in (c for c in block_cfgs if c["seed"] == sd):
            payload = AM.load(cfg)
            if payload is None:
                continue

            y_t, probs_dict, *_rest = payload
            meta = _rest[-1] if len(_rest) > 0 else None

            qlr_key = next(k for k in probs_dict.keys() if k.startswith("QLR-"))
            y_prob = probs_dict[qlr_key]

            base = metrics_from_probs(y_t, y_prob)
            rec1, prec1, fnr1, _ = metrics_at_fpr(y_t, y_prob, 0.01)
            rec5, _, _, _ = metrics_at_fpr(y_t, y_prob, 0.05)

            base.update(
                {
                    "Recall_FPR1": rec1,
                    "Precision_FPR1": prec1,
                    "FNR_FPR1": fnr1,
                    "Recall_FPR5": rec5,
                }
            )

            train_s = float(meta.get("train_time_s", np.nan)) if isinstance(meta, dict) else float(np.nan)
            pred_s = float(meta.get("pred_time_s", np.nan)) if isinstance(meta, dict) else float(np.nan)

            rows_qlr.append(
                {
                    "encoding": cfg["encoding"],
                    "layers": cfg["n_layers"],
                    "seed": sd,
                    "n_samples": ns,
                    **base,
                    "Train_s": train_s,
                    "Pred_s": pred_s,
                }
            )

    if not rows_qlr or not rows_bl:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty, empty

    df_qlr = pd.DataFrame(rows_qlr)
    df_bl = pd.DataFrame(rows_bl)

    qlr_cols = [
        "PR_AUC", "ROC_AUC", "F1", "Precision", "Recall", "Brier", "ECE",
        "Recall_FPR1", "Precision_FPR1", "FNR_FPR1", "Recall_FPR5",
        "Train_s", "Pred_s",
    ]

    bl_cols = qlr_cols[:]  # same numeric columns (model vs encoding/layers differs)

    n_seeds = len(seeds)

    qlr_g = df_qlr.groupby(["encoding", "layers", "n_samples"], as_index=False)
    qlr_mean = qlr_g[qlr_cols].mean(numeric_only=True)
    qlr_se   = qlr_g[qlr_cols].std(numeric_only=True)
    # Convert metric/runtime standard deviations to standard errors.
    # Keep grouping columns such as layers and n_samples unchanged.
    qlr_se[qlr_cols] = qlr_se[qlr_cols] / np.sqrt(n_seeds)

    bl_g = df_bl.groupby(["model", "n_samples"], as_index=False)
    bl_mean = bl_g[bl_cols].mean(numeric_only=True)
    bl_se   = bl_g[bl_cols].std(numeric_only=True)
    # Convert metric/runtime standard deviations to standard errors.
    # Keep grouping columns such as n_samples unchanged.
    bl_se[bl_cols] = bl_se[bl_cols] / np.sqrt(n_seeds)

    # Paired deltas vs the best classical baseline, per metric and seed
    DELTA_METRICS = ["ROC_AUC", "PR_AUC", "Recall_FPR1", "Recall_FPR5", "Brier", "ECE"]

    # Metrics where lower is better: the best classical baseline is the
    # minimum (idxmin), not the maximum. Δ is still QLR − best_classical,
    # so for these metrics a negative Δ means QLR is better.
    LOWER_IS_BETTER = {"Brier", "ECE"}

    EXCLUDE_FROM_DELTA = {"QSVM"}  # exclude QSVM only for selecting best baseline in Δ

    # Determine classical models dynamically (anything in baselines except excluded)
    all_models = set(df_bl["model"].unique())
    classical_models = sorted(all_models - EXCLUDE_FROM_DELTA)

    if not classical_models:
        raise ValueError("No classical models available for delta computation (after exclusions).")

    delta_rows: List[dict] = []
    for sd in seeds:
        # Baselines for this seed, restricted to classical models only
        bl_seed = df_bl[(df_bl["seed"] == sd) & (df_bl["model"].isin(classical_models))].copy()
        if bl_seed.empty:
            continue
        bl_seed = bl_seed.set_index("model")

        # QLR rows for this seed
        qlr_seed = df_qlr[df_qlr["seed"] == sd]
        if qlr_seed.empty:
            continue

        for m in DELTA_METRICS:
            if m not in bl_seed.columns:
                continue

            # best classical baseline for this metric/seed
            bl_vals = pd.to_numeric(bl_seed[m], errors="coerce")
            if bl_vals.isna().all():
                continue

            best_model = bl_vals.idxmin() if m in LOWER_IS_BETTER else bl_vals.idxmax()
            best_val = float(bl_vals.loc[best_model])

            # Δ for every QLR row at this seed
            for _, r in qlr_seed.iterrows():
                r_val = float(r[m])
                delta_rows.append(
                    {
                        "seed": sd,
                        "encoding": r["encoding"],
                        "layers": int(r["layers"]),
                        "metric": m,
                        "vs": f"BEST_CLASSICAL({best_model})",
                        "delta": r_val - best_val,
                    }
                )

    df_delta = pd.DataFrame(delta_rows)

    print(
        f"[AGG] collected qlr_rows={len(rows_qlr)} "
        f"baseline_rows={len(rows_bl)} "
        f"delta_rows={len(delta_rows)}"
    )


    # Aggregate paired deltas across seeds. Do not group by ``vs`` here, because
    # the best classical model can change from seed to seed. Including ``vs``
    # would split the three paired deltas into smaller groups and produce NaN
    # standard errors when a group contains only one seed.
    d_mean = df_delta.groupby(["encoding", "layers", "metric"], as_index=False)["delta"].mean()
    d_se   = df_delta.groupby(["encoding", "layers", "metric"], as_index=False)["delta"].std()
    d_se["delta"] = d_se["delta"] / np.sqrt(n_seeds)


    return qlr_mean, qlr_se, bl_mean, bl_se, d_mean, d_se
