# scripts/baselines_weighting_comparison_N1000.py
# ------------------------------------------------------------
# Weighted vs unweighted classical baselines at N=1000, seeds 0/1/2.
#
# QLR trains with an unweighted BCE loss (runner.py passes
# class_weight=None to HybridQLRClassifier), while the reported classical
# baselines train weighted (class_weight="balanced" for LogReg/SVM-RBF,
# scale_pos_weight=neg/pos for XGBoost). This script re-runs both arms so
# the calibration comparison can be made without that asymmetry.
#
# Both arms are trained fresh in this one process, so they share the same
# sklearn/xgboost versions and the same paired stratified subsets; the
# weighted arm is additionally cross-checked against the existing cached
# baseline probabilities (read-only) and any drift is reported.
#
# QSVM is skipped (include_qsvm=False): it costs hours of kernel
# evaluation and its SVC is unweighted in both arms anyway.
#
# Writes only new files, all under results/baselines_weighting_N1000/.
# Nothing in caches/, tables/ or the rest of results/ is read-modify-written.
# ------------------------------------------------------------

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qlr.aggregate import metrics_from_probs, metrics_at_fpr
from qlr.baselines import train_baselines_for_seed
from qlr.config import Config
from qlr.data import load_htru2, HTRU2Preprocessor
from qlr.paths import baseline_cache_path

N_SAMPLES = 1000
SEEDS = [0, 1, 2]

# Metrics requested for the comparison, in reporting order.
METRICS = ["PR_AUC", "ROC_AUC", "Recall_FPR1", "Recall_FPR5", "Brier", "ECE"]

# caches/ stores the SVM under "CSVM"; use a reader-facing display name.
DISPLAY_NAME = {"LogReg": "LogReg", "CSVM": "SVM-RBF", "XGBoost": "XGBoost"}
MODEL_ORDER = ["LogReg", "CSVM", "XGBoost"]

OUT_DIR = Config.RESULTS_DIR / "baselines_weighting_N1000"


def summarize(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Metrics of interest for one model on one seed."""
    base = metrics_from_probs(y_true, y_prob)
    rec1, _, _, _ = metrics_at_fpr(y_true, y_prob, 0.01)
    rec5, _, _, _ = metrics_at_fpr(y_true, y_prob, 0.05)
    base["Recall_FPR1"] = float(rec1)
    base["Recall_FPR5"] = float(rec5)
    return {m: float(base[m]) for m in METRICS}


def crosscheck_against_cache(seed: int, probs: dict[str, np.ndarray]) -> dict[str, float]:
    """
    Max |Δp| between the freshly trained weighted probabilities and the
    corresponding reference-cache values. Read-only; a missing cache is not an
    error.
    """
    path = baseline_cache_path(N_SAMPLES, seed)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        cached = pickle.load(f)
    out = {}
    for name, p_new in probs.items():
        p_old = cached["probs"].get(name)
        if p_old is not None and len(p_old) == len(p_new):
            out[name] = float(np.max(np.abs(np.asarray(p_old, dtype=float) - p_new)))
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_htru2(Config.DATA_PATH)
    X_train, X_test, y_train, y_test = HTRU2Preprocessor(
        test_size=Config.TEST_SIZE, random_state=Config.DATA_SEED
    ).preprocess(df)

    rows: list[dict] = []
    raw_probs: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    drift: dict[int, dict[str, float]] = {}

    for variant, class_balanced in [("weighted", True), ("unweighted", False)]:
        for seed in SEEDS:
            print(f"\n=== {variant} | N={N_SAMPLES} seed={seed} ===", flush=True)
            t0 = time.perf_counter()
            probs, times = train_baselines_for_seed(
                N_SAMPLES, seed, X_train, y_train, X_test, y_test,
                class_balanced=class_balanced,
                include_qsvm=False,
            )
            print(f"[{variant}] seed={seed} done in {time.perf_counter() - t0:.1f}s", flush=True)

            raw_probs[(variant, seed)] = probs
            if variant == "weighted":
                drift[seed] = crosscheck_against_cache(seed, probs)

            for name in MODEL_ORDER:
                if name not in probs:
                    continue
                rows.append({
                    "variant": variant,
                    "model": DISPLAY_NAME[name],
                    "seed": seed,
                    "n_samples": N_SAMPLES,
                    **summarize(y_test, probs[name]),
                    "Train_s": float(times.get(f"{name}_train_s", np.nan)),
                })

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(OUT_DIR / "per_seed.csv", index=False)

    # mean +/- SE over the three seeds (SE = sd/sqrt(n_seeds), ddof=1,
    # matching qlr/aggregate.py).
    n_seeds = len(SEEDS)
    g = per_seed.groupby(["variant", "model"], as_index=False)
    mean = g[METRICS].mean(numeric_only=True)
    se = g[METRICS].std(numeric_only=True)
    se[METRICS] = se[METRICS] / np.sqrt(n_seeds)

    summary = mean.merge(se, on=["variant", "model"], suffixes=("_mean", "_se"))
    summary["model"] = pd.Categorical(
        summary["model"], [DISPLAY_NAME[m] for m in MODEL_ORDER], ordered=True
    )
    summary = summary.sort_values(["model", "variant"]).reset_index(drop=True)
    summary.to_csv(OUT_DIR / "summary_mean_se.csv", index=False)

    with open(OUT_DIR / "raw_test_probs.pkl", "wb") as f:
        pickle.dump({"y_true": y_test, "probs": raw_probs, "seeds": SEEDS,
                     "n_samples": N_SAMPLES}, f)

    # Side-by-side report
    lines: list[str] = []
    lines.append(f"Classical baselines at N={N_SAMPLES}, seeds {SEEDS} "
                 f"(mean +/- SE over {n_seeds} seeds)")
    lines.append("weighted   = class_weight='balanced' (LogReg, SVM-RBF), "
                 "scale_pos_weight=neg/pos (XGBoost)  [reported setup]")
    lines.append("unweighted = class_weight=None, scale_pos_weight=1        "
                 "[matches QLR's unweighted BCE]")
    lines.append("")
    header = f"{'model':<9}{'metric':<13}{'weighted':>22}{'unweighted':>22}{'delta (unw-w)':>16}"
    lines.append(header)
    lines.append("-" * len(header))

    for disp in [DISPLAY_NAME[m] for m in MODEL_ORDER]:
        for metric in METRICS:
            row = {}
            for variant in ("weighted", "unweighted"):
                r = summary[(summary["model"] == disp) & (summary["variant"] == variant)]
                if r.empty:
                    row[variant] = (np.nan, np.nan)
                else:
                    row[variant] = (float(r[f"{metric}_mean"].iloc[0]),
                                    float(r[f"{metric}_se"].iloc[0]))
            w_m, w_s = row["weighted"]
            u_m, u_s = row["unweighted"]
            lines.append(
                f"{disp:<9}{metric:<13}"
                f"{w_m:>13.4f} +/- {w_s:<6.4f}"
                f"{u_m:>13.4f} +/- {u_s:<6.4f}"
                f"{u_m - w_m:>+16.4f}"
            )
        lines.append("")

    if any(drift.values()):
        lines.append("Cross-check, freshly trained weighted arm vs cached "
                     "reference-cache probabilities (max |delta p| on the test set):")
        for seed in SEEDS:
            d = drift.get(seed, {})
            if d:
                parts = ", ".join(f"{DISPLAY_NAME[k]}={v:.2e}" for k, v in d.items())
                lines.append(f"  seed {seed}: {parts}")
        lines.append("")

    report = "\n".join(lines)
    (OUT_DIR / "report.txt").write_text(report + "\n")
    print("\n" + report)
    print(f"[OUT] wrote {OUT_DIR}/per_seed.csv, summary_mean_se.csv, "
          f"report.txt, raw_test_probs.pkl")


if __name__ == "__main__":
    main()
