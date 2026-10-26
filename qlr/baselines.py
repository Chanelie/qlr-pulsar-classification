# qlr/baselines.py
# ------------------------------------------------------------
# Classical baselines (LogReg, SVM-RBF, XGBoost) and the QSVM
# reference, trained per (n_samples, seed) on the same stratified
# subset as QLR (paired-seed protocol).
# ------------------------------------------------------------

from __future__ import annotations
from typing import Dict, Tuple
import time
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .data import subset_indices_for
from .qsvm import qsvm_predict_proba_timed
from .config import Config


def train_baselines_for_seed(
    n_samples: int | None,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    class_balanced: bool = True,
    include_qsvm: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """
    Train the classical baselines (and QSVM where enabled) for one
    (n_samples, seed) pair and return test-set probabilities and runtimes.

    Parameters
    ----------
    class_balanced : bool, default True
        Imbalance handling for the classical baselines. The default
        reproduces the reported (weighted) setup exactly:
        ``class_weight="balanced"`` for LogReg and SVM-RBF, and
        ``scale_pos_weight = n_negative / n_positive`` for XGBoost.
        Set to False for the unweighted variant (``class_weight=None``,
        ``scale_pos_weight=1``), which matches QLR's unweighted BCE loss
        and so removes the weighting asymmetry from calibration
        comparisons. QSVM is unaffected by this switch: its precomputed-kernel
        SVC is fitted with ``class_weight="balanced"`` in both cases.
    include_qsvm : bool, default True
        When False, the QSVM reference is skipped regardless of
        ``Config.QSVM_ENABLED_SIZES``. Useful for the classical-only
        weighting comparison, where QSVM adds hours of kernel evaluation
        and does not depend on the weighting choice.

    Returns
    -------
    probs : dict mapping model name -> np.ndarray of p(y=1|x) on X_test
    times : dict mapping "<Model>_train_s" / "<Model>_pred_s" -> float
    """
    eff_n = n_samples or len(X_train)

    # Paired subset (same indices as the QLR run for this seed)
    sub_idx = subset_indices_for(n_samples, seed, X_train, y_train)
    Xb, yb = X_train[sub_idx], y_train[sub_idx]

    # Degenerate subset guard: a single class present cannot be fitted
    classes = np.unique(yb)
    if classes.size < 2:
        probs: Dict[str, np.ndarray] = {
            "LogReg": np.full(len(X_test), 0.5, dtype=float),
            "CSVM": np.full(len(X_test), 0.5, dtype=float),
            "XGBoost": np.full(len(X_test), 0.5, dtype=float),
        }
        times: Dict[str, float] = {
            "LogReg_train_s": np.nan,
            "LogReg_pred_s": np.nan,
            "CSVM_train_s": np.nan,
            "CSVM_pred_s": np.nan,
            "XGBoost_train_s": np.nan,
            "XGBoost_pred_s": np.nan,
            "QSVM_train_s": np.nan,
            "QSVM_pred_s": np.nan,
        }
        return probs, times

    # Runtime dictionary
    times: Dict[str, float] = {}

    # Emphasis on PR-AUC
    cv_scoring_metric = 'average_precision'

    # Imbalance handling, selected once and shared by the three classical models.
    # class_balanced=True is the reported setup; False is the unweighted variant.
    sk_class_weight = "balanced" if class_balanced else None
    weight_tag = "weighted" if class_balanced else "unweighted"

    # Logistic Regression
    print(f"[BL] LogReg (Tuning, {weight_tag}) start", flush=True)
    t0 = time.perf_counter()
    lr_grid = {'C': [0.1, 1.0, 10.0]}
    lr_search = GridSearchCV(
        LogisticRegression(solver="lbfgs", max_iter=2000, class_weight=sk_class_weight, random_state=seed),
        param_grid=lr_grid, scoring=cv_scoring_metric, cv=3, n_jobs=1
    )
    lr_search.fit(Xb, yb)
    best_lr = lr_search.best_estimator_
    times["LogReg_train_s"] = time.perf_counter() - t0
    print(f"[BL] LogReg done ({times['LogReg_train_s']:.2f}s) - Best params: {lr_search.best_params_}", flush=True)

    t0 = time.perf_counter()
    p_lr = best_lr.predict_proba(X_test)[:, 1].astype(float)
    times["LogReg_pred_s"] = time.perf_counter() - t0
    print(f"[BL] LogReg predict done ({times['LogReg_pred_s']:.2f}s)", flush=True)

    # SVM with RBF kernel
    # Imbalance handling: class_weight only. (scale_pos_weight is an XGBoost
    # parameter and does not exist for sklearn's SVC.)
    print(f"[BL] CSVM (Tuning, {weight_tag}) start", flush=True)
    t0 = time.perf_counter()
    svm_grid = {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']}
    svm_search = GridSearchCV(
        SVC(kernel="rbf", probability=True, class_weight=sk_class_weight, random_state=seed),
        param_grid=svm_grid, scoring=cv_scoring_metric, cv=3, n_jobs=1
    )
    svm_search.fit(Xb, yb)
    best_svm = svm_search.best_estimator_
    times["CSVM_train_s"] = time.perf_counter() - t0
    print(f"[BL] CSVM done ({times['CSVM_train_s']:.2f}s) - Best params: {svm_search.best_params_}", flush=True)

    t0 = time.perf_counter()
    p_svm = best_svm.predict_proba(X_test)[:, 1].astype(float)
    times["CSVM_pred_s"] = time.perf_counter() - t0
    print(f"[BL] CSVM predict done ({times['CSVM_pred_s']:.2f}s)", flush=True)

    # XGBoost
    # Imbalance handling: scale_pos_weight = n_negative / n_positive
    # (weighted variant) or 1.0 (unweighted variant).
    print(f"[BL] XGBoost (Tuning, {weight_tag}) start", flush=True)
    pos = float((yb == 1).sum())
    neg = float((yb == 0).sum())
    spw = ((neg / pos) if pos > 0 else 1.0) if class_balanced else 1.0

    t0 = time.perf_counter()
    xgb_grid = {'max_depth': [3, 5], 'learning_rate': [0.01, 0.1], 'n_estimators': [50, 100]}
    xgb_search = GridSearchCV(
        XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", random_state=seed, n_jobs=1),
        param_grid=xgb_grid, scoring=cv_scoring_metric, cv=3, n_jobs=1
    )
    xgb_search.fit(Xb, yb)
    best_xgb = xgb_search.best_estimator_
    times["XGBoost_train_s"] = time.perf_counter() - t0
    print(f"[BL] XGBoost done ({times['XGBoost_train_s']:.2f}s) - Best params: {xgb_search.best_params_}", flush=True)

    t0 = time.perf_counter()
    p_xgb = best_xgb.predict_proba(X_test)[:, 1].astype(float)
    times["XGBoost_pred_s"] = time.perf_counter() - t0
    print(f"[BL] XGBoost predict done ({times['XGBoost_pred_s']:.2f}s)", flush=True)

    # Base output probabilities
    probs: Dict[str, np.ndarray] = {
        "LogReg": p_lr,
        "CSVM": p_svm,
        "XGBoost": p_xgb,
    }

    # QSVM (IQP fidelity kernel), only at the enabled sample sizes
    if include_qsvm and eff_n in getattr(Config, "QSVM_ENABLED_SIZES", (200, 500, 1000)):
        print("[BL] QSVM start", flush=True)

        p_qsvm, qsvm_train_s, qsvm_pred_s = qsvm_predict_proba_timed(
            X_train=Xb,
            y_train=yb,
            X_test=X_test,
            n_qubits=8,
            C=1.0,
            seed=seed,
            angle_scale=np.pi / 8,
            n_repeats=1,
            device_name="lightning.qubit",
            enforce_psd=True,
            normalize_kernel=True,
        )

        probs["QSVM"] = p_qsvm.astype(float)
        times["QSVM_train_s"] = float(qsvm_train_s)
        times["QSVM_pred_s"] = float(qsvm_pred_s)
        print(f"[BL] QSVM done ({times['QSVM_train_s']:.2f}s)", flush=True)
        print(f"[BL] QSVM predict done ({times['QSVM_pred_s']:.2f}s)", flush=True)

    times.setdefault("QSVM_train_s", np.nan)
    times.setdefault("QSVM_pred_s", np.nan)

    return probs, times
