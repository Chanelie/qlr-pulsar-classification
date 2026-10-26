# qlr/runner.py
# ------------------------------------------------------------
# Per-config QLR training 
# Logging + runtime tracking (train/predict).
# ------------------------------------------------------------

from __future__ import annotations

from typing import List, Tuple, Dict
import traceback
from pathlib import Path
import time
import numpy as np
import pickle

from .config import Config
from .data import subset_indices_for
from .quantum import QuantumCircuitDesigner
from .qlr_model import HybridQLRClassifier
from .cache import ArtifactManager, config_slug
from .aggregate import aggregate_block
from .baselines import train_baselines_for_seed
from .paths import baseline_cache_path

# Import from plots.py
from .plots import (
    collect_mean_probs_for_block_layer,
    plot_bar_metrics,
    plot_roc_pr,
    plot_reliability,
    plot_reliability_combined,
    plot_confusions,
    plot_ece_vs_depth,
    plot_runtime_vs_data,
    plot_runtime_vs_depth,
    plot_murphy_depth,
    plot_depth_scaling,
    WEIGHTED_DISPLAY_LABELS,
)

# Logging controls
LOG_LEVEL = "info"  # {"silent", "info", "debug"}

def log_msg(msg: str, level: str = "info") -> None:
    if LOG_LEVEL == "silent":
        return
    if LOG_LEVEL == "info" and level == "debug":
        return
    print(msg, flush=True)

# Baseline caching
def save_baselines_for_block(
    block_cfgs: List[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    seeds = sorted({cfg["seed"] for cfg in block_cfgs})
    ns_vals = sorted({cfg["n_samples"] for cfg in block_cfgs})
    if len(ns_vals) != 1:
        raise ValueError("save_baselines_for_block expects one n_samples value")
    ns = ns_vals[0]

    Path(Config.CACHE_DIR).mkdir(parents=True, exist_ok=True)

    for sd in seeds:
        path = baseline_cache_path(ns, sd)

        if path.exists():
            log_msg(f"[BL-CACHE] hit ns={ns} seed={sd}", level="info")
            continue

        log_msg(f"[BL-CACHE] train ns={ns} seed={sd}", level="info")
        bl_probs, bl_times = train_baselines_for_seed(ns, sd, X_train, y_train, X_test, y_test)

        with open(path, "wb") as f:
            pickle.dump(
                {
                    "y_true": y_test,
                    "probs": bl_probs,
                    "times": bl_times,
                },
                f,
            )

        log_msg(f"[BL-CACHE] saved {path.name}", level="info")


# Single-config training
def train_single_config(
    cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple:
    slug = config_slug(cfg)

    fail_marker = Path(Config.RESULTS_DIR) / "failures" / f"{slug}.txt"
    fail_marker.parent.mkdir(parents=True, exist_ok=True)

    try:
        sub_idx = subset_indices_for(cfg.get("n_samples"), cfg["seed"], X_train, y_train)
        X_sub, y_sub = X_train[sub_idx], y_train[sub_idx]
        log_msg(f"[DATA]  {slug} subset={len(X_sub)}", level="debug")

        log_msg(
            f"[START] {slug} | n={cfg.get('n_samples','full')} enc={cfg['encoding']} "
            f"L={cfg['n_layers']} seed={cfg['seed']} subset={len(X_sub)}",
            level="info",
        )

        qc = QuantumCircuitDesigner(
            n_qubits=cfg["n_qubits"],
            n_layers=cfg["n_layers"],
            encoding=cfg["encoding"],
            device_name="lightning.qubit",
            seed=cfg["seed"],  # seed-scoped theta init, independent of run order
        )

        log_msg(
            f"[MODEL] {slug} enc={cfg['encoding']} q={cfg['n_qubits']} "
            f"L={cfg['n_layers']} epochs={cfg['epochs']}",
            level="debug",
        )

        qlr = HybridQLRClassifier(
            qc,
            learning_rate=cfg.get("learning_rate", Config.LEARNING_RATE),
            epochs=cfg["epochs"],
            batch_size=cfg.get("batch_size", Config.BATCH_SIZE),
            l2=cfg.get("l2", Config.L2_REG),
            class_weight=None,
            tolerance=cfg.get("tolerance", Config.TOLERANCE),
            max_patience=cfg.get("max_patience", Config.MAX_PATIENCE),
            verbose=False,
            log_every=10,
            seed=cfg["seed"],  # seeds head init and the minibatch shuffle
        ).fit(X_sub, y_sub)

        if LOG_LEVEL == "debug" and qlr.loss_history:
            head = qlr.loss_history[:3]
            tail = qlr.loss_history[-3:] if len(qlr.loss_history) > 3 else []
            log_msg(f"[DEBUG] {slug} loss_hist_len={len(qlr.loss_history)}", level="debug")
            log_msg(f"[DEBUG] {slug} first_losses={', '.join(f'{v:.6f}' for v in head)}", level="debug")
            if tail:
                log_msg(f"[DEBUG] {slug} last_losses={', '.join(f'{v:.6f}' for v in tail)}", level="debug")

        t0 = time.perf_counter()
        y_prob = qlr.predict_proba(X_test)
        pred_time = time.perf_counter() - t0

        probs_dict = {f"QLR-{cfg['encoding']}-{cfg['n_qubits']}q-{cfg['n_layers']}L": y_prob}
        
        meta = dict(
            regime=cfg.get("regime", "run"),
            encoding=cfg["encoding"],
            n_qubits=cfg["n_qubits"],
            n_layers=cfg["n_layers"],
            epochs=cfg["epochs"],
            seed=cfg["seed"],
            subset_size=int(len(X_sub)),
            train_time_s=float(getattr(qlr, "training_time", np.nan)),
            pred_time_s=float(pred_time),
            device=str(qc.dev.name),
            diff_method="parameter-shift",
        )

        log_msg(
            f"[DONE]  {slug} | train={meta['train_time_s']:.2f}s pred={meta['pred_time_s']:.2f}s",
            level="info",
        )

        return y_test, probs_dict, qlr.loss_history, sub_idx, meta
    
    except Exception as e:
        tb = traceback.format_exc()
        log_msg(f"[FAIL] {slug} | {type(e).__name__}: {e}", level="info")
        log_msg(tb, level="debug")
        fail_marker.write_text(tb)
        raise

def run_experiments(
    configs: List[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    overwrite: bool = False,
) -> Dict[str, Tuple]:
    AM = ArtifactManager()
    results: Dict[str, Tuple] = {}

    configs_cached = [cfg for cfg in configs if not overwrite and AM.exists(cfg)]
    configs_to_run = [cfg for cfg in configs if overwrite or not AM.exists(cfg)]

    log_msg(
        f"[PLAN] total={len(configs)} cached={len(configs_cached)} to_run={len(configs_to_run)}",
        level="info",
    )

    for cfg in configs_cached:
        slug = config_slug(cfg)
        results[slug] = AM.load(cfg)
        log_msg(f"[CACHE] {slug}", level="debug")

    if not configs_to_run:
        log_msg("[PLAN] nothing to run (all cached).", level="info")
        return results

    for cfg in configs_to_run:
        slug = config_slug(cfg)
        try:
            result = train_single_config(cfg, X_train, y_train, X_test, y_test)
            AM.save(cfg, *result)
            results[slug] = result
            log_msg(f"[SAVE]  {slug}", level="debug")
        except Exception as e:
            log_msg(f"[FAIL]  {slug} | {e}", level="info")

    return results

# Block runner
def run_sample_block(
    n_samples: int | None,
    configs: List[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    make_plots: bool = False,
    plot_layer: int | None = None,
    write_tables: bool = True,
    figs_dir: Path | None = None,
) -> None:
    block_cfgs = [cfg for cfg in configs if cfg["n_samples"] == n_samples]
    if not block_cfgs:
        log_msg(f"[SKIP] n_samples={n_samples} (no configs)", level="info")
        return

    all_layers = sorted({cfg["n_layers"] for cfg in block_cfgs})
    if plot_layer is not None:
        all_layers = [L for L in all_layers if L == plot_layer]

    figs_root = figs_dir or (Path("results") / "figs")
    if make_plots:
        figs_root.mkdir(parents=True, exist_ok=True)

    save_baselines_for_block(block_cfgs, X_train, y_train, X_test, y_test)

    for L in all_layers:
        layer_cfgs = [cfg for cfg in block_cfgs if cfg["n_layers"] == L]
        if not layer_cfgs:
            continue

        log_msg(f"[BLOCK] n={n_samples} L={L} configs={len(layer_cfgs)}", level="info")
        run_experiments(layer_cfgs, X_train, y_train, X_test, y_test, overwrite=False)

    log_msg(f"[AGG] start aggregate_block n={n_samples} (all layers)", level="info")
    log_msg(
        f"[AGG] expecting up to {len(block_cfgs)} payloads for n={n_samples} (all layers)",
        level="info",
    )

    try:
        qlr_mean, qlr_se, bl_mean, bl_se, d_mean, d_se = aggregate_block(
            block_cfgs, ArtifactManager(), X_train, y_train, X_test, y_test
        )
        log_msg(f"[AGG] done aggregate_block n={n_samples} (all layers)", level="info")
    except Exception as e:
        log_msg(f"[AGG][FAIL] n={n_samples} | {type(e).__name__}: {e}", level="info")
        raise

    if qlr_mean.empty or bl_mean.empty:
        log_msg(f"[WARN] n={n_samples} (no metrics; skipping tables/plots)", level="info")
        return

    out = Config.TABLES_DIR
    out.mkdir(parents=True, exist_ok=True)


    # Order the different models by predefined rank (LogReg -> CSVM -> XGBoost -> QSVM)
    ORDER = {"LogReg": 0, "CSVM": 1, "XGBoost": 2, "QSVM": 99}
    bl_mean = bl_mean.sort_values(["model"], key=lambda s: s.map(lambda m: ORDER.get(m, 50))).reset_index(drop=True)
    bl_se   = bl_se.sort_values(["model"],  key=lambda s: s.map(lambda m: ORDER.get(m, 50))).reset_index(drop=True)

    block_tag = f"N{n_samples}" if n_samples is not None else "Full"
    n_dir = figs_root / (f"N{n_samples}" if n_samples is not None else "NFull")

    if write_tables:
        bl_mean.to_csv(out / f"baselines_mean_{block_tag}.csv", index=False)
        bl_se.to_csv(out / f"baselines_se_{block_tag}.csv", index=False)

    for L in all_layers:
        qlr_mean_L = qlr_mean[qlr_mean["layers"] == L].copy()
        qlr_se_L   = qlr_se[qlr_se["layers"] == L].copy()
        d_mean_L   = d_mean[d_mean["layers"] == L].copy()
        d_se_L     = d_se[d_se["layers"] == L].copy()

        if qlr_mean_L.empty:
            log_msg(f"[WARN] n={n_samples} L={L} (no QLR rows after aggregation)", level="info")
            continue

        tag = f"N{n_samples}_L{L}" if n_samples is not None else f"Full_L{L}"

        if write_tables:
            qlr_mean_L.to_csv(out / f"qlr_mean_{tag}.csv", index=False)
            qlr_se_L.to_csv(out / f"qlr_se_{tag}.csv", index=False)
            d_mean_L.to_csv(out / f"deltas_mean_{tag}.csv", index=False)
            d_se_L.to_csv(out / f"deltas_se_{tag}.csv", index=False)
            log_msg(f"[TABLES] wrote {tag}", level="info")

        if make_plots:
            sample_tag = f"N{n_samples}" if n_samples is not None else "NFull"
            layer_dir = figs_root / sample_tag / f"L{L}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            log_msg(f"[PLOTS] n={n_samples} L={L} -> {layer_dir}", level="info")

            plot_bar_metrics(qlr_mean_L, qlr_se_L, bl_mean, bl_se, n_samples, L, layer_dir)

            y_true, mean_probs = collect_mean_probs_for_block_layer(
                block_cfgs, ArtifactManager(), X_train, y_train, X_test, y_test, L
            )
            plot_roc_pr(y_true, mean_probs, n_samples, L, layer_dir)
            plot_reliability(y_true, mean_probs, n_samples, L, layer_dir)
            plot_confusions(y_true, mean_probs, n_samples, L, layer_dir)

        
    if make_plots:
        n_dir = figs_root / (f"N{n_samples}" if n_samples is not None else "NFull")

        # Combined reliability figure for the paper: L in {1, 3, 10}
        reliability_layers = [L for L in all_layers if L in {1, 3, 10}]
        if reliability_layers:
            layers_data = {}
            for L in reliability_layers:
                y_true_L, mean_probs_L = collect_mean_probs_for_block_layer(
                    block_cfgs,
                    ArtifactManager(),
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    layer=L,
                )
                layers_data[L] = (y_true_L, mean_probs_L)

            plot_reliability_combined(
                layers_data=layers_data,
                n_samples=n_samples,
                out_dir=n_dir,
                display_labels=WEIGHTED_DISPLAY_LABELS,
            )

        plot_ece_vs_depth(
            calibration_files={
                200:  Config.TABLES_DIR / "calibration_N200_all_layers.csv",
                500:  Config.TABLES_DIR / "calibration_N500_all_layers.csv",
                1000: Config.TABLES_DIR / "calibration_N1000_all_layers.csv",
            },
            out_dir=n_dir,
        )

        runtime_files = {
            200:  Config.TABLES_DIR / "runtime_N200_all_layers.csv",
            500:  Config.TABLES_DIR / "runtime_N500_all_layers.csv",
            1000: Config.TABLES_DIR / "runtime_N1000_all_layers.csv",
        }

        plot_runtime_vs_data(
            runtime_files=runtime_files,
            fixed_layer=3,
            out_dir=n_dir,
        )

        plot_runtime_vs_depth(
            runtime_files=runtime_files,
            fixed_n=n_samples,
            out_dir=n_dir,
        )

        plot_murphy_depth(
            cache_dir  = Config.CACHE_DIR,
            n_values   = [200, 500, 1000],
            layers     = [1, 2, 3, 5, 10],
            encodings  = ["angle", "data_reuploading"],
            qubits_map = {"angle": 8, "data_reuploading": 8},
            epochs     = 60,
            n_seeds    = 3,
            out_dir    = n_dir,
        )

        plot_depth_scaling(
            n_values   = [200, 500, 1000],
            layers     = [1, 2, 3, 5, 10],
            tables_dir = Config.TABLES_DIR,
            out_dir    = n_dir,
        )



        if LOG_LEVEL == "debug":
            log_msg(f"\n=== Baselines (avg over seeds) @ n={n_samples} ===", level="debug")
            log_msg(
                bl_mean.sort_values(["model"])[["model", "ROC_AUC", "PR_AUC", "F1", "Brier"]]
                .to_string(index=False),
                level="debug",
            )

            log_msg(f"\n=== Δ(QLR - BEST_BASELINE) ROC_AUC @ n={n_samples}, L={L} ===", level="debug")
            log_msg(
                d_mean_L.sort_values(["encoding", "layers"])[["encoding", "layers", "delta"]]
                .to_string(index=False),
                level="debug",
            )

