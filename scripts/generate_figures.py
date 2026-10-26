#!/usr/bin/env python3
"""Single figure-generation entry point for the active QLR project.

No model is trained here. Every figure is rebuilt from completed caches,
saved prediction bundles, and summary tables.

Examples (run from the repository root):

    python run.py figures
    python run.py figures --set thesis
    python run.py figures --set diagnostics
    python run.py figures --set all
    python run.py figures --set terminology
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qlr import plots as P  # noqa: E402
from qlr.config import Config  # noqa: E402


N_VALUES = [200, 500, 1000]
LAYERS = [1, 2, 3, 5, 10]
SEEDS = [0, 1, 2]
ENCODINGS = ["amplitude", "angle", "data_reuploading"]


def _load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _qlr_cache_path(n: int, encoding: str, layer: int, seed: int) -> Path:
    qubits = 3 if encoding == "amplitude" else 8
    return Config.CACHE_DIR / (
        f"N{n}_enc-{encoding}_q{qubits}_L{layer}_E{Config.EPOCHS}_seed{seed}.pkl"
    )


def collect_probabilities(
    n: int,
    layer: int,
    *,
    classical_weighting: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return paired, seed-averaged predictions for one N/L panel."""
    if classical_weighting not in {"weighted", "unweighted"}:
        raise ValueError("classical_weighting must be 'weighted' or 'unweighted'")

    accumulated: dict[str, list[np.ndarray]] = {
        **{f"QLR-{encoding}-L{layer}": [] for encoding in ENCODINGS},
        "LogReg": [],
        "CSVM": [],
        "XGBoost": [],
        "QSVM": [],
    }
    y_reference: np.ndarray | None = None

    for seed in SEEDS:
        subset_reference: np.ndarray | None = None
        for encoding in ENCODINGS:
            path = _qlr_cache_path(n, encoding, layer, seed)
            if not path.exists():
                raise FileNotFoundError(f"Missing QLR cache: {path}")
            payload = _load_pickle(path)
            meta = payload.get("meta", {})
            if (
                meta.get("subset_size") != n
                or meta.get("encoding") != encoding
                or meta.get("n_layers") != layer
                or meta.get("seed") != seed
            ):
                raise ValueError(f"Cache metadata mismatch: {path}")

            y_test = np.asarray(payload["y_test"])
            subset_idx = np.asarray(payload["subset_idx"])
            if y_reference is None:
                y_reference = y_test
            if not np.array_equal(y_test, y_reference):
                raise ValueError(f"Test-label mismatch: {path}")
            if subset_reference is None:
                subset_reference = subset_idx
            if not np.array_equal(subset_idx, subset_reference):
                raise ValueError(f"Training-subset mismatch within seed {seed}: {path}")

            probabilities = np.asarray(next(iter(payload["probs"].values())), dtype=float)
            accumulated[f"QLR-{encoding}-L{layer}"].append(probabilities)

        weighted_path = Config.CACHE_DIR / f"baselines_N{n}_seed{seed}.pkl"
        weighted = _load_pickle(weighted_path)
        if not np.array_equal(np.asarray(weighted["y_true"]), y_reference):
            raise ValueError(f"Weighted baseline test-label mismatch: {weighted_path}")

        if classical_weighting == "weighted":
            for model in ("LogReg", "CSVM", "XGBoost"):
                accumulated[model].append(np.asarray(weighted["probs"][model], dtype=float))
        else:
            path = (
                Config.RESULTS_DIR
                / "calibrated_baselines_train"
                / f"preds_N{n}_seed{seed}.pkl"
            )
            unweighted = _load_pickle(path)
            if (
                unweighted.get("N") != n
                or unweighted.get("seed") != seed
                or not np.array_equal(np.asarray(unweighted["y_test"]), y_reference)
                or not np.array_equal(np.asarray(unweighted["subset_idx"]), subset_reference)
            ):
                raise ValueError(f"Unweighted baseline pairing mismatch: {path}")
            for model in ("LogReg", "CSVM", "XGBoost"):
                key = (model, "unweighted", "none")
                accumulated[model].append(np.asarray(unweighted["probs"][key], dtype=float))

        # The saved QSVM uses class_weight="balanced" in both comparisons.
        accumulated["QSVM"].append(np.asarray(weighted["probs"]["QSVM"], dtype=float))

    if y_reference is None:
        raise RuntimeError(f"No predictions found for N={n}, L={layer}")
    means = {name: np.vstack(values).mean(axis=0) for name, values in accumulated.items()}
    return y_reference, means


def required_tables() -> list[Path]:
    tables = Config.TABLES_DIR
    paths = [
        *(tables / f"calibration_N{n}_all_layers.csv" for n in N_VALUES),
        *(tables / f"runtime_N{n}_all_layers.csv" for n in N_VALUES),
        *(tables / f"baselines_mean_N{n}.csv" for n in N_VALUES),
        *(tables / f"baselines_se_N{n}.csv" for n in N_VALUES),
        *(tables / f"calibrated_baselines_train_N{n}.csv" for n in N_VALUES),
    ]
    paths.extend(
        tables / f"qlr_{kind}_N{n}_L{layer}.csv"
        for kind in ("mean", "se")
        for n in N_VALUES
        for layer in LAYERS
    )
    return paths


def verify_inputs() -> None:
    missing = [path for path in required_tables() if not path.exists()]
    missing.extend(
        path
        for n in N_VALUES
        for encoding in ENCODINGS
        for layer in LAYERS
        for seed in SEEDS
        if not (path := _qlr_cache_path(n, encoding, layer, seed)).exists()
    )
    missing.extend(
        Config.CACHE_DIR / f"baselines_N{n}_seed{seed}.pkl"
        for n in N_VALUES
        for seed in SEEDS
        if not (Config.CACHE_DIR / f"baselines_N{n}_seed{seed}.pkl").exists()
    )
    missing.extend(
        Config.RESULTS_DIR / "calibrated_baselines_train" / f"preds_N{n}_seed{seed}.pkl"
        for n in N_VALUES
        for seed in SEEDS
        if not (
            Config.RESULTS_DIR
            / "calibrated_baselines_train"
            / f"preds_N{n}_seed{seed}.pkl"
        ).exists()
    )
    if missing:
        raise SystemExit("Missing required figure input(s):\n  " + "\n  ".join(map(str, missing)))


def generate_thesis(out_dir: Path) -> None:
    tables = Config.TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    unweighted_tables = {
        n: tables / f"calibrated_baselines_train_N{n}.csv" for n in N_VALUES
    }
    calibration_tables = {
        n: tables / f"calibration_N{n}_all_layers.csv" for n in N_VALUES
    }
    runtime_tables = {n: tables / f"runtime_N{n}_all_layers.csv" for n in N_VALUES}

    P.plot_depth_scaling(
        n_values=N_VALUES,
        layers=LAYERS,
        tables_dir=tables,
        out_dir=out_dir,
        unweighted_reference_files=unweighted_tables,
    )
    P.plot_ece_vs_depth(
        calibration_files=calibration_tables,
        out_dir=out_dir,
        unweighted_reference_files=unweighted_tables,
    )

    reliability = {
        layer: collect_probabilities(1000, layer, classical_weighting="unweighted")
        for layer in (1, 3, 10)
    }
    P.plot_reliability_combined(
        reliability,
        n_samples=1000,
        out_dir=out_dir,
        display_labels=P.UNWEIGHTED_THESIS_DISPLAY_LABELS,
        show_inset=False,
        stem="reliability_combined_N1000_unweighted",
    )
    P.plot_runtime_vs_data(runtime_tables, fixed_layer=3, out_dir=out_dir)
    P.plot_runtime_vs_depth(runtime_tables, fixed_n=1000, out_dir=out_dir)
    P.plot_murphy_depth(
        cache_dir=Config.CACHE_DIR,
        n_values=N_VALUES,
        layers=LAYERS,
        encodings=["angle", "data_reuploading"],
        qubits_map={"angle": 8, "data_reuploading": 8},
        epochs=Config.EPOCHS,
        n_seeds=len(SEEDS),
        out_dir=out_dir,
    )


def generate_terminology(out_dir: Path) -> None:
    """Regenerate only thesis figures affected by terminology changes."""
    tables = Config.TABLES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    calibration_tables = {
        n: tables / f"calibration_N{n}_all_layers.csv" for n in N_VALUES
    }
    unweighted_tables = {
        n: tables / f"calibrated_baselines_train_N{n}.csv" for n in N_VALUES
    }
    P.plot_ece_vs_depth(
        calibration_files=calibration_tables,
        out_dir=out_dir,
        unweighted_reference_files=unweighted_tables,
    )


def generate_diagnostics(out_dir: Path) -> None:
    tables = Config.TABLES_DIR
    shared = out_dir / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    calibration_tables = {
        n: tables / f"calibration_N{n}_all_layers.csv" for n in N_VALUES
    }
    runtime_tables = {n: tables / f"runtime_N{n}_all_layers.csv" for n in N_VALUES}

    P.plot_depth_scaling(N_VALUES, LAYERS, tables, shared)
    P.plot_ece_vs_depth(calibration_tables, shared)
    P.plot_runtime_vs_data(runtime_tables, fixed_layer=3, out_dir=shared)
    P.plot_murphy_depth(
        cache_dir=Config.CACHE_DIR,
        n_values=N_VALUES,
        layers=LAYERS,
        encodings=["angle", "data_reuploading"],
        qubits_map={"angle": 8, "data_reuploading": 8},
        epochs=Config.EPOCHS,
        n_seeds=len(SEEDS),
        out_dir=shared,
    )

    for n in N_VALUES:
        n_dir = out_dir / f"N{n}"
        n_dir.mkdir(parents=True, exist_ok=True)
        P.plot_runtime_vs_depth(runtime_tables, fixed_n=n, out_dir=n_dir)
        reliability_layers = {
            layer: collect_probabilities(n, layer, classical_weighting="weighted")
            for layer in (1, 3, 10)
        }
        P.plot_reliability_combined(
            reliability_layers,
            n_samples=n,
            out_dir=n_dir,
            display_labels=P.WEIGHTED_DISPLAY_LABELS,
            show_inset=True,
        )

        baseline_mean = pd.read_csv(tables / f"baselines_mean_N{n}.csv")
        baseline_se = pd.read_csv(tables / f"baselines_se_N{n}.csv")
        for layer in LAYERS:
            layer_dir = n_dir / f"L{layer}"
            qlr_mean = pd.read_csv(tables / f"qlr_mean_N{n}_L{layer}.csv")
            qlr_se = pd.read_csv(tables / f"qlr_se_N{n}_L{layer}.csv")
            y_true, probabilities = collect_probabilities(
                n, layer, classical_weighting="weighted"
            )
            P.plot_bar_metrics(
                qlr_mean, qlr_se, baseline_mean, baseline_se,
                n, layer, layer_dir,
            )
            P.plot_roc_pr(y_true, probabilities, n, layer, layer_dir)
            P.plot_reliability(y_true, probabilities, n, layer, layer_dir)
            P.plot_confusions(y_true, probabilities, n, layer, layer_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate QLR figures from saved results; never trains models."
    )
    parser.add_argument(
        "--set",
        choices=("thesis", "diagnostics", "terminology", "all"),
        default="thesis",
        help="figure set to rebuild (default: thesis)",
    )
    parser.add_argument(
        "--thesis-out",
        type=Path,
        default=Config.RESULTS_DIR / "figs" / "thesis",
    )
    parser.add_argument(
        "--diagnostics-out",
        type=Path,
        default=Config.RESULTS_DIR / "figs" / "diagnostics",
    )
    parser.add_argument(
        "--terminology-out",
        type=Path,
        default=(
            Config.RESULTS_DIR
            / "figs"
            / "terminology_standardised"
        ),
    )
    args = parser.parse_args()

    verify_inputs()
    if args.set in {"thesis", "all"}:
        generate_thesis(args.thesis_out)
        print(f"Thesis figures: {args.thesis_out}")
    if args.set in {"diagnostics", "all"}:
        generate_diagnostics(args.diagnostics_out)
        print(f"Diagnostic figures: {args.diagnostics_out}")
    if args.set in {"terminology", "all"}:
        generate_terminology(args.terminology_out)
        print(f"Terminology-standardised figures: {args.terminology_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
