# qlr/config.py
# ------------------------------------------------------------
# Central configuration + one-shot environment metadata writer
# Keep this small and “dumb”; override values from pulsar_qlr.py via CLI.
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import math
import os
import json
import platform
import typing as _t

import pennylane as qml


class Config:
    """Project-wide defaults. Override in main() via CLI to keep provenance stable."""

    # Data & experiment setup 
    DATA_STEM: str = "HTRU2"
    DATA_SEED: int = 42
    TEST_SIZE: float = 0.20  # fraction of data to hold out for test

    # Define configuration
    # Note: N_QUBITS is now set dynamically in pulsar_qlr.py based on encoding
    N_QUBITS: int = 8
    
    ENCODINGS_DEFAULT: list[str] = ["amplitude", "angle", "data_reuploading"]

    # Angle-embedding scale factor α: features are encoded as RY(α · x_i).
    # Used by the QLR angle and data-reuploading encodings (see quantum.py).
    # Note: the QSVM baseline kernel uses a separately hardcoded scale of
    # π/8 (see baselines.py / qsvm.py); the two are intentionally distinct.
    ANGLE_SCALE: float = math.pi / 4

    LAYER_CONFIGS: list[int] = [1, 2, 3, 5, 10]
    SEEDS: list[int] = [0, 1, 2]
    N_SAMPLES_LIST: list[_t.Union[int, None]] = [200, 500, 1000, 5000, 10000]  # None for full set

    # Training hyperparameters 
    EPOCHS: int = 60
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 0.01
    L2_REG: float = 1e-4
    TOLERANCE: float = 1e-6
    MAX_PATIENCE: int = 15
    
    # Control which sample sizes trigger the expensive QSVM baseline
    QSVM_ENABLED_SIZES: tuple[int, ...] = (50, 200, 500, 1000)

    # Output paths to save results
    # Public repository layout: qlr/, scripts/, data/, tables/, and results/
    # all live directly under the repository root.
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent
    REPOSITORY_ROOT: Path = ROOT_DIR
    DATA_PATH: Path = Path(
        os.environ.get("QLR_DATA_PATH", REPOSITORY_ROOT / "data" / "HTRU2.csv")
    ).resolve()
    CACHE_DIR: Path = ROOT_DIR / "caches"
    RESULTS_DIR: Path = ROOT_DIR / "results"
    MODELS_DIR: Path = ROOT_DIR / "models"
    TABLES_DIR: Path = ROOT_DIR / "tables"
    META_PATH: Path = RESULTS_DIR / "environment_meta.json"

    # Thread env for BLAS/OpenMP hygiene (export in shell for best effect) 
    THREAD_ENV_VARS: dict[str, str] = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }

    @classmethod
    def setup_directories(cls) -> None:
        """Create standard output folders if missing."""
        for p in (cls.CACHE_DIR, cls.RESULTS_DIR, cls.MODELS_DIR, cls.TABLES_DIR):
            p.mkdir(parents=True, exist_ok=True)


def write_environment_meta_once() -> None:
    """
    Write a one-time JSON snapshot of the runtime environment for reproducibility.
    Skips writing if the file already exists.
    """
    Config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if Config.META_PATH.exists():
        return

    meta = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "pennylane": qml.__version__,
        "backends": ["lightning.qubit (strictly enforced)"],
        "diff_method": "parameter-shift",
        "threads": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }
    with open(Config.META_PATH, "w") as f:
        json.dump(meta, f, indent=2)
