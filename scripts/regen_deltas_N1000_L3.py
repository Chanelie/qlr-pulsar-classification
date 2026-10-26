# scripts/regen_deltas_N1000_L3.py
# ------------------------------------------------------------
# Regenerate the paired-delta tables at the benchmark config
# (N=1000, L=3) with the extended DELTA_METRICS list in
# qlr/aggregate.py (now including Brier and ECE, for which the
# best classical baseline is selected by minimum, not maximum).
#
# Uses cached artifacts only — no retraining. Writes NEW files:
#   tables/deltas_mean_N1000_L3_extended.csv
#   tables/deltas_se_N1000_L3_extended.csv
# The original deltas_*_N1000_L3.csv are left untouched.
# ------------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qlr.aggregate import aggregate_block
from qlr.cache import ArtifactManager
from qlr.config import Config
from qlr.data import load_htru2, HTRU2Preprocessor

N_SAMPLES = 1000
LAYERS = 3
SEEDS = [0, 1, 2]
ENCODINGS = ["amplitude", "angle", "data_reuploading"]


def main() -> None:
    df = load_htru2(Config.DATA_PATH)
    X_train, X_test, y_train, y_test = HTRU2Preprocessor(
        test_size=Config.TEST_SIZE, random_state=Config.DATA_SEED
    ).preprocess(df)

    block_cfgs = [
        {
            "regime": f"N{N_SAMPLES}",
            "n_samples": N_SAMPLES,
            "encoding": enc,
            "n_qubits": 3 if enc == "amplitude" else 8,
            "n_layers": LAYERS,
            "epochs": Config.EPOCHS,
            "seed": sd,
        }
        for enc in ENCODINGS
        for sd in SEEDS
    ]

    _, _, _, _, d_mean, d_se = aggregate_block(
        block_cfgs, ArtifactManager(), X_train, y_train, X_test, y_test
    )

    tag = f"N{N_SAMPLES}_L{LAYERS}"
    mean_path = Config.TABLES_DIR / f"deltas_mean_{tag}_extended.csv"
    se_path = Config.TABLES_DIR / f"deltas_se_{tag}_extended.csv"
    d_mean.to_csv(mean_path, index=False)
    d_se.to_csv(se_path, index=False)

    print(f"[OUT] wrote {mean_path}")
    print(f"[OUT] wrote {se_path}\n")
    merged = d_mean.merge(d_se, on=["encoding", "layers", "metric"], suffixes=("_mean", "_se"))
    print(merged.to_string(index=False))


if __name__ == "__main__":
    main()
