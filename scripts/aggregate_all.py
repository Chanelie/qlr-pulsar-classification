#!/usr/bin/env python3
"""
The single aggregation + plotting pass, run ONCE after the grid completes.

run_one.py deliberately does none of this: with 8 concurrent jobs, writing
tables/*.csv and results/figs/** from each job would race and produce
partial, interleaved output. Every config is trained first; this reads the
finished caches and produces the tables and figures in one shot.

  ./scripts/aggregate_all.py --check     report cache coverage, write nothing
  ./scripts/aggregate_all.py             aggregate everything present
  ./scripts/aggregate_all.py --samples 1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qlr.config import Config                                    # noqa: E402
from qlr.cache import ArtifactManager, config_slug               # noqa: E402
from qlr.data import load_htru2, HTRU2Preprocessor               # noqa: E402
from qlr.paths import baseline_cache_path                        # noqa: E402
from qlr.runner import run_sample_block                          # noqa: E402
from qlr import plots as P                                       # noqa: E402

ENCODINGS = ["amplitude", "angle", "data_reuploading"]
LAYERS = [1, 2, 3, 5, 10]
SEEDS = [0, 1, 2]
SAMPLES = [200, 500, 1000]


def build_configs(samples: list[int]) -> list[dict]:
    return [
        {
            "regime": f"N{n}", "n_samples": n, "encoding": enc,
            "n_qubits": 3 if enc == "amplitude" else 8,
            "n_layers": L, "epochs": Config.EPOCHS, "seed": sd,
        }
        for n in samples for enc in ENCODINGS for L in LAYERS for sd in SEEDS
    ]


def coverage(configs: list[dict]) -> tuple[list[str], list[str]]:
    AM = ArtifactManager()
    missing_qlr = [config_slug(c) for c in configs if not AM.exists(c)]
    missing_bl = [
        baseline_cache_path(n, sd).name
        for n in sorted({c["n_samples"] for c in configs}) for sd in SEEDS
        if not baseline_cache_path(n, sd).exists()
    ]
    return missing_qlr, missing_bl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, nargs="*", default=SAMPLES)
    ap.add_argument("--check", action="store_true",
                    help="report cache coverage and exit without writing")
    ap.add_argument("--force", action="store_true",
                    help="aggregate even with caches missing (partial tables)")
    args = ap.parse_args()

    configs = build_configs(args.samples)
    missing_qlr, missing_bl = coverage(configs)
    total = len(configs)

    print(f"[agg] QLR caches      : {total - len(missing_qlr)}/{total} present")
    print(f"[agg] baseline caches : "
          f"{len(args.samples) * len(SEEDS) - len(missing_bl)}/"
          f"{len(args.samples) * len(SEEDS)} present")
    print(f"[agg] figure text path: "
          f"{'LaTeX (text.usetex)' if P.USETEX_ACTIVE else 'mathtext Computer Modern fallback'}")

    if missing_qlr or missing_bl:
        for s in missing_qlr[:20]:
            print(f"       missing QLR      {s}")
        if len(missing_qlr) > 20:
            print(f"       ... and {len(missing_qlr) - 20} more")
        for s in missing_bl:
            print(f"       missing baseline {s}")
        if not args.check and not args.force:
            print("[agg] refusing to aggregate with caches missing; "
                  "re-run the missing configs or pass --force for partial tables.")
            return 1

    if args.check:
        return 0 if not (missing_qlr or missing_bl) else 1

    X_train, X_test, y_train, y_test = HTRU2Preprocessor(
        test_size=Config.TEST_SIZE, random_state=Config.DATA_SEED
    ).preprocess(load_htru2(Config.DATA_PATH))

    for n in args.samples:
        print(f"[agg] === N={n} ===")
        run_sample_block(
            n_samples=n, configs=configs,
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            make_plots=True, plot_layer=None, write_tables=True,
        )

    print("[agg] done. tables -> tables/ , figures -> results/figs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
