#!/usr/bin/env python3
"""
Split-robustness wrapper. Trains one job and exits.

Identical to the grid (scripts/run_one.py) except for one thing: the
stratified 80/20 train/test split is drawn with random_state = seed instead
of Config.DATA_SEED (= 42). HTRU2Preprocessor fits the StandardScaler on the
training split only, exactly as in the grid. Everything else (subset drawing
via qlr.data.subset_indices_for, circuit/head init, shuffle, 60 epochs, Adam,
early stopping, baseline grids, class weighting, QSVM) is reused unchanged
from the qlr package.

  --kind qlr        QLR-angle (or --encoding) at N, L, seed
  --kind baselines  LogReg/CSVM/XGBoost weighted (+QSVM, as in the grid) AND
                    LogReg/CSVM/XGBoost unweighted (no QSVM, as in
                    scripts/baselines_weighting_comparison_N1000.py)

Outputs go ONLY to caches_split_robustness/ (never caches/).
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import faulthandler
import json
import pickle
import socket
import sys
import time
from pathlib import Path

faulthandler.enable(all_threads=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qlr.config import Config                                   # noqa: E402
from qlr.data import load_htru2, HTRU2Preprocessor, subset_indices_for  # noqa: E402
from qlr.cache import ArtifactManager, config_slug              # noqa: E402
from qlr.runner import train_single_config                      # noqa: E402
from qlr.baselines import train_baselines_for_seed              # noqa: E402

OUT = ROOT / "caches_split_robustness"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["qlr", "baselines"], required=True)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--encoding", type=str, default="angle")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rec_dir = OUT / "run_records"
    rec_dir.mkdir(exist_ok=True)

    # ---- the ONLY change vs the grid: split random_state = seed ------------
    X_train, X_test, y_train, y_test = HTRU2Preprocessor(
        test_size=Config.TEST_SIZE, random_state=args.seed
    ).preprocess(load_htru2(Config.DATA_PATH))

    aff = sorted(os.sched_getaffinity(0))
    prov = dict(hostname=socket.gethostname(), affinity=aff,
                pinned_core=aff[0] if len(aff) == 1 else None,
                concurrency_at_launch=int(os.environ.get("QLR_CONCURRENCY", "0")) or None,
                split_random_state=args.seed, test_size=Config.TEST_SIZE,
                thread_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    t0 = time.perf_counter()

    if args.kind == "qlr":
        cfg = {
            # distinct regime tag so slugs/failure markers never collide with the grid
            "regime": f"splitN{args.samples}",
            "n_samples": args.samples,
            "encoding": args.encoding,
            "n_qubits": 3 if args.encoding == "amplitude" else 8,
            "n_layers": args.layers,
            "epochs": Config.EPOCHS,
            "seed": args.seed,
        }
        AM = ArtifactManager(base_dir=OUT)
        session = f"qlr_N{args.samples}_L{args.layers}_{args.encoding}_s{args.seed}"
        if AM.exists(cfg) and not args.force:
            print(f"[SPLIT] cache hit {config_slug(cfg)}", flush=True)
            return 0
        y_te, probs, history, sub_idx, meta = train_single_config(cfg, X_train, y_train, X_test, y_test)
        meta = dict(meta)
        meta["execution"] = {**prov, "wall_clock_s": time.perf_counter() - t0}
        AM.save(cfg, y_te, probs, history, sub_idx, meta)
        rec = dict(kind="qlr", session=session, slug=config_slug(cfg), status="ok",
                   train_time_s=meta["train_time_s"], pred_time_s=meta["pred_time_s"],
                   epochs_run=len(history), **meta["execution"])
    else:
        session = f"bl_N{args.samples}_s{args.seed}"
        path = OUT / f"baselines_N{args.samples}_seed{args.seed}.pkl"
        if path.exists() and not args.force:
            print(f"[SPLIT] cache hit {path.name}", flush=True)
            return 0
        p_w, t_w = train_baselines_for_seed(args.samples, args.seed, X_train, y_train, X_test, y_test,
                                            class_balanced=True, include_qsvm=True)
        p_u, t_u = train_baselines_for_seed(args.samples, args.seed, X_train, y_train, X_test, y_test,
                                            class_balanced=False, include_qsvm=False)
        sub_idx = subset_indices_for(args.samples, args.seed, X_train, y_train)
        with open(path, "wb") as f:
            pickle.dump({"y_true": y_test, "probs_weighted": p_w, "times_weighted": t_w,
                         "probs_unweighted": p_u, "times_unweighted": t_u,
                         "subset_idx": sub_idx,
                         "execution": {**prov, "wall_clock_s": time.perf_counter() - t0}}, f)
        rec = dict(kind="baselines", session=session, status="ok",
                   times_weighted=t_w, times_unweighted=t_u,
                   wall_clock_s=time.perf_counter() - t0, **prov)

    (rec_dir / f"{session}.json").write_text(json.dumps(rec, indent=2, default=str))
    print(f"[SPLIT] done {session} wall={time.perf_counter() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
