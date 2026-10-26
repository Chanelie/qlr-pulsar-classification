#!/usr/bin/env python3
"""
Train exactly ONE configuration and exit.

Deliberately does NOT aggregate, write tables, or plot -- with 8 of these
running concurrently those steps would race on tables/*.csv and
results/figs/**. A single aggregation pass runs after the grid completes
(scripts/aggregate_all.py).

Two job kinds:

  --kind qlr        one (N, encoding, L, seed) QLR fit -> caches/<slug>.pkl
                    Never touches the baseline cache.
  --kind baselines  one (N, seed) LogReg+CSVM+XGBoost+QSVM fit ->
                    caches/baselines_N<n>_seed<s>.pkl
                    Sole writer of that file.

Execution provenance (pinned core, wall-clock, hostname, concurrency at
launch, attempt number) is written into both the cache payload's `meta`
and results/run_records/<session>.json so that the exact runtime conditions are
recorded. Runtimes measured under different concurrency are not directly
comparable.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import pickle
import platform
import socket
import sys
import time
from pathlib import Path

# Single-threaded BLAS must be set before numpy/PennyLane import. The
# launcher exports these too; this is the belt-and-braces copy so that a
# hand-run job is single-threaded as well.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_k] = "1"

faulthandler.enable(all_threads=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from qlr.config import Config                                      # noqa: E402
from qlr.data import load_htru2, HTRU2Preprocessor                 # noqa: E402
from qlr.cache import ArtifactManager, config_slug                 # noqa: E402
from qlr.paths import baseline_cache_path                          # noqa: E402
from qlr.runner import train_single_config                         # noqa: E402
from qlr.baselines import train_baselines_for_seed                 # noqa: E402


def _pinned_core() -> int | None:
    """The core this process is actually pinned to, read from the OS."""
    try:
        aff = sorted(os.sched_getaffinity(0))
        return aff[0] if len(aff) == 1 else None
    except Exception:
        return None


def _provenance(session: str, attempt: int) -> dict:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        "session": session,
        "hostname": socket.gethostname(),
        "pinned_core": _pinned_core(),
        "affinity": affinity,
        "concurrency_at_launch": int(os.environ.get("QLR_CONCURRENCY", "0")) or None,
        "attempt": attempt,
        "retries": attempt - 1,
        "pid": os.getpid(),
        "python": platform.python_version(),
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }


def _write_record(session: str, rec: dict) -> Path:
    out = Config.RESULTS_DIR / "run_records"
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{session}.json"
    p.write_text(json.dumps(rec, indent=2, default=str))
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["qlr", "baselines"], required=True)
    ap.add_argument("--samples", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--layers", type=int)
    ap.add_argument("--encoding", type=str)
    ap.add_argument("--session", type=str, default="")
    ap.add_argument("--attempt", type=int,
                    default=int(os.environ.get("QLR_ATTEMPT", "1")))
    ap.add_argument("--force", action="store_true",
                    help="recompute even if the cache entry exists")
    args = ap.parse_args()

    if args.kind == "qlr" and (args.layers is None or args.encoding is None):
        ap.error("--kind qlr requires --layers and --encoding")

    session = args.session or (
        f"qlr_N{args.samples}_L{args.layers}_{args.encoding}_s{args.seed}"
        if args.kind == "qlr" else f"bl_N{args.samples}_s{args.seed}"
    )

    Config.setup_directories()
    prov = _provenance(session, args.attempt)
    print(f"[RUN_ONE] {session} core={prov['pinned_core']} "
          f"concurrency={prov['concurrency_at_launch']} attempt={args.attempt}",
          flush=True)

    # ---- data (identical preprocessing for every job) ----------------------
    data_path = Config.DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    X_train, X_test, y_train, y_test = HTRU2Preprocessor(
        test_size=Config.TEST_SIZE, random_state=Config.DATA_SEED
    ).preprocess(load_htru2(data_path))

    t0 = time.perf_counter()
    rec = {"kind": args.kind, "n_samples": args.samples, "seed": args.seed,
           "layers": args.layers, "encoding": args.encoding, **prov}

    if args.kind == "qlr":
        cfg = {
            "regime": f"N{args.samples}",
            "n_samples": args.samples,
            "encoding": args.encoding,
            "n_qubits": 3 if args.encoding == "amplitude" else 8,
            "n_layers": args.layers,
            "epochs": Config.EPOCHS,
            "seed": args.seed,
        }
        AM = ArtifactManager()
        slug = config_slug(cfg)
        rec["slug"] = slug
        rec["artifact"] = str(AM._path(cfg))

        if AM.exists(cfg) and not args.force:
            print(f"[RUN_ONE] cache hit, nothing to do: {slug}", flush=True)
            rec.update(status="cached", wall_clock_s=0.0)
            _write_record(session, rec)
            return 0

        # NOTE: no save_baselines_for_block() here -- QLR jobs must never
        # touch caches/baselines_*.pkl. The bl_* jobs own those files.
        y_te, probs, history, sub_idx, meta = train_single_config(
            cfg, X_train, y_train, X_test, y_test
        )
        wall = time.perf_counter() - t0
        meta = dict(meta)
        meta["execution"] = {**prov, "wall_clock_s": wall, "exit_status": 0}
        AM.save(cfg, y_te, probs, history, sub_idx, meta)
        rec.update(status="ok", wall_clock_s=wall,
                   train_time_s=meta.get("train_time_s"),
                   pred_time_s=meta.get("pred_time_s"))

    else:
        path = baseline_cache_path(args.samples, args.seed)
        rec["artifact"] = str(path)
        if path.exists() and not args.force:
            print(f"[RUN_ONE] cache hit, nothing to do: {path.name}", flush=True)
            rec.update(status="cached", wall_clock_s=0.0)
            _write_record(session, rec)
            return 0

        bl_probs, bl_times = train_baselines_for_seed(
            args.samples, args.seed, X_train, y_train, X_test, y_test
        )
        wall = time.perf_counter() - t0
        with open(path, "wb") as f:
            pickle.dump({
                "y_true": y_test,
                "probs": bl_probs,
                "times": bl_times,
                "execution": {**prov, "wall_clock_s": wall, "exit_status": 0},
            }, f)
        rec.update(status="ok", wall_clock_s=wall,
                   model_times={k: float(v) for k, v in bl_times.items()})

    rec["exit_status"] = 0
    p = _write_record(session, rec)
    print(f"[RUN_ONE] done {session} wall={rec['wall_clock_s']:.2f}s "
          f"record={p.name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
