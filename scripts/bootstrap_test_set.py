#!/usr/bin/env python3
"""
Paired, class-stratified bootstrap of the fixed test set (no training).

Inputs (read-only):
  caches/N{N}_enc-{enc}_q{q}_L{L}_E60_seed{s}.pkl      QLR test probabilities
  caches/baselines_N{N}_seed{s}.pkl                      QSVM test probabilities (reference)
  results/calibrated_baselines_train/preds_N{N}_seed{s}.pkl
        LogReg / CSVM / XGBoost x {weighted, unweighted} x {none, sigmoid, isotonic}
        (weighted/none is bit-identical to the grid baseline cache)

Procedure per (N, seed):
  * B = 2000 resamples of the 3580-point test set, with replacement WITHIN each
    class (328 positives, 3252 negatives are kept fixed per replicate).
  * One index matrix per seed (rng = default_rng(20260929 + seed)), shared by
    every model at that seed, so every QLR-vs-comparator difference is paired.
    Different seeds use independent index matrices, so pooled intervals
    reflect both the seed and the test-set sampling variability.
  * Metrics per replicate, computed with the project's code:
    PR-AUC (average_precision_score, as qlr.aggregate.metrics_from_probs),
    ROC-AUC (roc_auc_score), Recall@FPR1% / 5% (qlr.aggregate.metrics_at_fpr),
    Brier (brier_score_loss), ECE (qlr.aggregate.ece_from_probs, 15 equal-mass bins).
Replicates are stored in results/bootstrap_test_set/reps_N{N}_seed{s}.npz.

Outputs: tables/bootstrap_intervals.csv with rows
  kind  = value | delta   (delta = QLR − comparator; for Brier/ECE, negative = QLR better)
  scope = seed | pooled_<seeds>   (pooled = replicates concatenated across seeds;
          the point estimate is the mean of the per-seed full-test-set values)
  point, lo (2.5 %), hi (97.5 %), p_gt0 (share of replicates with Δ > 0)

Usage:
  compute: python scripts/bootstrap_test_set.py compute --samples 200 500 1000 --seeds 0 1 2 --jobs 8
           python scripts/bootstrap_test_set.py compute --samples 200 --seeds 3 4 5 6 7 8 9 \
                  --layers 3 --encodings angle --jobs 7
  build  : python scripts/bootstrap_test_set.py build
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import contextlib
import io
import pickle
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from qlr.aggregate import ece_from_probs, metrics_at_fpr   # noqa: E402

B = 2000
METRICS = ["PR_AUC", "ROC_AUC", "Recall_FPR1", "Recall_FPR5", "Brier", "ECE"]
REP_DIR = ROOT / "results" / "bootstrap_test_set"
ENCS = ["amplitude", "angle", "data_reuploading"]
LAYERS = [1, 2, 3, 5, 10]


def metric_vec(y, p) -> np.ndarray:
    with contextlib.redirect_stdout(io.StringIO()):   # metrics_at_fpr may print warnings
        r1 = metrics_at_fpr(y, p, 0.01)[0]
        r5 = metrics_at_fpr(y, p, 0.05)[0]
    return np.array([average_precision_score(y, p), roc_auc_score(y, p), r1, r5,
                     brier_score_loss(y, p), ece_from_probs(y, p, n_bins=15)])


def boot_indices(y, seed: int) -> np.ndarray:
    rng = np.random.default_rng(20260929 + seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.hstack([rng.choice(pos, size=(B, len(pos)), replace=True),
                      rng.choice(neg, size=(B, len(neg)), replace=True)])


def load_models(n, seed, layers, encs) -> tuple[np.ndarray, dict]:
    models, y_ref = {}, None
    for enc in encs:
        q = 3 if enc == "amplitude" else 8
        for L in layers:
            f = ROOT / "caches" / f"N{n}_enc-{enc}_q{q}_L{L}_E60_seed{seed}.pkl"
            if f.exists():
                d = pickle.load(open(f, "rb"))
                y_ref = d["y_test"] if y_ref is None else y_ref
                assert np.array_equal(d["y_test"], y_ref)
                models[f"QLR-{enc}-L{L}"] = np.asarray(list(d["probs"].values())[0], float)
    fb = ROOT / "results" / "calibrated_baselines_train" / f"preds_N{n}_seed{seed}.pkl"
    if fb.exists():
        d = pickle.load(open(fb, "rb"))
        y_ref = d["y_test"] if y_ref is None else y_ref
        assert np.array_equal(d["y_test"], y_ref)
        for (m, w, meth), p in d["probs"].items():
            models[f"{m}|{w}|{meth}"] = np.asarray(p, float)
    fq = ROOT / "caches" / f"baselines_N{n}_seed{seed}.pkl"
    if fq.exists() and fq.stat().st_size:
        d = pickle.load(open(fq, "rb"))
        assert np.array_equal(d["y_true"], y_ref)
        if "QSVM" in d["probs"]:
            models["QSVM|weighted|none"] = np.asarray(d["probs"]["QSVM"], float)
    return np.asarray(y_ref), models


def compute(task) -> str:
    n, seed, layers, encs = task
    y, models = load_models(n, seed, layers, encs)
    out = REP_DIR / f"reps_N{n}_seed{seed}.npz"
    old = dict(np.load(out, allow_pickle=False)) if out.exists() else {}
    idx = boot_indices(y, seed)
    res = {}
    for name, p in models.items():
        key_pt, key_rep = f"point::{name}", f"reps::{name}"
        if key_rep in old:
            continue
        res[key_pt] = metric_vec(y, p)
        res[key_rep] = np.vstack([metric_vec(y[ix], p[ix]) for ix in idx])
    REP_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **old, **res)
    return f"[bootstrap] N={n} seed={seed}: {len(res)//2} new series ({len(models)} models)"


def build() -> Path:
    data = {}
    for f in sorted(REP_DIR.glob("reps_N*_seed*.npz")):
        n, s = map(int, re.findall(r"N(\d+)_seed(\d+)", f.name)[0])
        z = np.load(f)
        for k in z.files:
            kind, name = k.split("::", 1)
            data.setdefault((n, s, name), {})[kind] = z[k]
    rows = []

    def add(kind, scope, n, L, enc, seed, model, pt, reps, n_seeds):
        lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
        gt = (reps > 0).mean(axis=0)
        for j, m in enumerate(METRICS):
            rows.append(dict(kind=kind, scope=scope, N=n, L=L, encoding=enc, seed=seed, model=model,
                             metric=m, point=pt[j], lo=lo[j], hi=hi[j],
                             p_gt0=gt[j] if kind == "delta" else np.nan, B=reps.shape[0], n_seeds=n_seeds))

    keys = sorted(data)
    qlr = [(n, s, nm) for n, s, nm in keys if nm.startswith("QLR-")]
    comps = sorted({nm for _, _, nm in keys if not nm.startswith("QLR-")})
    # per-seed values
    for n, s, nm in keys:
        d = data[(n, s, nm)]
        if nm.startswith("QLR-"):
            enc, L = re.match(r"QLR-(.+)-L(\d+)", nm).groups()
            add("value", "seed", n, int(L), enc, s, nm, d["point"], d["reps"], 1)
        else:
            add("value", "seed", n, "", "", s, nm, d["point"], d["reps"], 1)
    # per-seed deltas
    for n, s, nm in qlr:
        enc, L = re.match(r"QLR-(.+)-L(\d+)", nm).groups()
        for c in comps:
            if (n, s, c) in data:
                q, b = data[(n, s, nm)], data[(n, s, c)]
                add("delta", "seed", n, int(L), enc, s, c, q["point"] - b["point"], q["reps"] - b["reps"], 1)
    # pooled across seeds: 0-2 for every cell; also every available seed if > 3
    cells = sorted({(n, nm) for n, _, nm in qlr})
    for n, nm in cells:
        enc, L = re.match(r"QLR-(.+)-L(\d+)", nm).groups()
        have = sorted(s for (nn, s, m) in qlr if nn == n and m == nm)
        sets = [[0, 1, 2]] if set([0, 1, 2]) <= set(have) else []
        if len(have) > 3:
            sets.append(have)
        for ss in sets:
            tag = f"pooled_{min(ss)}-{max(ss)}"
            add("value", tag, n, int(L), enc, "", nm,
                np.mean([data[(n, s, nm)]["point"] for s in ss], 0),
                np.vstack([data[(n, s, nm)]["reps"] for s in ss]), len(ss))
            for c in comps:
                if all((n, s, c) in data for s in ss):
                    add("delta", tag, n, int(L), enc, "", c,
                        np.mean([data[(n, s, nm)]["point"] - data[(n, s, c)]["point"] for s in ss], 0),
                        np.vstack([data[(n, s, nm)]["reps"] - data[(n, s, c)]["reps"] for s in ss]), len(ss))
    # pooled comparator values (once per N and seed set)
    for n in sorted({k[0] for k in keys}):
        seeds_n = sorted({s for (nn, s, _) in keys if nn == n})
        sets = [[0, 1, 2]] if set([0, 1, 2]) <= set(seeds_n) else []
        if len(seeds_n) > 3:
            sets.append(seeds_n)
        for ss in sets:
            for c in comps:
                if all((n, s, c) in data for s in ss):
                    add("value", f"pooled_{min(ss)}-{max(ss)}", n, "", "", "", c,
                        np.mean([data[(n, s, c)]["point"] for s in ss], 0),
                        np.vstack([data[(n, s, c)]["reps"] for s in ss]), len(ss))
    df = pd.DataFrame(rows)
    path = ROOT / "tables" / "bootstrap_intervals.csv"
    df.to_csv(path, index=False)
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compute", "build"])
    ap.add_argument("--samples", type=int, nargs="+", default=[200, 500, 1000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--layers", type=int, nargs="+", default=LAYERS)
    ap.add_argument("--encodings", nargs="+", default=ENCS)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    if a.mode == "compute":
        tasks = [(n, s, a.layers, a.encodings) for n in a.samples for s in a.seeds]
        with Pool(a.jobs) as pool:
            for msg in pool.imap_unordered(compute, tasks):
                print(msg, flush=True)
    print(f"[bootstrap] wrote {build()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
