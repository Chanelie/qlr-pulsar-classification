"""
Shared helpers for the ten-seed and split-robustness analyses. Metrics use the
repository implementations in qlr.aggregate and qlr.plots; the bin-consistent
Murphy terms come from posthoc_calibration_train.murphy_aligned.
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from qlr.aggregate import metrics_from_probs, metrics_at_fpr      # noqa: E402
from qlr.plots import compute_murphy_components                   # noqa: E402
from posthoc_calibration_train import murphy_aligned              # noqa: E402

METRICS = ["PR_AUC", "ROC_AUC", "Recall_FPR1", "Recall_FPR5", "Brier", "ECE",
           "F1", "Precision", "Recall", "Precision_FPR1", "FNR_FPR1",
           "Murphy_reliability", "Murphy_resolution",
           "Murphy_reliability_aligned", "Murphy_resolution_aligned", "Murphy_uncertainty"]
KEY_METRICS = ["PR_AUC", "ROC_AUC", "Recall_FPR1", "Recall_FPR5", "Brier", "ECE"]
LOWER_IS_BETTER = {"Brier", "ECE", "FNR_FPR1", "Murphy_reliability", "Murphy_reliability_aligned"}


def all_metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p, float)
    m = metrics_from_probs(y, p)
    with contextlib.redirect_stdout(io.StringIO()):
        r1, p1, f1, _ = metrics_at_fpr(y, p, 0.01)
        r5, _, _, _ = metrics_at_fpr(y, p, 0.05)
    m.update(Recall_FPR1=r1, Precision_FPR1=p1, FNR_FPR1=f1, Recall_FPR5=r5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = compute_murphy_components(y, p)
    ra, sa = murphy_aligned(y, p)
    m.update(Murphy_reliability=mu["reliability"], Murphy_resolution=mu["resolution"],
             Murphy_uncertainty=mu["uncertainty"],
             Murphy_reliability_aligned=ra, Murphy_resolution_aligned=sa)
    return {k: float(m[k]) for k in METRICS}


def exact_wilcoxon(d: np.ndarray) -> tuple[float, int]:
    """Exact two-sided Wilcoxon signed-rank p-value (zeros dropped, 'wilcox').
    Returns (p, n_nonzero); p = 1.0 if every difference is zero."""
    d = np.asarray(d, float)
    nz = int(np.sum(d != 0))
    if nz == 0:
        return 1.0, 0
    return float(wilcoxon(d, method="exact", zero_method="wilcox").pvalue), nz


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjustment of a family of p-values."""
    order = np.argsort(p.values)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p.values[i])
        adj[i] = min(1.0, run)
    return pd.Series(adj, index=p.index)


def analyse(per_seed: pd.DataFrame, ref: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """per_seed: columns [model, seed] + METRICS.
    Returns (summary mean±SE per model, per-seed deltas, delta summary with Wilcoxon)."""
    g = per_seed.groupby("model", sort=False)
    n = g.size()
    summ = []
    for mdl, sub in g:
        for k in METRICS:
            v = sub[k].values
            summ.append(dict(model=mdl, metric=k, n_seeds=len(v), mean=v.mean(),
                             se=v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan))
    summ = pd.DataFrame(summ)
    q = per_seed[per_seed.model == ref].set_index("seed")
    drows, dsum = [], []
    for mdl, sub in g:
        if mdl == ref:
            continue
        sub = sub.set_index("seed")
        seeds = sorted(set(q.index) & set(sub.index))
        for k in METRICS:
            d = q.loc[seeds, k].values - sub.loc[seeds, k].values
            for s, v in zip(seeds, d):
                drows.append(dict(comparator=mdl, metric=k, seed=s, delta=v))
            p, nz = exact_wilcoxon(d)
            dsum.append(dict(comparator=mdl, metric=k, n_pairs=len(d), n_nonzero=nz,
                             mean_delta=d.mean(), se_delta=d.std(ddof=1) / np.sqrt(len(d)),
                             n_qlr_better=int(np.sum(d < 0) if k in LOWER_IS_BETTER else np.sum(d > 0)),
                             wilcoxon_p_exact=p))
    dsum = pd.DataFrame(dsum)
    dsum["wilcoxon_p_holm_within_metric"] = dsum.groupby("metric")["wilcoxon_p_exact"].transform(holm)
    return summ, pd.DataFrame(drows), dsum


def md_tables(summ, dsum, ref, order, title_note="") -> str:
    """Readable markdown: value table and Δ table for the key metrics."""
    out = []
    piv = summ[summ.metric.isin(KEY_METRICS)].copy()
    piv["cell"] = piv.apply(lambda r: f"{r['mean']:.4f} ± {r['se']:.4f}", axis=1)
    t = piv.pivot(index="model", columns="metric", values="cell").reindex(order)[KEY_METRICS]
    out.append("| Model | " + " | ".join(KEY_METRICS) + " |")
    out.append("|---|" + "---|" * len(KEY_METRICS))
    for mdl, r in t.iterrows():
        out.append(f"| {mdl} | " + " | ".join(r.fillna("—")) + " |")
    out.append("")
    out.append(f"Paired Δ = {ref} − comparator, mean ± SE over seeds; exact two-sided Wilcoxon p "
               f"(Holm-adjusted within metric across comparators in brackets); k/n = seeds where {ref} is better. "
               f"For Brier and ECE a negative Δ means {ref} is better. {title_note}")
    out.append("")
    ds = dsum[dsum.metric.isin(KEY_METRICS)].copy()
    ds["cell"] = ds.apply(lambda r: f"{r.mean_delta:+.4f} ± {r.se_delta:.4f}; p={r.wilcoxon_p_exact:.3g} "
                                    f"[{r.wilcoxon_p_holm_within_metric:.3g}]; {r.n_qlr_better}/{r.n_pairs}", axis=1)
    t2 = ds.pivot(index="comparator", columns="metric", values="cell").reindex([o for o in order if o != ref])[KEY_METRICS]
    out.append("| Comparator | " + " | ".join(KEY_METRICS) + " |")
    out.append("|---|" + "---|" * len(KEY_METRICS))
    for mdl, r in t2.iterrows():
        out.append(f"| {mdl} | " + " | ".join(r.fillna("—")) + " |")
    return "\n".join(out)
