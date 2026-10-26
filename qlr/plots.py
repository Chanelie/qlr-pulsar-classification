# qlr/plots.py


from __future__ import annotations
from pathlib import Path
import pickle
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

import warnings
warnings.filterwarnings("ignore")

from .baselines import train_baselines_for_seed
from .cache import ArtifactManager
from .paths import baseline_cache_path


# ------------------------------------------------------------------
# Central manuscript figure style
# ------------------------------------------------------------------
# Figures are drawn at their FINAL size in the document and are not
# rescaled by LaTeX. Include them with \includegraphics without a
# width= argument (or with width equal to the size below); any
# \includegraphics[width=\columnwidth]{...} on a figure already
# authored at \columnwidth is a no-op, but scaling to a *different*
# width will re-introduce the mismatched-text-size problem this
# configuration exists to fix.
import os
import shutil
import subprocess

import matplotlib as mpl


def _usetex_available() -> bool:
    """
    True when matplotlib's text.usetex path can actually render.

    Overridable with QLR_USETEX=0/1. Beyond `latex` and `dvipng`,
    matplotlib's usetex preamble pulls in type1ec.sty, which TeX Live
    ships in the separate `cm-super` package -- a plain texlive install
    has type1cm.sty but not type1ec.sty, and usetex then fails at the
    first savefig rather than at import.
    """
    env = os.environ.get("QLR_USETEX")
    if env is not None:
        return env.strip().lower() not in ("", "0", "false", "no")

    if shutil.which("latex") is None or shutil.which("dvipng") is None:
        return False
    try:
        r = subprocess.run(
            ["kpsewhich", "type1ec.sty"],
            capture_output=True, timeout=15,
        )
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False


#: True when LaTeX text rendering is active; False when the mathtext
#: Computer Modern fallback is in use. Reported by run_one/aggregation
#: so the manuscript can state which path produced the figures.
USETEX_ACTIVE: bool = _usetex_available()

# Final figure widths in the thesis document (inches).  These values avoid
# authoring at a journal-wide 6.9 inches and then shrinking the text when the
# PDF is included at the narrower thesis text width.
WIDTH_1COL: float = 2.85
WIDTH_2COL: float = 5.9

# Figure typography must remain legible beside an 11 pt thesis body after the
# small amount of scaling performed by LaTeX.  Axis labels use the thesis body
# size; ticks and legends are only one point smaller.  Keeping these values in
# one place prevents multi-panel plots from silently reverting to tiny text.
AXIS_LABEL_FS = 11
TICK_FS = 10
TITLE_FS = 11
LEGEND_FS = 10
LINE_W = 1.4
MARKER_SZ = 5.0

_RC: dict = {
    "text.usetex":      USETEX_ACTIVE,
    "font.family":      "serif",
    "font.size":        11,
    "axes.labelsize":   11,
    "axes.titlesize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.titlesize": 11,
    "lines.linewidth":  1.4,
    "lines.markersize": 5.0,
    "axes.linewidth":   0.8,
    "savefig.dpi":      600,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
    "figure.constrained_layout.use": True,
}

if USETEX_ACTIVE:
    _RC["font.serif"] = ["Computer Modern Roman"]
else:
    # No LaTeX: render math with matplotlib's bundled Computer Modern and
    # set body text in cmr10 (also bundled), so the fallback still matches
    # the manuscript's CM body text. cmr10 has no U+2212, so negative tick
    # labels must use the ASCII hyphen.
    _RC["mathtext.fontset"] = "cm"
    _RC["mathtext.rm"] = "cmr10"
    _RC["font.serif"] = ["cmr10", "CMU Serif", "DejaVu Serif"]
    _RC["axes.unicode_minus"] = False

mpl.rcParams.update(_RC)


# ------------------------------------------------------------------
# Fixed model -> colour assignment
# ------------------------------------------------------------------
# A model must keep the same colour no matter which other models are
# drawn beside it. Leaving the colour to matplotlib assigns it from the
# default cycle in plotting order, per axes, so a panel that is missing
# one model shifts every later series one colour earlier -- and a legend
# captured from a different panel then labels those series wrongly
# rather than merely leaving one out.
#
# The values below reproduce the default-cycle order for a complete
# panel, so figures with every model present are unchanged.
MODEL_COLORS: Dict[str, str] = {
    "QLR-amplitude":        "tab:blue",
    "QLR-angle":            "tab:orange",
    "QLR-data_reuploading": "tab:green",
    "LogReg":               "tab:red",
    "CSVM":                 "tab:purple",
    "XGBoost":              "tab:brown",
    "QSVM":                 "tab:pink",
}

# Internal cache/model identifiers stay unchanged, while every published
# figure uses one consistent set of reader-facing names.
MODEL_DISPLAY_LABELS: Dict[str, str] = {
    "QLR-amplitude":        "QLR-amplitude",
    "QLR-angle":            "QLR-angle",
    "QLR-data_reuploading": "QLR-DR",
    "LogReg":               "Logistic regression",
    "CSVM":                 "SVM-RBF",
    "XGBoost":              "XGBoost",
    "QSVM":                 "QSVM",
    "Perfect calibration":  "Perfect calibration",
    "Perfectly Calibrated": "Perfect calibration",
}

WEIGHTED_DISPLAY_LABELS: Dict[str, str] = {
    **MODEL_DISPLAY_LABELS,
    "LogReg":  "Logistic regression (class-weighted)",
    "CSVM":    "SVM-RBF (class-weighted)",
    "XGBoost": "XGBoost (class-weighted)",
    "QSVM":    "QSVM (class-weighted)",
}

UNWEIGHTED_THESIS_DISPLAY_LABELS: Dict[str, str] = {
    **MODEL_DISPLAY_LABELS,
    "LogReg":  "Logistic regression (unweighted)",
    "CSVM":    "SVM-RBF (unweighted)",
    "XGBoost": "XGBoost (unweighted)",
    # The saved QSVM comparator was trained with balanced class weights.
    "QSVM":    "QSVM (class-weighted)",
}


def series_key(series_name: str) -> str:
    """
    Identity of a plotted series, independent of the panel it appears in.

    QLR series are named "QLR-<encoding>-L<layer>". The layer suffix
    changes from panel to panel, so it is stripped and the series keyed
    on the encoding alone. Everything else -- the baselines, and labels
    such as "Perfect calibration" -- is returned unchanged.
    """
    if series_name.startswith("QLR-"):
        return series_name.rsplit("-L", 1)[0]
    return series_name


def model_color(series_name: str) -> str:
    """
    Colour for one plotted series, stable across panels.

    An unrecognised name gets grey rather than silently borrowing some
    other model's colour.
    """
    return MODEL_COLORS.get(series_key(series_name), "tab:gray")


def display_label(
    series_name: str,
    labels: Dict[str, str] | None = None,
) -> str:
    """Reader-facing label for an internal model/series identifier."""
    key = series_key(series_name)
    mapping = MODEL_DISPLAY_LABELS if labels is None else labels
    return mapping.get(key, key)


def new_fig(*, width: float = WIDTH_1COL, height: float | None = None, **kwargs):
    """plt.subplots at a final-document width, with constrained_layout on."""
    if height is None:
        height = width * 0.62          # ~golden ratio
    kwargs.setdefault("constrained_layout", True)
    return plt.subplots(figsize=(width, height), **kwargs)


def save_figure(fig, out_dir, stem: str) -> None:
    """Write one figure as PDF (vector, for the manuscript) and PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png")
    plt.close(fig)


def _style_ax(ax, xlabel=None, ylabel=None, title=None):
    """Apply axis text. Sizes come from rcParams; no per-call overrides."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)


# Recall@FPR helpers
def recall_at_fpr(y_true, y_prob, fpr_target=0.01):
    """Return recall and threshold at the closest FPR ≤ fpr_target."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    mask = fpr <= fpr_target
    if not np.any(mask):
        return 0.0, 1.0
    i = np.argmax(tpr[mask])
    return float(tpr[mask][i]), float(thr[mask][i])


# Metric names (for ordering or LaTeX table columns)
METRICS = ["PR_AUC", "ROC_AUC", "F1", "Recall_FPR1"]
BRIER_METRIC = "Brier"
ECE_METRIC = "ECE" 

# Display-only terminology. Internal table/cache keys remain unchanged so a
# label-only figure refresh cannot alter or recompute any stored metric.
METRIC_DISPLAY_LABELS = {
    "ROC_AUC": "AUROC",
}


# Build mean probabilities across seeds
def collect_mean_probs_for_block_layer(
    block_cfgs: List[dict],
    AM: ArtifactManager,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray,  y_test: np.ndarray,
    layer: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    seeds = sorted({c["seed"] for c in block_cfgs})
    n_samples = next(iter({c["n_samples"] for c in block_cfgs}))
    encodings = sorted({c["encoding"] for c in block_cfgs if c["n_layers"] == layer})

    acc: Dict[str, List[np.ndarray]] = {
        **{f"QLR-{enc}-L{layer}": [] for enc in encodings},
        "LogReg": [],
        "CSVM": [],
        "XGBoost": [],
        "QSVM": [],
    }

    ref_y = None

    for sd in seeds:
        path = baseline_cache_path(n_samples, sd)

        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            bl_probs = data["probs"]
        else:
            bl_probs, _ = train_baselines_for_seed(
                n_samples, sd, X_train, y_train, X_test, y_test
            )

        for k in ["LogReg", "CSVM", "XGBoost", "QSVM"]:
            if k in bl_probs:
                acc[k].append(bl_probs[k])

        for enc in encodings:
            key_name = f"QLR-{enc}-L{layer}"
            match = [
                c for c in block_cfgs
                if c["seed"] == sd and c["n_layers"] == layer and c["encoding"] == enc
            ]
            if not match:
                continue

            payload = AM.load(match[0])
            if payload is None:
                continue

            y_t, probs_dict, *_ = payload
            if ref_y is None:
                ref_y = y_t

            qkey = next(k for k in probs_dict.keys() if k.startswith("QLR-"))
            acc[key_name].append(probs_dict[qkey])

    mean_probs: Dict[str, np.ndarray] = {}
    for k, lst in acc.items():
        if len(lst) == 0:
            continue
        mean_probs[k] = np.vstack(lst).mean(axis=0)

    if ref_y is None:
        raise RuntimeError("No cached QLR payload found for requested block/layer.")

    return ref_y, mean_probs


def plot_calibration_scores(
    labels, brier_vals, ece_vals, n_samples, layer, out_dir,
    *, brier_errs=None, ece_errs=None,
):
    """
    Plot calibration-focused metrics (lower is better):
      - Brier score
      - ECE
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preserve missing values as gaps. Replacing NaN with zero would make a
    # missing result look like perfect calibration.
    brier_vals = np.asarray(brier_vals, dtype=float)
    ece_vals   = np.asarray(ece_vals, dtype=float)
    brier_errs = None if brier_errs is None else np.asarray(brier_errs, dtype=float)
    ece_errs   = None if ece_errs is None else np.asarray(ece_errs, dtype=float)

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(WIDTH_2COL, 2.8), constrained_layout=True)
    ax.bar(x - width/2, brier_vals, width=width, yerr=brier_errs,
           capsize=2, label="Brier")
    ax.bar(x + width/2, ece_vals, width=width, yerr=ece_errs,
           capsize=2, label="ECE")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score (lower is better)")
    ax.set_title(f"Calibration metrics - N={n_samples}, L={layer}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    # sensible y-limit
    finite = np.concatenate([brier_vals[np.isfinite(brier_vals)],
                             ece_vals[np.isfinite(ece_vals)]])
    y_max = float(finite.max()) if finite.size else 1.0
    ax.set_ylim(0.0, y_max * 1.15 + 1e-12)

    save_figure(fig, out_dir, f"calibration_N{n_samples}_L{layer}")
    plt.close(fig)



def plot_bar_metrics(qlr_mean, qlr_std, bl_mean, bl_std, n_samples, layer, out_dir):
    """
    Bar chart of main metrics (PR_AUC, ROC_AUC, F1, Recall@1% FPR)
    for baselines and QLR, at fixed (n_samples, layer).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blm = bl_mean[bl_mean["n_samples"] == n_samples].copy()
    bls = bl_std[bl_std["n_samples"] == n_samples].copy()

    qlm = qlr_mean[
        (qlr_mean["n_samples"] == n_samples) &
        (qlr_mean["layers"] == layer)
    ].copy()
    qls = qlr_std[
        (qlr_std["n_samples"] == n_samples) &
        (qlr_std["layers"] == layer)
    ].copy()

    if blm.empty and qlm.empty:
        print(f"[plot_bar_metrics] No data for n={n_samples}, L={layer}")
        return

    labels = []
    rows_main = []

    for _, row in blm.iterrows():
        name = row["model"]
        labels.append("SVM-RBF" if name == "CSVM" else name)
        rows_main.append((name, row, bls.loc[row.name]))

    for _, row in qlm.iterrows():
        enc = row["encoding"]
        L = int(row["layers"])
        name = f"QLR-{enc}-L{L}"
        labels.append(name)
        rows_main.append((name, row, qls.loc[row.name]))

    metrics = METRICS
    x = np.arange(len(metrics))

    n_models = len(rows_main)
    total_width = 0.80
    width = total_width / max(1, n_models)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width

    fig, ax = plt.subplots(figsize=(WIDTH_2COL, 3.0), constrained_layout=True)

    for i, (name, mean_row, std_row) in enumerate(rows_main):
        vals = np.array([float(mean_row.get(m, np.nan)) for m in metrics], dtype=float)
        errs = np.array([float(std_row.get(m, np.nan)) for m in metrics], dtype=float)
        ax.bar(
            x + offsets[i],
            vals,
            width=width,
            yerr=errs,
            capsize=2,
            color=model_color(name),
            label=display_label(name, WEIGHTED_DISPLAY_LABELS),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [METRIC_DISPLAY_LABELS.get(metric, metric) for metric in metrics],
        rotation=0,
        ha="center",
    )
    ax.set_ylabel("Score")
    ax.set_title(f"Metrics (mean across seeds) - N={n_samples}, L={layer}")
    ax.grid(axis="y", alpha=0.3)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    save_figure(fig, out_dir, f"metrics_N{n_samples}_L{layer}")
    plt.close(fig)

        # Calibration plot: Brier + ECE
    brier_vals = []
    ece_vals = []
    brier_errs = []
    ece_errs = []
    for _, mean_row, std_row in rows_main:
        brier_vals.append(float(mean_row.get(BRIER_METRIC, np.nan)))
        ece_vals.append(float(mean_row.get(ECE_METRIC, np.nan)))
        brier_errs.append(float(std_row.get(BRIER_METRIC, np.nan)))
        ece_errs.append(float(std_row.get(ECE_METRIC, np.nan)))

    display_labels = [display_label(label, WEIGHTED_DISPLAY_LABELS) for label in labels]
    plot_calibration_scores(
        display_labels, brier_vals, ece_vals, n_samples, layer, out_dir,
        brier_errs=brier_errs, ece_errs=ece_errs,
    )




def plot_roc_pr(y_true, mean_probs, n_samples, layer, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_2COL, 2.7), constrained_layout=True)

    ax1.plot([0, 1], [0, 1], "k--", lw=1)
    for name, p in mean_probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax1.plot(
            fpr,
            tpr,
            color=model_color(name),
            label=(
                f"{display_label(name, WEIGHTED_DISPLAY_LABELS)} "
                f"AUROC={auc(fpr, tpr):.3f}"
            ),
        )
    ax1.set_title(f"ROC - N={n_samples}, L={layer}")
    ax1.set_xlabel("FPR")
    ax1.set_ylabel("TPR")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for name, p in mean_probs.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax2.plot(rec, prec, color=model_color(name),
                 label=display_label(name, WEIGHTED_DISPLAY_LABELS))
    ax2.set_title(f"Precision-Recall - N={n_samples}, L={layer}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    save_figure(fig, out_dir, f"roc_pr_N{n_samples}_L{layer}")
    plt.close(fig)





def plot_reliability(y_true, mean_probs, n_samples, layer, out_dir, n_bins=15):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 3.2), constrained_layout=True)

    # Draw the reference first so it cannot hide a model curve that lies on it.
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")

    # Store calibration curves for reuse in inset
    curves = {}
    for name, p in mean_probs.items():
        prob_true, prob_pred = calibration_curve(
            y_true, p, n_bins=n_bins, strategy="quantile"
        )
        curves[name] = (prob_pred, prob_true)
        ax.plot(prob_pred, prob_true, marker="o", color=model_color(name),
                label=display_label(name, WEIGHTED_DISPLAY_LABELS))
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_aspect("equal")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.99, 0.08),
        fontsize=LEGEND_FS,
        frameon=True,
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=1.6,
        handletextpad=0.5,
    )
    ax.grid(True, alpha=0.3)

    # Inset: zoomed to [0, 0.1] ---
    axins = inset_axes(ax, width="28%", height="28%", loc="upper left", borderpad=2.0)

    axins.plot([0, 1], [0, 1], "k--", lw=1)
    for name, (prob_pred, prob_true) in curves.items():
        axins.plot(prob_pred, prob_true, marker="o", markersize=3,
                   color=model_color(name))
    axins.set_xlim([0.0, 0.1])
    axins.set_ylim([0.0, 0.1])
    axins.set_aspect("equal")
    axins.grid(True, alpha=0.3)
    axins.tick_params(labelsize=8)
    axins.set_xticks([0.0, 0.05, 0.10])
    axins.set_yticks([0.0, 0.05, 0.10])
    axins.set_xlabel("")
    axins.set_ylabel("")

    save_figure(fig, out_dir, f"reliability_N{n_samples}_L{layer}")
    plt.close(fig)




def plot_reliability_combined(
    layers_data: dict,
    n_samples: int,
    out_dir,
    n_bins: int = 15,
    *,
    display_labels: Dict[str, str] | None = None,
    show_inset: bool = True,
    stem: str | None = None,
):
    """
    Create one full-width multi-panel reliability figure with a shared legend.
    layers_data maps layer L to (y_true, mean_probs). Model identifiers remain
    internal keys; ``display_labels`` controls only the published legend text.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = sorted(layers_data.keys())
    n_panels = len(layers)

    if n_panels == 0:
        return

    # Full-width figure for journal-style multi-panel display.
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(WIDTH_2COL, 3.65),
        constrained_layout=True,
        sharey=True,
    )

    if n_panels == 1:
        axes = [axes]

    handles, legend_labels = None, None

    for ax, L in zip(axes, layers):
        y_true, mean_probs = layers_data[L]
        curves = {}

        # Draw the reference first so a model point on the diagonal remains
        # visible. Older figures drew it last and visually hid some QLR-DR points.
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")

        for name, p in mean_probs.items():
            prob_true, prob_pred = calibration_curve(
                y_true, p, n_bins=n_bins, strategy="quantile"
            )
            curves[name] = (prob_pred, prob_true)
            ax.plot(prob_pred, prob_true, marker="o", markersize=MARKER_SZ, linewidth=LINE_W,
                    color=model_color(name), label=name)

        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax.set_aspect("equal")
        ax.set_title(f"$L={L}$", fontsize=TITLE_FS)
        ax.grid(True, alpha=0.3)

        if show_inset:
            # Inset for low-probability region. Tick labels are suppressed
            # to reduce clutter; the main axes define the quantities.
            axins = inset_axes(ax, width="30%", height="30%", loc="upper left", borderpad=2.0)
            axins.plot([0, 1], [0, 1], "k--", lw=0.9)
            for name, (prob_pred, prob_true) in curves.items():
                axins.plot(prob_pred, prob_true, marker="o", markersize=2.5, linewidth=1.0,
                           color=model_color(name))

            axins.set_xlim([0.0, 0.1])
            axins.set_ylim([0.0, 0.1])
            axins.set_aspect("equal")
            axins.grid(True, alpha=0.3)
            axins.tick_params(labelbottom=False, labelleft=False)
            axins.set_xlabel("")
            axins.set_ylabel("")

        if handles is None:
            handles, legend_labels = ax.get_legend_handles_labels()
            # Captured from one panel but serving all of them, so the
            # per-panel "-L<layer>" suffix would be misleading: drop it
            # and let the panel titles carry the depth.
            legend_labels = [display_label(lbl, display_labels) for lbl in legend_labels]
            # Keep the reference line last, after the model entries.
            if "Perfect calibration" in legend_labels:
                i = legend_labels.index("Perfect calibration")
                handles.append(handles.pop(i))
                legend_labels.append(legend_labels.pop(i))

    axes[0].set_ylabel("Fraction of positives", fontsize=AXIS_LABEL_FS)
    # One shared label stays readable at thesis width. Repeating this long
    # phrase below all three square panels makes neighbouring labels collide
    # when the figure is set at the final 5.9-inch text width.
    fig.supxlabel(
        "Mean predicted probability",
        fontsize=AXIS_LABEL_FS,
        # The legend occupies the outside-lower strip. Place the shared label
        # explicitly in the clear band between the square panels and legend.
        y=0.29,
    )

    # "outside lower center" (not "lower center" + a hand-tuned
    # bbox_to_anchor) so constrained_layout reserves the strip the legend
    # occupies and shrinks the axes to fit. A manual offset is measured
    # from the figure edge and takes no account of the legend's own
    # height, so it can overlap the shared x-label when entries wrap.
    fig.legend(
        handles,
        legend_labels,
        loc="outside lower center",
        ncol=3,
        fontsize=LEGEND_FS,
        frameon=True,
        borderpad=0.45,
        labelspacing=0.45,
        handlelength=1.8,
        handletextpad=0.6,
        columnspacing=1.4,
    )

    if stem is None:
        stem = f"reliability_combined_N{n_samples}"
    save_figure(fig, out_dir, stem)
    plt.close(fig)


def plot_confusions(y_true, mean_probs, n_samples, layer, out_dir, threshold=0.5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(mean_probs.keys())
    k = len(names)
    cols = 3
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols,
                             figsize=(WIDTH_2COL, (WIDTH_2COL / cols) * rows * 0.95),
                             constrained_layout=True)
    axes = np.array(axes).reshape(rows, cols)

    for i, name in enumerate(names):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        y_pred = (mean_probs[name] >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Non-Pulsar", "Pulsar"])
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(display_label(name, WEIGHTED_DISPLAY_LABELS), fontsize=TITLE_FS)

    for j in range(k, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"Confusion matrices - N={n_samples}, L={layer} (thr={threshold:.2f})")
    save_figure(fig, out_dir, f"confusions_N{n_samples}_L{layer}")
    plt.close(fig)


# Runtime plot (single kind)
def plot_runtime_scaling(
    qlr_mean: pd.DataFrame,
    bl_mean: pd.DataFrame,
    *,
    layer: int,
    out_dir: Path,
    kind: str = "Train_s",   # "Train_s" or "Pred_s"
):
    """
    Plot runtime scaling vs dataset size at fixed layer.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 2.3), constrained_layout=True)

    # QLR
    for enc in sorted(qlr_mean["encoding"].dropna().unique()):
        df = qlr_mean[
            (qlr_mean["layers"] == layer) &
            (qlr_mean["encoding"] == enc)
        ].sort_values("n_samples")

        if df.empty or kind not in df.columns:
            continue

        ax.plot(
            df["n_samples"],
            df[kind],
            marker="o",
            label=f"QLR-{enc}",
        )

    # Baselines (includes QSVM if present in bl_mean)
    for model in bl_mean["model"].unique():
        df = bl_mean[bl_mean["model"] == model].sort_values("n_samples")
        if df.empty or kind not in df.columns:
            continue

        label = "SVM-RBF" if model == "CSVM" else model
        ax.plot(
            df["n_samples"],
            df[kind],
            marker="o",
            linestyle="--",
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training samples (N)")
    ax.set_ylabel("Seconds")
    ax.set_title(f"{'Training' if kind == 'Train_s' else 'Prediction'} time scaling (L={layer})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    save_figure(fig, out_dir, f"runtime_{kind}_L{layer}")
    plt.close(fig)


# Runtime plot (both kinds)
def plot_runtime_scaling_both(
    qlr_mean: pd.DataFrame,
    bl_mean: pd.DataFrame,
    *,
    layer: int,
    out_dir: Path,
):
    """
    Convenience wrapper: generates TWO plots:
      - runtime_Train_s_L{layer}.(png/pdf)
      - runtime_Pred_s_L{layer}.(png/pdf)
    """
    plot_runtime_scaling(qlr_mean, bl_mean, layer=layer, out_dir=out_dir, kind="Train_s")
    plot_runtime_scaling(qlr_mean, bl_mean, layer=layer, out_dir=out_dir, kind="Pred_s")





def _unweighted_logreg_mean(table_path: Path, metric: str) -> float:
    """Return the three-seed unweighted, uncalibrated LogReg mean."""
    df = pd.read_csv(table_path)
    rows = df[
        (df["row_type"] == "mean")
        & (df["seed_set"].astype(str) == "0-2")
        & (df["model"] == "LogReg")
        & (df["weighting"] == "unweighted")
        & (df["method"] == "none")
    ]
    if len(rows) != 1:
        raise ValueError(
            f"expected one unweighted LogReg mean in {table_path}, found {len(rows)}"
        )
    return float(rows.iloc[0][metric])


def plot_ece_vs_depth(
    calibration_files,
    out_dir,
    *,
    unweighted_reference_files=None,
):
    """
    calibration_files: dict mapping N to CSV path
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = [1, 2, 3, 5, 10]
    n_values = [200, 500, 1000]

    # QLR variant identifiers as they appear in the Model column
    qlr_models = {
        "QLR_angle":            ("QLR-angle",          "o-",  "tab:orange"),
        "QLR_amplitude":        ("QLR-amplitude",       "s-",  "tab:blue"),
        "QLR_data_reuploading": ("QLR-DR",             "^-",  "tab:green"),
    }

    # Default references reproduce the grid figures. Thesis figures may pass
    # unweighted_reference_files to use the like-for-like LogReg comparison.
    ref_models = {
        "CSVM": ("SVM-RBF (class-weighted)", "--", "tab:purple"),
        "QSVM": ("QSVM (class-weighted)",    ":",  "tab:pink"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_2COL, 3.15), sharey=True,
                             constrained_layout=True)

    for ax, n in zip(axes, n_values):
        df = pd.read_csv(calibration_files[n])



        # Extract QLR ECE per layer 
        for model_key, (label, linestyle, color) in qlr_models.items():
            ece_per_layer = []
            for l in layers:
                # Model name format in CSV: e.g. "QLR_angle (L=1)"
                row = df[df["Model"] == f"{model_key} (L={l})"]
                if not row.empty:
                    # Summary tables store mean ± standard error.
                    val = row["ECE"].values[0]
                    mean = float(str(val).split("±")[0].strip())
                    std  = float(str(val).split("±")[1].strip())
                    ece_per_layer.append((mean, std))
                else:
                    # NaN, not None: errorbar rejects None in yerr and
                    # renders NaN as a gap. Same convention as plot_depth_scaling.
                    ece_per_layer.append((np.nan, np.nan))

            means = [v[0] for v in ece_per_layer]
            errors = [v[1] for v in ece_per_layer]

            ax.errorbar(
                layers, means, yerr=errors,
                fmt=linestyle, color=color,
                label=label, capsize=3, linewidth=LINE_W, markersize=MARKER_SZ
            )


        # Reference horizontal lines
        if unweighted_reference_files is not None:
            mean = _unweighted_logreg_mean(
                Path(unweighted_reference_files[n]), "ECE"
            )
            ax.axhline(
                mean, linestyle="--", color="gray", linewidth=1.4,
                label="Logistic regression (unweighted)",
            )
            reference_subset = {"QSVM": ref_models["QSVM"]}
        else:
            reference_subset = ref_models

        for model_key, (label, linestyle, color) in reference_subset.items():
            row = df[df["Model"] == model_key]
            if not row.empty:
                val = row["ECE"].values[0]
                mean = float(str(val).split("±")[0].strip())
                ax.axhline(
                    mean, linestyle=linestyle, color=color,
                    linewidth=1.4, label=label
                )

        ax.set_title(f"$N={n}$", fontsize=TITLE_FS)
        ax.set_xlabel("Number of layers $L$", fontsize=AXIS_LABEL_FS)
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(
        "ECE",
        fontsize=AXIS_LABEL_FS,
        # Preserve the original panel positions after shortening the rotated
        # label; this compensates only for the changed glyph bounds.
        labelpad=5.2,
    )

    # Single shared legend below the figure. "outside lower center" lets
    # constrained_layout reserve the strip; a manual bbox_to_anchor offset
    # ignores the legend's own height and starts overlapping the x-labels
    # as soon as the entries wrap to a second row.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="outside lower center",
        ncol=3,
    )

    save_figure(fig, out_dir, "ece_vs_depth_all_N")
    plt.close(fig)



def plot_runtime_vs_data(runtime_files, fixed_layer, out_dir):
    """
    Training time as a function of dataset size at a fixed circuit depth.
    runtime_files: dict mapping N to CSV path, e.g.
        {200: Path(...), 500: Path(...), 1000: Path(...)}
    fixed_layer: int, the fixed L value to plot
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qlr_models = {
        "QLR_angle":            ("QLR-angle",  "o-",  "tab:orange"),
        "QLR_amplitude":        ("QLR-amplitude", "s-", "tab:blue"),
        "QLR_data_reuploading": ("QLR-DR",     "^-",  "tab:green"),
    }
    ref_models = {
        "QSVM": ("QSVM", "--", "tab:pink"),
    }

    n_values = sorted(runtime_files.keys())
    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 2.3), constrained_layout=True)

    for model_key, (label, linestyle, color) in qlr_models.items():
        means, errors = [], []
        for n in n_values:
            df = pd.read_csv(runtime_files[n])
            row = df[df["Model"] == f"{model_key} (L={fixed_layer})"]
            if not row.empty:
                val = str(row["Train_s"].values[0])
                means.append(float(val.split("±")[0].strip()))
                errors.append(float(val.split("±")[1].strip()))
            else:
                # NaN, not None: errorbar rejects None in yerr and
                # renders NaN as a gap. Same convention as plot_depth_scaling.
                means.append(np.nan)
                errors.append(np.nan)

        ax.errorbar(
            n_values, means, yerr=errors,
            fmt=linestyle, color=color,
            label=label, capsize=3, linewidth=LINE_W, markersize=MARKER_SZ
        )

    # QSVM reference line (also varies with N)
    qsvm_means = []
    for n in n_values:
        df = pd.read_csv(runtime_files[n])
        row = df[df["Model"] == "QSVM"]
        if not row.empty:
            val = str(row["Train_s"].values[0])
            qsvm_means.append(float(val.split("±")[0].strip()))
        else:
            # NaN, not None: renders as a gap instead of an object-dtype array.
            qsvm_means.append(np.nan)

    ax.plot(
        n_values, qsvm_means,
        "--", color="tab:pink", linewidth=1.4, label="QSVM"
    )

    ax.set_xlabel("Training set size $N$", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Training time (s)", fontsize=AXIS_LABEL_FS)
    ax.set_yscale("log")
    ax.set_xticks(n_values)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, out_dir, f"runtime_vs_data_L{fixed_layer}")
    plt.close(fig)



def plot_runtime_vs_depth(runtime_files, fixed_n, out_dir):
    """
    Training time as a function of circuit depth at a fixed dataset size.
    runtime_files: dict mapping N to CSV path
    fixed_n: int, the fixed N value to plot
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers = [1, 2, 3, 5, 10]

    qlr_models = {
        "QLR_angle":            ("QLR-angle",    "o-",  "tab:orange"),
        "QLR_amplitude":        ("QLR-amplitude", "s-", "tab:blue"),
        "QLR_data_reuploading": ("QLR-DR",        "^-", "tab:green"),
    }

    df = pd.read_csv(runtime_files[fixed_n])
    fig, ax = plt.subplots(figsize=(WIDTH_1COL, 2.3), constrained_layout=True)

    for model_key, (label, linestyle, color) in qlr_models.items():
        means, errors = [], []
        for l in layers:
            row = df[df["Model"] == f"{model_key} (L={l})"]
            if not row.empty:
                val = str(row["Train_s"].values[0])
                means.append(float(val.split("±")[0].strip()))
                errors.append(float(val.split("±")[1].strip()))
            else:
                # NaN, not None: errorbar rejects None in yerr and
                # renders NaN as a gap. Same convention as plot_depth_scaling.
                means.append(np.nan)
                errors.append(np.nan)

        ax.errorbar(
            layers, means, yerr=errors,
            fmt=linestyle, color=color,
            label=label, capsize=3, linewidth=LINE_W, markersize=MARKER_SZ
        )

    ax.set_xlabel("Number of layers $L$", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Training time (s)", fontsize=AXIS_LABEL_FS)
    ax.set_yscale("log")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_figure(fig, out_dir, f"runtime_vs_depth_N{fixed_n}")
    plt.close(fig)




def compute_murphy_components(y_true, probs):
    """
    Compute Murphy decomposition components and calibration slope
    from predicted probabilities and true labels.
    Returns dict with keys: reliability, resolution, uncertainty, cal_slope.
    """
    # Murphy decomposition via binning
    n_bins = 15
    prob_true, prob_pred = calibration_curve(
        y_true, probs, n_bins=n_bins, strategy="quantile"
    )
    # Bin counts (approximate equal-mass)
    n = len(y_true)
    base_rate = y_true.mean()

    # Reconstruct bin sizes from quantile binning
    quantiles = np.percentile(probs, np.linspace(0, 100, n_bins + 1))
    bin_counts = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            mask = (probs >= quantiles[i]) & (probs < quantiles[i + 1])
        else:
            mask = (probs >= quantiles[i]) & (probs <= quantiles[i + 1])
        bin_counts.append(mask.sum())
    bin_counts = np.array(bin_counts[:len(prob_true)])

    weights = bin_counts / bin_counts.sum()

    reliability = np.sum(weights * (prob_pred - prob_true) ** 2)
    resolution  = np.sum(weights * (prob_true - base_rate) ** 2)
    uncertainty = base_rate * (1 - base_rate)

    # Calibration slope via linear regression on reliability diagram
    if len(prob_pred) >= 2:
        coeffs = np.polyfit(prob_pred, prob_true, 1)
        cal_slope = coeffs[0]
    else:
        cal_slope = np.nan

    return {
        "reliability": reliability,
        "resolution":  resolution,
        "uncertainty": uncertainty,
        "cal_slope":   cal_slope,
    }



def plot_murphy_depth(
    cache_dir,
    n_values,
    layers,
    encodings,
    qubits_map,
    epochs,
    n_seeds,
    out_dir,
):
    """
    Murphy decomposition and calibration slope plots for specified encodings.
    """
    

    cache_dir = Path(cache_dir)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    color_map = {
        "angle":            ("QLR-angle", "tab:orange"),
        "data_reuploading": ("QLR-DR",    "tab:green"),
    }

    # Aggregate Murphy components per encoding, N, L
    # results[encoding][N][L] = list of component dicts (one per seed)
    results = {enc: {n: {l: [] for l in layers} for n in n_values}
               for enc in encodings}

    for enc in encodings:
        q = qubits_map[enc]
        for n in n_values:
            for l in layers:
                for seed in range(n_seeds):
                    fname = (cache_dir /
                             f"N{n}_enc-{enc}_q{q}_L{l}_E{epochs}_seed{seed}.pkl")
                    if not fname.exists():
                        continue
                    with open(fname, "rb") as f:
                        data = pickle.load(f)

                    y_true = data["y_test"]
                    # Get the single prob array from the probs dict
                    probs  = list(data["probs"].values())[0]

                    components = compute_murphy_components(y_true, probs)
                    results[enc][n][l].append(components)



    # Compute mean and standard error across available seeds
    def agg(enc, n, l, key):
        vals = [r[key] for r in results[enc][n][l] if r]
        if not vals:
            return np.nan, np.nan
        vals = np.asarray(vals, dtype=float)
        if len(vals) < 2:
            return float(np.mean(vals)), 0.0
        std = np.std(vals, ddof=1)
        se = std / np.sqrt(len(vals))
        return float(np.mean(vals)), float(se)
    


    # Plot 1: Reliability term vs depth 
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_2COL, 2.9), sharey=True,
                             constrained_layout=True)
    for ax, n in zip(axes, n_values):
        for enc in encodings:
            label, color = color_map[enc]
            means = [agg(enc, n, l, "reliability")[0] for l in layers]
            ses = [agg(enc, n, l, "reliability")[1] for l in layers]
            ax.errorbar(layers, means, yerr=ses, fmt="o-", color=color,
                        label=label, capsize=3, linewidth=LINE_W, markersize=MARKER_SZ)
        ax.set_title(f"$N={n}$", fontsize=TITLE_FS)
        ax.set_xlabel("Circuit depth $L$", fontsize=AXIS_LABEL_FS)
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Murphy reliability", fontsize=AXIS_LABEL_FS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(encodings))
    save_figure(fig, out_dir, "murphy_reliability_depth")
    plt.close(fig)



    # Plot 2: Resolution term vs depth 
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_2COL, 2.9), sharey=True,
                             constrained_layout=True)
    for ax, n in zip(axes, n_values):
        for enc in encodings:
            label, color = color_map[enc]
            means = [agg(enc, n, l, "resolution")[0] for l in layers]
            ses = [agg(enc, n, l, "resolution")[1] for l in layers]
            ax.errorbar(layers, means, yerr=ses, fmt="o-", color=color,
                        label=label, capsize=3, linewidth=LINE_W, markersize=MARKER_SZ)
        ax.set_title(f"$N={n}$", fontsize=TITLE_FS)
        ax.set_xlabel("Circuit depth $L$", fontsize=AXIS_LABEL_FS)
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Murphy resolution", fontsize=AXIS_LABEL_FS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(encodings))
    save_figure(fig, out_dir, "murphy_resolution_depth")
    plt.close(fig)


    # Plot 3: Calibration slope vs depth 
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_2COL, 2.9), sharey=True,
                             constrained_layout=True)
    for ax, n in zip(axes, n_values):
        for enc in encodings:
            label, color = color_map[enc]
            means = [agg(enc, n, l, "cal_slope")[0] for l in layers]
            ses = [agg(enc, n, l, "cal_slope")[1] for l in layers]
            ax.errorbar(layers, means, yerr=ses, fmt="o-", color=color,
                        label=label, capsize=3, linewidth=LINE_W, markersize=MARKER_SZ)
        ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.0,
                   label="Perfect calibration")
        ax.set_title(f"$N={n}$", fontsize=TITLE_FS)
        ax.set_xlabel("Circuit depth $L$", fontsize=AXIS_LABEL_FS)
        ax.set_xticks(layers)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Calibration slope", fontsize=AXIS_LABEL_FS)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(encodings) + 1)
    save_figure(fig, out_dir, "calibration_slope_depth")
    plt.close(fig)




def plot_depth_scaling(
    n_values,
    layers,
    tables_dir,
    out_dir,
    *,
    unweighted_reference_files=None,
):
    
    tables_dir = Path(tables_dir)
    out_dir    = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qlr_encodings = {
        "angle":            ("QLR-angle",  "o-",  "tab:orange"),
        "amplitude":        ("QLR-amplitude", "s-", "tab:blue"),
        "data_reuploading": ("QLR-DR",     "^-",  "tab:green"),
    }

    metrics = {
        "PR_AUC":      "Precision-Recall AUC",
        "Recall_FPR1": r"Recall at $\mathrm{FPR}=1\%$",
    }

    for metric_col, ylabel in metrics.items():

        fig, axes = plt.subplots(1, 3, figsize=(WIDTH_2COL, 3.15), sharey=True,
                             constrained_layout=True)

        for ax, n in zip(axes, n_values):

            # Load QLR data across layers 
            for enc, (label, fmt, color) in qlr_encodings.items():
                means, errors = [], []
                for l in layers:
                    mean_path = tables_dir / f"qlr_mean_N{n}_L{l}.csv"
                    se_path   = tables_dir / f"qlr_se_N{n}_L{l}.csv"
                    if not mean_path.exists() or not se_path.exists():
                        means.append(np.nan)
                        errors.append(np.nan)
                        continue
                    df_m = pd.read_csv(mean_path)
                    df_s = pd.read_csv(se_path)
                    row_m = df_m[df_m["encoding"] == enc]
                    row_s = df_s[df_s["encoding"] == enc]
                    if row_m.empty:
                        means.append(np.nan)
                        errors.append(np.nan)
                    else:
                        means.append(float(row_m.iloc[0][metric_col]))
                        errors.append(float(row_s.iloc[0][metric_col])
                                      if not row_s.empty else 0.0)

                ax.errorbar(
                    layers, means, yerr=errors,
                    fmt=fmt, color=color,
                    label=label, capsize=3,
                    linewidth=LINE_W, markersize=MARKER_SZ
                )

            # Load baseline references
            bl_mean_path = tables_dir / f"baselines_mean_N{n}.csv"
            if bl_mean_path.exists():
                df_bl = pd.read_csv(bl_mean_path)

                if unweighted_reference_files is not None:
                    best_val = _unweighted_logreg_mean(
                        Path(unweighted_reference_files[n]), metric_col
                    )
                    ax.axhline(
                        best_val, linestyle="--", color="gray",
                        linewidth=1.4,
                        label="Logistic regression (unweighted)",
                    )
                else:
                    # Original grid reference: best weighted classical model.
                    classical = df_bl[df_bl["model"] != "QSVM"]
                    if not classical.empty:
                        best_idx = classical[metric_col].idxmax()
                        best_val = classical.loc[best_idx, metric_col]
                        best_model = classical.loc[best_idx, "model"]
                        ax.axhline(
                            best_val, linestyle="--", color="gray",
                            linewidth=1.4,
                            label=f"Best classical ({best_model}, class-weighted)"
                        )

                # QSVM reference
                qsvm_row = df_bl[df_bl["model"] == "QSVM"]
                if not qsvm_row.empty:
                    qsvm_val = float(qsvm_row.iloc[0][metric_col])
                    ax.axhline(
                        qsvm_val, linestyle=":", color="tab:pink",
                        linewidth=1.4, label="QSVM (class-weighted)"
                    )

            ax.set_title(f"$N={n}$", fontsize=TITLE_FS)
            ax.set_xlabel("Number of layers $L$", fontsize=AXIS_LABEL_FS)
            ax.set_xticks(layers)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel(ylabel, fontsize=AXIS_LABEL_FS)

        # Single shared legend below figure. "outside lower center" lets
        # constrained_layout reserve the strip; a manual bbox_to_anchor offset
        # ignores the legend's own height and starts overlapping the x-labels
        # as soon as the entries wrap to a second row.
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="outside lower center",
            ncol=3,
        )

        metric_tag = metric_col.replace("_", "")
        save_figure(fig, out_dir, f"depth_scaling_{metric_tag}")


