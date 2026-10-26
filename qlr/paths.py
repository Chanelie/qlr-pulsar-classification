# qlr/paths.py
from __future__ import annotations

from pathlib import Path
from .config import Config


def baseline_cache_path(n_samples: int | None, seed: int) -> Path:
    """Cache path for the baseline probabilities of one (n_samples, seed) pair."""
    ns_tag = "Full" if n_samples is None else f"N{n_samples}"
    return Path(Config.CACHE_DIR) / f"baselines_{ns_tag}_seed{seed}.pkl"