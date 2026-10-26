# qlr/cache.py
# ------------------------------------------------------------
# Per-config artifact caching & stable config slug
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
from .config import Config


def config_slug(cfg: dict) -> str:
    """
    Stable identifier for a single run configuration.
    Expected cfg keys: regime, encoding, n_qubits, n_layers, epochs, seed
    """
    return (
        f"{cfg.get('regime','run')}"
        f"_enc-{cfg['encoding']}"
        f"_q{cfg['n_qubits']}"
        f"_L{cfg['n_layers']}"
        f"_E{cfg['epochs']}"
        f"_seed{cfg['seed']}"
    )


class ArtifactManager:
    """
    Minimal pickle-based cache.
    Saves/loads payload:
        (y_test, probs_dict, history, subset_idx, meta)
    where:
        - y_test: np.ndarray
        - probs_dict: {model_name: np.ndarray of probs on X_test}
        - history: list[float] or None
        - subset_idx: np.ndarray or None
        - meta: dict (device, timings, etc.)
    """

    def __init__(self, base_dir: str | Path | None = None):
        self.base_dir = Path(base_dir or Config.CACHE_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, cfg: dict) -> Path:
        """Return the expected pickle path for this configuration."""
        return self.base_dir / f"{config_slug(cfg)}.pkl"

    def exists(self, cfg: dict) -> bool:
        """Check if cached artifact exists."""
        return self._path(cfg).exists()

    def save(
        self,
        cfg: dict,
        y_test: np.ndarray,
        probs_dict: dict[str, np.ndarray],
        history: list[float] | None,
        subset_idx: np.ndarray | None,
        meta: dict,
    ) -> None:
        """Serialize and write payload for this config."""
        payload = dict(
            y_test=np.asarray(y_test),
            probs=probs_dict,
            history=list(history) if history is not None else None,
            subset_idx=np.asarray(subset_idx) if subset_idx is not None else None,
            meta=meta,
        )
        path = self._path(cfg)
        try:
            with open(path, "wb") as f:
                pickle.dump(payload, f)
        except Exception as e:
            print(f"[cache] Failed to save {path.name}: {e}")

    def load(self, cfg: dict):
        """Read and deserialize cached payload for this config."""
        path = self._path(cfg)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
            return (
                payload.get("y_test"),
                payload.get("probs"),
                payload.get("history"),
                payload.get("subset_idx"),
                payload.get("meta"),
            )
        except Exception as e:
            print(f"[cache] Failed to load {path.name}: {e}")
            return None
