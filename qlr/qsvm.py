# qlr/qsvm.py
from __future__ import annotations

import time
import numpy as np
import pennylane as qml
from sklearn.svm import SVC


def make_projector_kernel(
    n_qubits: int = 8,
    angle_scale: float = np.pi / 8,
    n_repeats: int = 1,
    device_name: str = "lightning.qubit",
):
    """
    IQP fidelity kernel:
        k(x, x') = |<phi(x')|phi(x)>|^2
    """
    dev = qml.device(device_name, wires=n_qubits, shots=None)

    @qml.qnode(dev)
    def kernel_qnode(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        x1_scaled = np.asarray(x1, dtype=float).ravel()[:n_qubits] * angle_scale
        x2_scaled = np.asarray(x2, dtype=float).ravel()[:n_qubits] * angle_scale

        qml.IQPEmbedding(
            features=x1_scaled,
            wires=range(n_qubits),
            n_repeats=n_repeats,
        )
        qml.adjoint(qml.IQPEmbedding)(
            features=x2_scaled,
            wires=range(n_qubits),
            n_repeats=n_repeats,
        )

        return qml.probs(wires=range(n_qubits))

    def kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        return float(kernel_qnode(x1, x2)[0])

    return kernel


def _normalize_kernel(K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize K_ij by sqrt(K_ii K_jj)."""
    d = np.sqrt(np.clip(np.diag(K), eps, None))
    return K / np.outer(d, d)


def _enforce_psd(K: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Project K onto the PSD cone."""
    K = 0.5 * (K + K.T)
    evals, evecs = np.linalg.eigh(K)
    evals = np.clip(evals, eps, None)
    K = evecs @ np.diag(evals) @ evecs.T
    K = 0.5 * (K + K.T)
    return K


def validate_kernel(kernel_fn: callable, n_qubits: int = 8) -> None:
    """
    Small sanity check for the kernel.
    Run once before launching full experiments.
    """
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n_qubits)
    y = rng.standard_normal(n_qubits)

    self_sim = kernel_fn(x, x)
    cross = kernel_fn(x, y)

    assert abs(self_sim - 1.0) < 1e-4, (
        f"k(x,x) = {self_sim:.6f}, expected 1.0."
    )
    assert 0.0 <= cross <= 1.0 + 1e-6, (
        f"k(x,y) = {cross:.6f}, expected value in [0,1]."
    )

    X_small = rng.standard_normal((6, n_qubits))
    K = qml.kernels.square_kernel_matrix(
        X_small,
        kernel_fn,
        assume_normalized_kernel=True,
    )
    min_eval = float(np.linalg.eigvalsh(0.5 * (K + K.T)).min())

    print(
        f"[QSVM] validation passed | "
        f"k(x,x)={self_sim:.6f} | "
        f"k(x,y)={cross:.6f} | "
        f"min_eig={min_eval:.6e}"
    )


def kernel_diagnostics(
    X: np.ndarray,
    kernel_fn: callable,
    n_samples: int = 30,
    seed: int = 0,
) -> dict:
    """
    Quick diagnostic on a random subset.
    Useful before long runs.
    """
    X = np.asarray(X, dtype=float)
    N = len(X)

    if n_samples >= N:
        idx = np.arange(N)
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(N, size=n_samples, replace=False)

    X_sub = X[idx]

    K = qml.kernels.square_kernel_matrix(
        X_sub,
        kernel_fn,
        assume_normalized_kernel=True,
    )
    K = 0.5 * (K + K.T)

    diag = np.diag(K)
    off_diag = K[~np.eye(K.shape[0], dtype=bool)]
    evals = np.linalg.eigvalsh(K)

    return {
        "diag_mean": float(diag.mean()),
        "offdiag_mean": float(off_diag.mean()) if off_diag.size else np.nan,
        "offdiag_std": float(off_diag.std()) if off_diag.size else np.nan,
        "eig_min": float(evals.min()),
        "eig_max": float(evals.max()),
    }


def qsvm_predict_proba_timed(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    n_qubits: int = 8,
    C: float = 1.0,
    seed: int = 0,
    angle_scale: float = np.pi / 8,
    n_repeats: int = 1,
    device_name: str = "lightning.qubit",
    enforce_psd: bool = True,
    normalize_kernel: bool = True,
    run_validation: bool = False,
    return_diagnostics: bool = False,
):
    """
    Train a QSVM with an IQP kernel and return test-set probabilities.

    Returns
    -------
    probs   : np.ndarray of shape (len(X_test),)
    train_s : float
    pred_s  : float
    diagnostics : dict, optional
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train)

    if X_train.shape[1] != n_qubits:
        raise ValueError(
            f"X_train has {X_train.shape[1]} features but n_qubits={n_qubits}."
        )

    kernel_fn = make_projector_kernel(
        n_qubits=n_qubits,
        angle_scale=angle_scale,
        n_repeats=n_repeats,
        device_name=device_name,
    )

    if run_validation:
        validate_kernel(kernel_fn, n_qubits=n_qubits)

    t0 = time.perf_counter()

    K_train = qml.kernels.square_kernel_matrix(
        X_train,
        kernel_fn,
        assume_normalized_kernel=True,
    )
    K_train = 0.5 * (K_train + K_train.T)

    if normalize_kernel:
        K_train = _normalize_kernel(K_train)

    if enforce_psd:
        K_train = _enforce_psd(K_train)
        if normalize_kernel:
            K_train = _normalize_kernel(K_train)

    diagnostics = kernel_diagnostics(
        X_train,
        kernel_fn,
        n_samples=min(30, len(X_train)),
        seed=seed,
    )

    clf = SVC(
        kernel="precomputed",
        C=C,
        probability=True,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(K_train, y_train)
    train_s = time.perf_counter() - t0

    t1 = time.perf_counter()

    K_test = qml.kernels.kernel_matrix(X_test, X_train, kernel_fn)

    if normalize_kernel:
        train_self = np.array([kernel_fn(x, x) for x in X_train], dtype=float)
        test_self = np.array([kernel_fn(x, x) for x in X_test], dtype=float)

        train_diag = np.sqrt(np.clip(train_self, 1e-12, None))
        test_diag = np.sqrt(np.clip(test_self, 1e-12, None))

        K_test = K_test / np.outer(test_diag, train_diag)

    probs = clf.predict_proba(K_test)[:, 1].astype(float)
    pred_s = time.perf_counter() - t1

    if return_diagnostics:
        return probs, float(train_s), float(pred_s), diagnostics

    return probs, float(train_s), float(pred_s)