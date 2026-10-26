# qlr/qlr_model.py
# ------------------------------------------------------------
# Hybrid QLR model:
# - quantum feature map (QNode) -> logistic head (w, b)
# - BCE + L2 regularization, parameter-shift gradients (via QNode)
# - minibatch training with early stopping
# ------------------------------------------------------------

from __future__ import annotations

import time
import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
from matplotlib import pyplot as plt


class HybridQLRClassifier:
    """
    Quantum Logistic Regression:
      z(x; theta) = QNode(x, theta) in R^{n_qubits}
      p(y=1|x) = sigmoid( w · z + b )

    Training:
      - Loss: weighted BCE + L2(theta, w)
      - Optimizer: Adam (qml.AdamOptimizer)
      - Gradients: parameter-shift (provided by the bound QNode)
      - Early stopping: patience on loss; non-finite guard
    """

    def __init__(
        self,
        qfeat,                      # QuantumCircuitDesigner (provides the QNode)
        learning_rate: float = 0.01,
        epochs: int = 60,
        batch_size: int | None = 16,
        tolerance: float = 1e-6,
        max_patience: int = 10,
        l2: float = 1e-4,
        class_weight: dict | None = None,
        verbose: bool = True,
        log_every: int = 10,
        seed: int | None = 42,
    ):
        self.qfeat = qfeat
        self.qnode = qfeat.create_qnode()
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tolerance
        self.max_patience = max_patience
        self.l2 = l2
        self.class_weight = class_weight
        self.verbose = verbose
        self.log_every = log_every
        self.seed = seed

        # Global seeding kept for the logistic-head initialization below;
        # the minibatch shuffle uses a seed-scoped generator in fit().
        if seed is not None:
            np.random.seed(seed)
            qnp.random.seed(seed)

        # Logistic head parameters
        self.w = qnp.random.normal(0.0, 0.2, size=(qfeat.n_qubits,), requires_grad=True)
        self.b = qnp.array(0.0, requires_grad=True)

        self.opt = qml.AdamOptimizer(stepsize=self.learning_rate)
        self.loss_history: list[float] = []
        self.is_fitted = False
        self.training_time = 0.0


    # Helper functions
    @staticmethod
    def _sigmoid(t):
        t = qnp.clip(t, -20.0, 20.0)  # numerical stability
        return 1.0 / (1.0 + qnp.exp(-t))

    def _weights(self, yb: qnp.ndarray) -> qnp.ndarray:
        """Return per-sample class weights (default 1)."""
        if self.class_weight is None:
            return qnp.ones_like(yb)
        w0 = self.class_weight.get(0, 1.0)
        w1 = self.class_weight.get(1, 1.0)
        return qnp.where(yb > 0.5, w1, w0)

    def _bce_with_l2(
        self,
        logits: qnp.ndarray,
        y: qnp.ndarray,
        theta: qnp.ndarray,
        w: qnp.ndarray,
    ) -> qnp.ndarray:
        """Weighted BCE(σ(logits), y) + L2 regularization on θ and w."""
        p = self._sigmoid(logits)
        bce = -qnp.mean(
            self._weights(y)
            * (y * qnp.log(p + 1e-7) + (1.0 - y) * qnp.log(1.0 - p + 1e-7))
        )
        reg = self.l2 * (qnp.sum(theta**2) + qnp.sum(w**2))
        return bce + reg
    


    # Training 
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train on (X, y) with minibatches and early stopping."""
        t0 = time.time()
        X = qnp.array(X)
        y = qnp.array(y)
        n = len(X)
        if self.batch_size is None:
            self.batch_size = min(16, n)  

        prev_loss = float("inf")
        best_loss = float("inf")
        best_params = None
        patience = 0

        shuffle_base = self.seed if self.seed is not None else 0

        for it in range(self.epochs):
            # deterministic, seed-scoped, epoch-varying shuffle
            idx = np.random.default_rng(shuffle_base + it).permutation(n)

            losses = []

            for s in range(0, n, self.batch_size):
                bidx = idx[s : s + self.batch_size]
                Xb, yb = X[bidx], y[bidx]

                def cost_fn(theta, w, b):
                    # Forward pass through QNode + logistic head per sample
                    logits = []
                    for x in Xb:
                        z = qnp.array(self.qnode(x, theta))
                        logits.append(qnp.dot(w, z) + b)
                    logits = qnp.array(logits)
                    return self._bce_with_l2(logits, yb, theta, w)

                (self.qfeat.theta, self.w, self.b), bl = self.opt.step_and_cost(
                    cost_fn, self.qfeat.theta, self.w, self.b
                )
                losses.append(float(bl))

            
            

            if losses:
                avg_loss = float(np.mean(losses))
                self.loss_history.append(avg_loss)

                # Guard against NaNs/Infs
                if not np.isfinite(avg_loss):
                    if self.verbose:
                        print(f"[stop] non-finite loss @ epoch {it}")
                    break

                # Early-stopping 
                improved = avg_loss < best_loss - 1e-12
                if improved:
                    best_loss = avg_loss
                    best_params = (self.qfeat.theta.copy(), self.w.copy(), self.b)
                    patience = 0
                else:
                    patience += 1

                if abs(prev_loss - avg_loss) < self.tolerance:
                    patience += 1

                # 1-based epoch index for nicer logs
                epoch_num = it + 1

                # Logging every log_every epochs (plus first and last)
                if self.verbose and (
                    epoch_num == 1
                    or epoch_num % self.log_every == 0
                    or epoch_num == self.epochs
                ):
                    print(
                        f"[Epoch {epoch_num:03d}] "
                        f"train_loss={avg_loss:.4f}  "
                        f"patience={patience}/{self.max_patience}"
                    )

                prev_loss = avg_loss

                # Early stopping trigger
                if patience >= self.max_patience:
                    if self.verbose:
                        print(f"[early-stop] no improvement for {self.max_patience} epochs")
                    break

        

        if best_params is not None:
            self.qfeat.theta, self.w, self.b = best_params

        self.training_time = time.time() - t0
        self.is_fitted = True
        return self
    
    

    def plot_training_history(self, out_path="training_history.png"):
        """Plot the training loss curve for this QLR instance."""
        if not self.loss_history:
            print("No training history to plot.")
            return

        history = np.array(self.loss_history, dtype=float)

        plt.figure(figsize=(12, 5))

        # Full history
        plt.subplot(1, 2, 1)
        plt.plot(history, linewidth=2)
        plt.title("Training Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.grid(True, alpha=0.3)
        plt.yscale("log")

        # Last 20% for convergence detail
        plt.subplot(1, 2, 2)
        start_idx = max(1, len(history) // 5)
        plt.plot(history[start_idx:], color="red", linewidth=2)
        plt.title("Loss Convergence (Last 80%)")
        plt.xlabel("Epoch")
        plt.ylabel("Binary Cross-Entropy Loss")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Training history saved to {out_path}")

        
        

    # Predictions
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return p(y=1|x) for each x in X."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        X = qnp.array(X)
        probs = []
        for x in X:
            z = qnp.array(self.qnode(x, self.qfeat.theta))
            p = float(self._sigmoid(qnp.dot(self.w, z) + self.b))
            probs.append(p)
        return np.array(probs, dtype=float)
    


    def predict(self, X: np.ndarray, thr: float = 0.5) -> np.ndarray:
        """Binary label predictions at threshold thr."""
        return (self.predict_proba(X) >= thr).astype(int)
