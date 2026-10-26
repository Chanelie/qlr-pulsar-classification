# qlr/quantum.py
# ------------------------------------------------------------
# Variational quantum feature maps for hybrid QLR.
# Encodings: angle, amplitude, data re-uploading.
# Gradients: parameter-shift; analytic expectations (shots=None).
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as qnp
import matplotlib.pyplot as plt

from .config import Config


def make_amplitude_vector(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """Pad to 2**n_qubits and L2-normalize. Raises if features exceed capacity."""
    dim = 1 << n_qubits
    x = np.asarray(x, dtype=np.float64).ravel()
    if len(x) > dim:
        raise ValueError(
            f"Amplitude encoding: cannot fit {len(x)} features into "
            f"{n_qubits} qubits (max capacity {dim})."
        )
    v = np.zeros(dim, dtype=np.float64)
    v[:len(x)] = x
    nrm = np.linalg.norm(v)
    return v / (nrm if nrm > 0 else 1.0)


class QuantumCircuitDesigner:
    """
    Variational circuit supporting angle, amplitude, and data re-uploading
    encodings.

    - Trainable layer: RY, RZ rotations on all wires + CZ ring entanglement.
      Data re-uploading interleaves re-encodings with rotation blocks and
      alternating (brick-wall) CZ pairs.
    - Gradients: parameter-shift; shots=None (analytic expectation values).
    - Initial parameters ``theta`` are drawn from a seed-scoped generator so
      initialization depends only on ``seed``, not on global RNG state or on
      which configurations ran earlier in the same process.
    """

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        encoding: str = "angle",
        device_name: str = "lightning.qubit",
        seed: int | None = None,
    ):
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.encoding = encoding.lower().strip()
        self.seed = seed

        self.dev = qml.device(device_name.strip(), wires=self.n_qubits, shots=None)

        rng = np.random.default_rng(seed)
        theta_init = rng.normal(0.0, 0.15, size=(self.n_layers, self.n_qubits, 2))
        self.theta = qnp.array(theta_init, requires_grad=True)

        self.qnode = None

    def _rotation_block(self, th_layer: qnp.ndarray) -> None:
        """Single-qubit RY, RZ rotations on all wires."""
        for q in range(self.n_qubits):
            qml.RY(th_layer[q, 0], wires=q)
            qml.RZ(th_layer[q, 1], wires=q)

    def _entangler_ring(self) -> None:
        """CZ ring entanglement over all wires."""
        for q in range(self.n_qubits):
            qml.CZ(wires=[q, (q + 1) % self.n_qubits])

    def _entangler_alternating(self, layer_idx: int) -> None:
        """Brick-wall CZ pairs, alternating offset by layer parity."""
        if layer_idx % 2 == 0:
            pairs = [(q, q + 1) for q in range(0, self.n_qubits - 1, 2)]
        else:
            pairs = [(q, q + 1) for q in range(1, self.n_qubits - 1, 2)]
            pairs.append((self.n_qubits - 1, 0))
        for q0, q1 in pairs:
            qml.CZ(wires=[q0, q1])

    def _layer(self, th_layer: qnp.ndarray) -> None:
        """One trainable layer: rotation block followed by CZ ring."""
        self._rotation_block(th_layer)
        self._entangler_ring()

    def _encode_angle_standard(self, x: qnp.ndarray, angle_scale: float) -> None:
        """RY angle embedding of the first n_qubits features, scaled by angle_scale."""
        x_arr = qnp.array(x, dtype=qnp.float64).ravel()[:self.n_qubits]
        qml.AngleEmbedding(x_arr * angle_scale, wires=range(self.n_qubits), rotation="Y")

    def create_qnode(self, angle_scale: float = Config.ANGLE_SCALE):
        """Build and return the QNode z(x; theta) -> R^{n_qubits} of Pauli-Z expvals."""

        @qml.qnode(self.dev, interface="autograd", diff_method="parameter-shift")
        def qnode(x, theta):

            if self.encoding == "data_reuploading":
                for L in range(self.n_layers - 1):
                    self._encode_angle_standard(x, angle_scale)
                    self._rotation_block(theta[L])
                    self._entangler_alternating(L)
                self._encode_angle_standard(x, angle_scale)
                self._rotation_block(theta[self.n_layers - 1])

            elif self.encoding == "angle":
                self._encode_angle_standard(x, angle_scale)
                for L in range(self.n_layers):
                    self._layer(theta[L])

            elif self.encoding == "amplitude":
                v = make_amplitude_vector(x, self.n_qubits)
                qml.AmplitudeEmbedding(v, wires=range(self.n_qubits), normalize=False)
                for L in range(self.n_layers):
                    self._layer(theta[L])

            else:
                raise ValueError(
                    "encoding must be one of: 'angle', 'amplitude', 'data_reuploading'"
                )

            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

        self.qnode = qnode
        return qnode

    def visualize_circuit(self, x_sample=None, out_path=None):
        """Draw the circuit for a sample input and save it to out_path."""
        if self.qnode is None:
            self.create_qnode()
        if x_sample is None:
            x_sample = np.zeros(self.n_qubits, dtype=float)
        else:
            x_sample = np.asarray(x_sample, dtype=float).ravel()
        theta_val = np.array(self.theta, dtype=float)
        if out_path is None:
            out_path = f"quantum_circuit_{self.encoding}_q{self.n_qubits}_L{self.n_layers}.png"
        fig, _ = qml.draw_mpl(self.qnode, style="sketch")(x_sample, theta_val)
        fig.tight_layout()
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Circuit diagram saved to {out_path}")
