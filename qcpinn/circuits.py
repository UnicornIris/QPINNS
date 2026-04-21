"""Variational quantum circuits for QCPINN.

Provides ansatz definitions and QNode factories for:
  - Standard angle embedding (baseline)
  - Trainable embedding (TE) with learned per-qubit angles
  - Noise-injected variants (shot noise, depolarizing noise)

Uses PennyLane parameter broadcasting for efficient batch evaluation
when running on default.qubit with backprop differentiation.
"""

import math
import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Ansatz library
# ---------------------------------------------------------------------------

def layered_circuit(params, n_qubits):
    """RZ-RX-CNOT-RX-RZ per layer.  Expects 4*n_qubits params."""
    idx = 0
    for q in range(n_qubits):
        qml.RZ(params[idx], wires=q); idx += 1
        qml.RX(params[idx], wires=q); idx += 1
    for q in range(n_qubits):
        qml.CNOT(wires=[q, (q + 1) % n_qubits])
    for q in range(n_qubits):
        qml.RX(params[idx], wires=q); idx += 1
        qml.RZ(params[idx], wires=q); idx += 1


def alternating_tdcnot(params, n_qubits):
    """Thinly-dressed CNOT ansatz. Expects 4*n_qubits params (for even n)."""
    idx = 0

    def _tdcnot(ctrl, tgt):
        nonlocal idx
        qml.RY(params[idx], wires=ctrl); idx += 1
        qml.RY(params[idx], wires=tgt); idx += 1
        qml.CNOT(wires=[ctrl, tgt])
        qml.RZ(params[idx], wires=ctrl); idx += 1
        qml.RZ(params[idx], wires=tgt); idx += 1

    # Even pairs
    for i in range(0, n_qubits - 1, 2):
        _tdcnot(i, i + 1)
    # Odd pairs
    for i in range(1, n_qubits, 2):
        _tdcnot(i, (i + 1) % n_qubits)


def hea_ansatz(params, n_qubits):
    """Hardware-efficient ansatz: RX-RY-RZ per qubit + CNOT chain.

    Expects 3*n_qubits params per layer.
    This is what x-TE-QPINN (arXiv:2602.09291) uses.
    """
    idx = 0
    for q in range(n_qubits):
        qml.RX(params[idx], wires=q); idx += 1
        qml.RY(params[idx], wires=q); idx += 1
        qml.RZ(params[idx], wires=q); idx += 1
    for q in range(n_qubits - 1):
        qml.CNOT(wires=[q, q + 1])


def sim_circ_15(params, n_qubits):
    """Circuit 15 from arXiv:1905.10876. Expects 2*n_qubits params."""
    idx = 0
    # First rotation block
    for q in range(n_qubits):
        qml.RY(params[idx], wires=q); idx += 1
    # Entangling block 1
    for q in reversed(range(n_qubits)):
        qml.CNOT(wires=[q, (q + 1) % n_qubits])
    # Second rotation block
    for q in range(n_qubits):
        qml.RY(params[idx], wires=q); idx += 1
    # Entangling block 2
    for q in range(n_qubits):
        ctrl = (q + n_qubits - 1) % n_qubits
        tgt = (ctrl + 3) % n_qubits
        qml.CNOT(wires=[ctrl, tgt])


# Registry: name -> (function, params_per_layer)
ANSATZ_REGISTRY = {
    "layered_circuit": (layered_circuit, lambda n: 4 * n),
    "alternating_tdcnot": (alternating_tdcnot, lambda n: 4 * n),
    "hea": (hea_ansatz, lambda n: 3 * n),
    "sim_circ_15": (sim_circ_15, lambda n: 2 * n),
}


# ---------------------------------------------------------------------------
# Quantum Layer
# ---------------------------------------------------------------------------

class QuantumLayer(nn.Module):
    """Variational quantum circuit layer for QCPINN.

    Supports three modes:
      1. "angle"  -- standard AngleEmbedding(x) -> ansatz
      2. "te"     -- TE-produced angles: RY(alpha_i) -> ansatz
      3. "amplitude" -- AmplitudeEmbedding(x) -> ansatz

    When noise_model is provided, inserts depolarizing channels after each gate layer.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.n_qubits = config["num_qubits"]
        self.n_layers = config["num_quantum_layers"]
        self.ansatz_name = config.get("q_ansatz", "hea")
        self.encoding = config.get("encoding", "angle")
        self.shots = config.get("shots", None)
        self.noise_strength = config.get("noise_strength", 0.0)  # noise probability
        self.noise_type = config.get("noise_type", "depolarizing")  # depolarizing, amplitude_damping, phase_damping

        # Device setup
        qml_device = config.get("qml_device", "default.qubit")
        device_kwargs = {}
        backend = config.get("qiskit_backend", None)
        if backend is not None:
            device_kwargs["backend"] = backend
            # IonQ's QIS flow is sensitive to aggressive re-synthesis.
            # Thread the transpiler level through to the PennyLane-Qiskit device.
            device_kwargs["optimization_level"] = config.get("transpile_optimization_level", 1)

        # Determine diff method
        diff_method = config.get("diff_method", "backprop")
        if (isinstance(qml_device, str) and "qiskit" in qml_device) or (self.shots and self.shots > 0):
            diff_method = "parameter-shift"
        if backend is not None:
            diff_method = "parameter-shift"

        # Noise: use default.mixed for depolarizing noise simulation
        if self.noise_strength > 0 and qml_device == "default.qubit":
            qml_device = "default.mixed"
            diff_method = "backprop"

        self._use_noisy = self.noise_strength > 0

        # Ansatz setup
        if self.ansatz_name not in ANSATZ_REGISTRY:
            raise ValueError(f"Unknown ansatz: {self.ansatz_name}. Available: {list(ANSATZ_REGISTRY.keys())}")
        self.ansatz_fn, params_per_layer_fn = ANSATZ_REGISTRY[self.ansatz_name]
        params_per_layer = params_per_layer_fn(self.n_qubits)

        # Trainable variational parameters -- uniform initialization in [-pi, pi]
        # following standard VQA practice for better expressivity at initialization
        self.params = nn.Parameter(
            (torch.rand(self.n_layers, params_per_layer, dtype=torch.float64) * 2 - 1) * math.pi
        )

        # Build PennyLane device and QNodes
        self.dev = qml.device(qml_device, wires=self.n_qubits, shots=self.shots, **device_kwargs)

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def _circuit_angle(x, params_flat):
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")
            self._apply_ansatz_layers(params_flat)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def _circuit_te(angles, params_flat):
            for i in range(self.n_qubits):
                qml.RY(angles[..., i], wires=i)
            self._apply_ansatz_layers(params_flat)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self._circuit_angle = _circuit_angle
        self._circuit_te = _circuit_te
        print("Quantum circuit:")
        print("qubits =", self.n_qubits)
        print("layers =", self.n_layers)
        print("ansatz =", self.ansatz_name)

    def _apply_ansatz_layers(self, params_flat):
        """Apply ansatz layers with optional noise injection."""
        ppl = params_flat.shape[0] // self.n_layers
        for l in range(self.n_layers):
            layer_params = params_flat[l * ppl : (l + 1) * ppl]
            self.ansatz_fn(layer_params, self.n_qubits)
            # Inject noise after each layer
            if self.noise_strength > 0:
                for q in range(self.n_qubits):
                    if self.noise_type == "amplitude_damping":
                        qml.AmplitudeDamping(self.noise_strength, wires=q)
                    elif self.noise_type == "phase_damping":
                        qml.PhaseDamping(self.noise_strength, wires=q)
                    else:  # depolarizing (default)
                        qml.DepolarizingChannel(self.noise_strength, wires=q)

    def _run_batched(self, circuit_fn, batch_input):
        """Run circuit on a batch, using broadcasting when possible.

        For noiseless default.qubit with backprop, PennyLane supports
        parameter broadcasting natively (much faster than per-sample loop).
        For noisy circuits (default.mixed), falls back to per-sample loop.
        """
        params_flat = self.params.reshape(-1)

        if not self._use_noisy and batch_input.shape[0] > 1:
            # Use native parameter broadcasting for speed
            try:
                y = circuit_fn(batch_input, params_flat)
                if isinstance(y, (list, tuple)):
                    y = torch.stack(y, dim=-1)
                if y.dim() == 1:
                    y = y.unsqueeze(0)
                return y
            except Exception:
                pass  # Fall back to per-sample loop

        # Per-sample loop (required for noisy circuits or when broadcasting fails)
        outs = []
        for i in range(batch_input.shape[0]):
            y = circuit_fn(batch_input[i], params_flat)
            if isinstance(y, (list, tuple)):
                y = torch.stack([e if isinstance(e, torch.Tensor) else torch.as_tensor(e) for e in y])
            outs.append(y)
        return torch.stack(outs)  # [B, n_qubits]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard angle-embedding forward.  x: [B, n_qubits]."""
        return self._run_batched(self._circuit_angle, x)

    def forward_te(self, angles: torch.Tensor) -> torch.Tensor:
        """TE-embedding forward.  angles: [B, n_qubits], pre-computed by TE network."""
        return self._run_batched(self._circuit_te, angles)
