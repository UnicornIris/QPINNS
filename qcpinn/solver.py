"""Unified QCPINN solver.

A single hybrid quantum-classical model that supports:
  - "baseline":  classical preprocessor → angle embedding → quantum circuit → postprocessor
  - "te":        TE network → quantum circuit → postprocessor
  - "direct":    raw coordinates → angle embedding → quantum circuit → postprocessor
  - "repeat":    tiled coordinates → angle embedding → quantum circuit → postprocessor
  - "classical": classical-only MLP (for ablation baseline)

All modes share the same interface so the training loop is agnostic.
"""

import math
import os
import torch
import torch.nn as nn
import pennylane as qml
from typing import Optional
from torch.nn.utils.parametrizations import spectral_norm

from qcpinn.circuits import QuantumLayer
from qcpinn.embedding import TrainableEmbedding


class QCPINNSolver(nn.Module):
    """Unified QCPINN model."""

    def __init__(self, config: dict, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.config = config
        self.mode = config.get("mode", "baseline")  # "baseline", "te", "direct", "classical"
        self.n_qubits = config["num_qubits"]
        self.in_dim = config["input_dim"]
        self.out_dim = config.get("output_dim", 1)
        self.hidden_dim = config.get("hidden_dim", 50)
        self.dtype = torch.float64

        # Domain bounds for input rescaling (TE mode)
        dom_lo = config.get("domain_lo", [-1.0] * self.in_dim)
        dom_hi = config.get("domain_hi", [1.0] * self.in_dim)
        self.register_buffer("lo", torch.tensor(dom_lo, dtype=self.dtype))
        self.register_buffer("hi", torch.tensor(dom_hi, dtype=self.dtype))

        self._build_network()
        self._build_optimizer()
        self.loss_history = []
        self.loss_fn = nn.MSELoss()
        self.output_activation = self._build_output_activation()
        self.output_scale = float(self.config.get("output_scale", 1.0))

    def _build_network(self):
        """Construct the network components based on mode."""
        if self.mode == "classical":
            # Pure classical MLP for ablation comparison
            self.network = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim, dtype=self.dtype),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim, dtype=self.dtype),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.out_dim, dtype=self.dtype),
            ).to(self.device)
            return

        # Build quantum components
        self.quantum_layer = QuantumLayer(self.config)

        if self.mode == "te":
            # TE mode: trainable embedding maps raw coordinates → qubit angles
            # Default width=10, matching Berger et al. (2025)
            self.te = TrainableEmbedding(
                in_dim=self.in_dim,
                n_qubits=self.n_qubits,
                hidden_layers=self.config.get("te_hidden_layers", 2),
                width=self.config.get("te_width", 10),
                dtype=self.dtype,
            ).to(self.device)
            # No classical preprocessor — the TE IS the preprocessor
        elif self.mode in ("direct", "repeat"):
            # Direct: raw coordinates padded with zeros (only in_dim qubits encoded)
            # Repeat: raw coordinates tiled to fill all qubits — fair comparison to TE
            self.te = None
        else:
            # Baseline mode: classical preprocessor maps input → n_qubits features
            self.te = None
            self.preprocessor = nn.Sequential(
                nn.Linear(self.in_dim, self.hidden_dim, dtype=self.dtype),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.n_qubits, dtype=self.dtype),
            ).to(self.device)

        # Shared postprocessor: maps quantum output → PDE solution
        self.postprocessor = self._build_postprocessor().to(self.device)

        # Initialize classical weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _maybe_spectral_norm(self, layer: nn.Linear) -> nn.Module:
        if self.config.get("spectral_norm_postprocessor", False):
            return spectral_norm(layer)
        return layer

    def _build_postprocessor(self) -> nn.Module:
        """Construct the classical readout from quantum features."""
        post_type = self.config.get("postprocessor_type", "mlp").lower()
        if post_type == "linear":
            return self._maybe_spectral_norm(
                nn.Linear(self.n_qubits, self.out_dim, dtype=self.dtype)
            )
        if post_type == "mlp":
            return nn.Sequential(
                self._maybe_spectral_norm(
                    nn.Linear(self.n_qubits, self.hidden_dim, dtype=self.dtype)
                ),
                nn.Tanh(),
                self._maybe_spectral_norm(
                    nn.Linear(self.hidden_dim, self.out_dim, dtype=self.dtype)
                ),
            )
        raise ValueError(f"Unknown postprocessor_type: {post_type}")

    def _build_output_activation(self):
        """Optional bounded output activation for hardware-robust runs."""
        act_name = self.config.get("output_activation", "identity").lower()
        if act_name in ("identity", "none"):
            return nn.Identity()
        if act_name == "tanh":
            return nn.Tanh()
        raise ValueError(f"Unknown output_activation: {act_name}")

    def _apply_output_constraints(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Apply optional physics-aware constraints to the final solution output."""
        u = self.output_scale * self.output_activation(u)

        if self.config.get("hard_bc", False):
            problem = self.config.get("problem")
            if problem != "helmholtz":
                raise ValueError("hard_bc is currently only supported for the Helmholtz problem")

            envelope = torch.ones((x.shape[0], 1), dtype=self.dtype, device=self.device)
            bc_scale = float(self.config.get("hard_bc_scale", 10.0))
            span = torch.clamp(self.hi - self.lo, min=1e-8)
            for dim in range(self.in_dim):
                coord = x[:, dim : dim + 1]
                left_dist = torch.clamp((coord - self.lo[dim]) / span[dim], min=0.0)
                right_dist = torch.clamp((self.hi[dim] - coord) / span[dim], min=0.0)
                dist = torch.minimum(left_dist, right_dist)
                # Softly saturating boundary envelope:
                # exact zero on the boundary, near-identity in the interior.
                factor = 1.0 - torch.exp(-bc_scale * dist)
                envelope = envelope * factor
            u = envelope * u

        return u

    def _build_optimizer(self):
        """Build optimizer from all trainable parameters (no double-counting).

        Default is L-BFGS with strong Wolfe line search, matching
        Berger et al. (2025) and Tran et al. (2026).
        """
        params = list(filter(lambda p: p.requires_grad, self.parameters()))
        opt_name = self.config.get("optimizer", "lbfgs").lower()
        lr = self.config.get("lr", 1e-3)

        if opt_name == "lbfgs":
            self.optimizer = torch.optim.LBFGS(
                params, lr=lr, max_iter=20, line_search_fn="strong_wolfe"
            )
        else:
            self.optimizer = torch.optim.Adam(params, lr=lr)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=500, min_lr=1e-6
        )

    def _rescale(self, x: torch.Tensor) -> torch.Tensor:
        """Affine rescale x from [lo, hi] to [-0.95, 0.95] per dimension."""
        a = torch.tensor(-0.95, dtype=self.dtype, device=self.device)
        b = torch.tensor(0.95, dtype=self.dtype, device=self.device)
        span = torch.clamp(self.hi - self.lo, min=1e-8)
        return a + (x - self.lo) * ((b - a) / span)

    def extract_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the quantum-layer outputs before the classical readout."""
        if x.dim() != 2:
            raise ValueError(f"Expected [B, in_dim] input, got shape {x.shape}")
        x = x.to(dtype=self.dtype, device=self.device)

        if self.mode == "classical":
            raise ValueError("Classical mode does not expose quantum features")

        x_scaled = self._rescale(x)

        if self.mode == "te":
            angles = self.te(x_scaled)
            q_out = self.quantum_layer.forward_te(angles)
        elif self.mode == "repeat":
            reps = math.ceil(self.n_qubits / self.in_dim)
            x_tiled = x_scaled.repeat(1, reps)[:, :self.n_qubits]
            q_out = self.quantum_layer(x_tiled)
        elif self.mode == "direct":
            q_out = self.quantum_layer(x_scaled)
        else:
            features = self.preprocessor(x_scaled)
            q_out = self.quantum_layer(features)

        q_out = q_out.to(dtype=self.dtype, device=self.device)

        sigma = float(self.config.get("noise_augmentation_sigma", 0.0))
        if self.training and sigma > 0:
            q_out = torch.clamp(q_out + sigma * torch.randn_like(q_out), -1.0, 1.0)

        return q_out

    def readout_quantum_features(self, x: torch.Tensor, q_out: torch.Tensor) -> torch.Tensor:
        """Map quantum features to the final constrained PDE output."""
        q_out = q_out.to(dtype=self.dtype, device=self.device)
        return self._apply_output_constraints(x.to(dtype=self.dtype, device=self.device), self.postprocessor(q_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        All quantum modes rescale input to [-0.95, 0.95] following
        Berger et al. (2025) Eq. 9, ensuring consistent coordinate
        scaling across modes.

        Args:
            x: [B, in_dim] input coordinates (float64)
        Returns:
            u: [B, out_dim] PDE solution prediction
        """
        if x.dim() != 2:
            raise ValueError(f"Expected [B, in_dim] input, got shape {x.shape}")
        x = x.to(dtype=self.dtype, device=self.device)

        if self.mode == "classical":
            return self._apply_output_constraints(x, self.network(x))
        q_out = self.extract_quantum_features(x)
        return self.readout_quantum_features(x, q_out)

    def count_parameters(self):
        """Return a dict of parameter counts by component."""
        counts = {}
        total = 0
        for name, param in self.named_parameters():
            component = name.split(".")[0]
            if component not in counts:
                counts[component] = 0
            counts[component] += param.numel()
            total += param.numel()
        counts["total"] = total
        return counts

    def save_state(self, path: str):
        """Save complete model state."""
        state = {
            "config": self.config,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss_history": self.loss_history,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(state, path)

    @classmethod
    def load_state(cls, path: str, device=None):
        """Load model from checkpoint."""
        if device is None:
            device = torch.device("cpu")
        state = torch.load(path, map_location=device)
        model = cls(state["config"], device=device)
        model.load_state_dict(state["model_state_dict"])
        model.optimizer.load_state_dict(state["optimizer_state_dict"])
        model.scheduler.load_state_dict(state["scheduler_state_dict"])
        model.loss_history = state["loss_history"]
        return model
