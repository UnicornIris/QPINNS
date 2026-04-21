"""Trainable Embedding network for quantum angle encoding.

The TE maps classical PDE coordinates to per-qubit rotation angles via
a small MLP.  This is the core mechanism studied by Berger et al.
(TE-QPINN) and Tran et al. (x-TE-QPINN, arXiv:2602.09291).

Following Berger et al. Eq. 11, the final rotation angle is phi_i(x)*x_i,
where phi_i is the learned scaling factor and x_i is the raw coordinate
(cycled across qubits via modulus).  This provides a natural inductive
bias: when phi ≈ const, we recover standard angle encoding.

Our contribution: we additionally study how this embedding interacts
with shot noise and depolarizing noise, testing whether the TE can
learn to compensate for finite-shot and hardware errors.
"""

import torch
import torch.nn as nn
import math


class TrainableEmbedding(nn.Module):
    """Classical FNN that produces per-qubit angle scales.

    Architecture: in_dim → [width × hidden_layers with Tanh] → n_qubits
    Output phi_i(x) is multiplied element-wise by the cycled input
    coordinate x_{i mod d} to produce the final RY angle, following
    Berger et al. (2025) Eq. 11:  angle_i = phi_i(x) * x_{i mod d}.
    """

    def __init__(
        self,
        in_dim: int,
        n_qubits: int,
        hidden_layers: int = 2,
        width: int = 10,
        dtype=torch.float64,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_qubits = n_qubits
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev, width, dtype=dtype))
            layers.append(nn.Tanh())
            prev = width
        layers.append(nn.Linear(prev, n_qubits, dtype=dtype))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map coordinates [B, in_dim] → RY angles [B, n_qubits].

        Following Berger et al. Eq. 11:
            angle_i = phi_i(x) * x_{i mod d}
        where d = in_dim.  The phi values are unbounded (no tanh clamping)
        so the network can learn the optimal scaling freely.
        """
        phi = self.net(x)  # [B, n_qubits] — learned scaling factors
        # Cycle input coordinates across qubits: qubit i uses x_{i mod d}
        # e.g., 4 qubits, 2D input → (x1, x2, x1, x2)
        indices = [i % self.in_dim for i in range(self.n_qubits)]
        x_cycled = x[:, indices]  # [B, n_qubits]
        return phi * x_cycled  # [B, n_qubits] — the actual RY angles
