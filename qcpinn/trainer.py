"""Training loop for QCPINN experiments.

Supports both Adam and L-BFGS optimization.
Logs metrics, saves checkpoints, and handles all PDE types uniformly.
"""

import time
import os
import json
import torch
import logging
from typing import Dict, List, Optional

from qcpinn.pde import OPERATOR_REGISTRY as PDE_OPERATORS
from qcpinn.datasets import (
    DATASET_REGISTRY,
    helmholtz_dataset,
    helmholtz_exact_u,
    wave_exact_u,
    kg_exact_u,
    heat_1d_exact_u,
    get_helmholtz_params,
)
from qcpinn.pde import helmholtz_operator

# Exact solution registry for validation (function, domain_lo, domain_hi)
_EXACT_U_REGISTRY = {
    "helmholtz": (helmholtz_exact_u, [-1., -1.], [1., 1.]),
    "wave": (wave_exact_u, [0., 0.], [1., 1.]),
    "klein_gordon": (kg_exact_u, [0., 0.], [1., 1.]),
    "heat_1d": (heat_1d_exact_u, [0., -1.], [1., 1.]),
}


class Trainer:
    """Unified training loop for QCPINN experiments.

    Supports fixed collocation grids (matching Berger/Tran papers) or
    stochastic resampling per epoch.  Fixed grids are default for L-BFGS
    and strongly recommended for smooth loss curves.
    """

    def __init__(self, model, config: dict, log_dir: str, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device("cpu")
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.pde_name = config["problem"]
        self.batch_size = config.get("batch_size", 64)
        self.epochs = config.get("epochs", 5000)
        self.print_every = config.get("print_every", 100)
        self.bc_weight = config.get("bc_weight", 10.0)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.postprocessor_gain_penalty = config.get("postprocessor_gain_penalty", 0.0)
        # Whether to use fixed collocation points (default: True for deterministic training)
        self.fixed_collocation = config.get("fixed_collocation", True)

        # Setup logging
        self.logger = logging.getLogger(f"qcpinn.{log_dir}")
        self.logger.setLevel(logging.INFO)
        # Clear any old handlers to prevent accumulation
        self.logger.handlers.clear()
        fh = logging.FileHandler(os.path.join(log_dir, "training.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(ch)

        # Build datasets
        ds_fn, in_dim, out_dim, dom_lo, dom_hi = DATASET_REGISTRY[self.pde_name]
        if self.pde_name == "helmholtz":
            a1, a2, lam = get_helmholtz_params(self.config)
            self.bcs_samplers, self.res_sampler = helmholtz_dataset(
                self.device, a1=a1, a2=a2, lam=lam
            )
            self.pde_op = lambda model, x1, x2: helmholtz_operator(model, x1, x2, lam=lam)
            self._exact_fn = lambda X: helmholtz_exact_u(X, a1=a1, a2=a2)
        else:
            self.bcs_samplers, self.res_sampler = ds_fn(self.device)
            self.pde_op = PDE_OPERATORS[self.pde_name]
            entry = _EXACT_U_REGISTRY.get(self.pde_name)
            self._exact_fn = entry[0] if entry is not None else None

        # Pre-sample fixed collocation points if requested
        self._fixed_batch = None
        if self.fixed_collocation:
            self._fixed_batch = {
                "res": self.res_sampler.sample(self.batch_size),
                "bc": [sampler.sample(self.batch_size) for sampler in self.bcs_samplers],
            }

    def _validate(self, grid_points=50):
        """Compute validation L2 error on a uniform grid (no training data overlap).

        Uses only a forward pass (no PDE derivatives), so it's cheap even for
        quantum models.  Returns relative L2 percentage, or None if the PDE
        has no registered exact solution.
        """
        entry = _EXACT_U_REGISTRY.get(self.pde_name)
        if self._exact_fn is None or entry is None:
            return None

        _, lo, hi = entry
        dims = len(lo)
        grids = [
            torch.linspace(lo[i], hi[i], grid_points, dtype=torch.float64, device=self.device)
            for i in range(dims)
        ]
        mesh = torch.meshgrid(*grids, indexing="ij")
        X_val = torch.stack([m.flatten() for m in mesh], dim=1)

        with torch.no_grad():
            u_exact = self._exact_fn(X_val)
            u_pred = self.model(X_val)

        norm_exact = torch.norm(u_exact)
        if norm_exact < 1e-10:
            return None
        return (torch.norm(u_pred - u_exact) / norm_exact * 100).item()

    def _compute_loss(self, fixed_batch=None):
        """Compute total PINN loss = L_residual + bc_weight * L_boundary."""
        if fixed_batch is None:
            # Fresh stochastic collocation samples (default Adam path)
            X_res, f_res = self.res_sampler.sample(self.batch_size)
            bc_batches = [sampler.sample(self.batch_size) for sampler in self.bcs_samplers]
        else:
            X_res, f_res = fixed_batch["res"]
            bc_batches = fixed_batch["bc"]

        # All 2D PDE operators take (model, col0, col1) as first 3 args
        _, r_pred = self.pde_op(self.model, X_res[:, 0:1], X_res[:, 1:2])

        loss_r = self.model.loss_fn(r_pred, f_res)

        # Boundary losses
        loss_bc = torch.tensor(0.0, dtype=torch.float64, device=self.device)
        for X_bc, u_bc in bc_batches:
            u_pred = self.model(X_bc)
            loss_bc = loss_bc + self.model.loss_fn(u_pred, u_bc)

        total = loss_r + self.bc_weight * loss_bc
        if self.postprocessor_gain_penalty > 0 and hasattr(self.model, "postprocessor"):
            w1 = self.model.postprocessor[0].weight
            w2 = self.model.postprocessor[2].weight
            gain_proxy = torch.linalg.matrix_norm(w1, ord=2) * torch.linalg.matrix_norm(w2, ord=2)
            total = total + self.postprocessor_gain_penalty * gain_proxy
        return total, loss_r, loss_bc

    def train(self, callback=None):
        """Run the training loop.

        Args:
            callback: Optional callable(epoch, loss_val) invoked after each epoch.
                      Useful for Optuna pruning or custom logging.
        """
        self.logger.info("=" * 60)
        self.logger.info(f"Training {self.pde_name} | mode={self.config.get('mode')} | "
                         f"qubits={self.config.get('num_qubits')} | "
                         f"noise={self.config.get('noise_strength', 0.0)}")
        self.logger.info(f"Parameters: {self.model.count_parameters()}")
        self.logger.info("=" * 60)

        # Save config
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            # Convert non-serializable items
            cfg_save = {}
            for k, v in self.config.items():
                try:
                    json.dumps(v)
                    cfg_save[k] = v
                except (TypeError, ValueError):
                    cfg_save[k] = str(v)
            json.dump(cfg_save, f, indent=2)

        use_lbfgs = isinstance(self.model.optimizer, torch.optim.LBFGS)
        best_loss = float("inf")
        best_val_l2 = float("inf")
        val_every = self.config.get("val_every", 100)
        self.val_history = []

        for epoch in range(self.epochs + 1):
            t0 = time.time()

            # Determine the collocation batch for this epoch
            batch = self._fixed_batch  # None if stochastic mode

            if use_lbfgs:
                # L-BFGS line search assumes a consistent objective.
                # Use the fixed batch if available, else sample once per step.
                if batch is None:
                    batch = {
                        "res": self.res_sampler.sample(self.batch_size),
                        "bc": [sampler.sample(self.batch_size) for sampler in self.bcs_samplers],
                    }

                def closure():
                    self.model.optimizer.zero_grad()
                    loss, _, _ = self._compute_loss(fixed_batch=batch)
                    loss.backward()
                    return loss
                self.model.optimizer.step(closure)
                # Recompute for logging. This must run with autograd enabled
                # because the PDE operators build first/second derivatives.
                total, loss_r, loss_bc = self._compute_loss(fixed_batch=batch)
            else:
                self.model.optimizer.zero_grad()
                total, loss_r, loss_bc = self._compute_loss(fixed_batch=batch)
                total.backward()  # No retain_graph needed!
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.model.optimizer.step()

            # Step scheduler
            loss_val = total.item()
            self.model.loss_history.append(loss_val)

            if callback is not None:
                callback(epoch, loss_val)

            if not use_lbfgs:
                self.model.scheduler.step(total)

            dt = time.time() - t0

            # Save best
            if loss_val < best_loss:
                best_loss = loss_val
                self.model.save_state(os.path.join(self.log_dir, "best_model.pth"))

            # Validation-based model selection
            if epoch > 0 and epoch % val_every == 0:
                val_l2 = self._validate()
                if val_l2 is not None:
                    self.val_history.append((epoch, val_l2))
                    if val_l2 < best_val_l2:
                        best_val_l2 = val_l2
                        self.model.save_state(os.path.join(self.log_dir, "best_val_model.pth"))

            if epoch % self.print_every == 0:
                lr = self.model.optimizer.param_groups[0]["lr"]
                val_str = ""
                if self.val_history:
                    val_str = f" | val_l2={self.val_history[-1][1]:.2f}%"
                self.logger.info(
                    f"Epoch {epoch:5d} | loss={loss_val:.3e} | "
                    f"loss_r={loss_r.item():.3e} | loss_bc={loss_bc.item():.3e} | "
                    f"lr={lr:.2e} | dt={dt:.2f}s{val_str}"
                )

        # Final save
        self.model.save_state(os.path.join(self.log_dir, "final_model.pth"))
        val_msg = ""
        if best_val_l2 < float("inf"):
            val_msg = f" | Best val L2: {best_val_l2:.2f}%"
        self.logger.info(f"Training complete. Best loss: {best_loss:.4e}{val_msg}")
        return best_loss
