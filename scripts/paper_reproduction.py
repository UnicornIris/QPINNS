#!/usr/bin/env python3
"""Reproduce paper-quality TE-QPINN results on Helmholtz equation.

This script matches the methodology of:
  - Berger et al. (2025): L-BFGS, phi(x)*x embedding, fixed collocation
  - Tran et al. (2026): 4-8 qubits, 5+ layers, HEA ansatz

Key settings aligned with papers:
  - L-BFGS optimizer with strong Wolfe line search
  - Fixed collocation grid (no stochastic resampling)
  - phi(x)*x embedding (Berger Eq. 11)
  - TE width=10, 2 hidden layers (Berger architecture)
  - 4 qubits, 5 variational layers
  - Domain rescaled to [-0.95, 0.95]
  - BC weight = 10
"""

import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcpinn.solver import QCPINNSolver
from qcpinn.trainer import Trainer
from qcpinn.evaluation import evaluate_helmholtz, plot_results, plot_loss_history


def run_experiment(mode, config_overrides=None, label=None):
    """Run a single experiment and return results."""
    label = label or mode

    config = {
        "problem": "helmholtz",
        "input_dim": 2,
        "output_dim": 1,
        "num_qubits": 4,
        "num_quantum_layers": 5,     # Paper uses 5+
        "q_ansatz": "hea",
        "hidden_dim": 50,
        "mode": mode,
        "optimizer": "lbfgs",        # Key: L-BFGS like both papers
        "lr": 1e-3,
        "epochs": 300,               # L-BFGS converges much faster
        "batch_size": 200,           # Larger batch for L-BFGS stability
        "bc_weight": 10.0,
        "print_every": 10,
        "fixed_collocation": True,   # Key: fixed grid like papers
        "noise_strength": 0.0,
        "domain_lo": [-1.0, -1.0],
        "domain_hi": [1.0, 1.0],
        # TE parameters (matching Berger et al.)
        "te_hidden_layers": 2,
        "te_width": 10,
    }

    if config_overrides:
        config.update(config_overrides)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"experiments/paper_repro_{timestamp}_{label}"

    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"Mode: {mode}, Qubits: {config['num_qubits']}, Layers: {config['num_quantum_layers']}")
    print(f"Optimizer: {config['optimizer']}, Epochs: {config['epochs']}")
    print(f"{'='*60}")

    model = QCPINNSolver(config)
    params = model.count_parameters()
    print(f"Parameters: {params}")

    trainer = Trainer(model, config, log_dir)
    t0 = time.time()
    best_loss = trainer.train()
    train_time = time.time() - t0

    # Load best model for evaluation
    best_model = QCPINNSolver.load_state(os.path.join(log_dir, "best_model.pth"))
    metrics = evaluate_helmholtz(best_model, grid_points=100)

    print(f"\n--- Results for {label} ---")
    print(f"  Best loss:     {best_loss:.4e}")
    print(f"  Rel L2 (u):    {metrics['rel_l2_u']:.4f}%")
    print(f"  MSE (u):       {metrics['mse_u']:.4e}")
    print(f"  Max error (u): {metrics['max_err_u']:.4e}")
    print(f"  Train time:    {train_time:.1f}s")
    print(f"  Parameters:    {params['total']}")

    # Save plots
    plot_results(metrics, log_dir, title_prefix=f"{label}: ")
    plot_loss_history(model.loss_history, log_dir, title=f"{label} Loss History")

    # Save results
    results = {
        "label": label,
        "mode": mode,
        "config": {k: str(v) if not isinstance(v, (int, float, str, bool, list)) else v
                   for k, v in config.items()},
        "best_loss": best_loss,
        "rel_l2_u": metrics["rel_l2_u"],
        "mse_u": metrics["mse_u"],
        "max_err_u": metrics["max_err_u"],
        "train_time": train_time,
        "params": params,
        "loss_history": model.loss_history,
    }
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("PAPER REPRODUCTION: Helmholtz TE-QPINN")
    print("Methodology: L-BFGS + phi(x)*x + fixed collocation")
    print("=" * 60)

    all_results = {}

    # --- Experiment 1: Classical PINN baseline ---
    r = run_experiment("classical", {
        "hidden_dim": 50,
        "epochs": 300,
    }, label="classical_pinn")
    all_results["Classical PINN"] = r

    # --- Experiment 2: TE-QPINN (4 qubits, 5 layers) ---
    r = run_experiment("te", {
        "num_qubits": 4,
        "num_quantum_layers": 5,
        "epochs": 300,
    }, label="te_4q5l")
    all_results["TE-QPINN (4Q/5L)"] = r

    # --- Experiment 3: Baseline QPINN (4 qubits, 5 layers) ---
    r = run_experiment("baseline", {
        "num_qubits": 4,
        "num_quantum_layers": 5,
        "epochs": 300,
    }, label="baseline_4q5l")
    all_results["Baseline QPINN (4Q/5L)"] = r

    # --- Experiment 4: TE-QPINN with more qubits (6 qubits, 5 layers) ---
    r = run_experiment("te", {
        "num_qubits": 6,
        "num_quantum_layers": 5,
        "epochs": 300,
    }, label="te_6q5l")
    all_results["TE-QPINN (6Q/5L)"] = r

    # --- Summary ---
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Method':<25} {'Rel L2 (%)':<12} {'MSE':<12} {'Best Loss':<12} {'Params':<8} {'Time (s)':<10}")
    print("-" * 80)
    for name, r in all_results.items():
        print(f"{name:<25} {r['rel_l2_u']:<12.4f} {r['mse_u']:<12.4e} {r['best_loss']:<12.4e} "
              f"{r['params']['total']:<8} {r['train_time']:<10.1f}")
    print("=" * 80)

    # --- Comparison plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    for name, r in all_results.items():
        axes[0].semilogy(r["loss_history"], label=name, linewidth=1.2)
    axes[0].set_xlabel("L-BFGS Iteration", fontsize=12)
    axes[0].set_ylabel("Total Loss", fontsize=12)
    axes[0].set_title("Training Convergence", fontsize=14)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Error bars
    names = list(all_results.keys())
    l2_errors = [all_results[n]["rel_l2_u"] for n in names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = axes[1].bar(names, l2_errors, color=colors)
    axes[1].set_ylabel("Relative L2 Error (%)", fontsize=12)
    axes[1].set_title("Solution Accuracy", fontsize=14)
    axes[1].tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, l2_errors):
        axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    outdir = "experiments/paper_reproduction_summary"
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "comparison.pdf"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(outdir, "comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save combined results
    save_results = {}
    for name, r in all_results.items():
        save_results[name] = {k: v for k, v in r.items() if k != "loss_history"}
        save_results[name]["final_loss"] = r["loss_history"][-1] if r["loss_history"] else None
    with open(os.path.join(outdir, "all_results.json"), "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\nAll results saved to {outdir}/")


if __name__ == "__main__":
    main()
