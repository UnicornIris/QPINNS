#!/usr/bin/env python3
"""Heat equation: TE vs Baseline vs Classical comparison.

Inspired by the x-TE-QPINN papers (arXiv:2602.09291, 2602.14596), this script
runs a controlled comparison on the 1D heat equation:
    u_t = D * u_xx,   x in [-1,1], t in [0,1]
    u(x,0) = sin(pi*x),  u(-1,t) = u(1,t) = 0
    Exact: u(x,t) = sin(pi*x) * exp(-D*pi^2*t),  D = 0.01/pi

Three modes:
  - classical:  Pure MLP (no quantum)
  - baseline:   Classical preprocessor -> quantum circuit -> postprocessor
  - te:         Trainable embedding -> quantum circuit -> postprocessor

Key design choices (following the papers):
  - L-BFGS optimizer with strong Wolfe line search
  - Fixed collocation points per seed (no stochastic resampling)
  - HEA ansatz (RX-RY-RZ + CNOT chain)
  - 4 qubits, 3 quantum layers
  - Multiple seeds for statistical robustness
"""

import os
import sys
import json
import time
import copy
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcpinn.solver import QCPINNSolver
from qcpinn.trainer import Trainer
from qcpinn.evaluation import evaluate_heat_1d, plot_results, plot_loss_history


# ─── Configuration ──────────────────────────────────────────────────────────

# Shared defaults — quantum-specific overrides applied in run_study()
COMMON_CONFIG = {
    "problem": "heat_1d",
    "input_dim": 2,
    "output_dim": 1,
    "num_qubits": 4,
    "num_quantum_layers": 2,
    "hidden_dim": 30,
    "q_ansatz": "hea",
    "encoding": "angle",
    "bc_weight": 10.0,
    "grad_clip": 1.0,
    "noise_strength": 0.0,
    "shots": None,
    "qml_device": "default.qubit",
    "domain_lo": [0.0, -1.0],
    "domain_hi": [1.0, 1.0],
    "te_hidden_layers": 2,
    "te_width": 16,
}

# Classical: L-BFGS, big batch, many epochs (fast ~10s)
CLASSICAL_OVERRIDES = {
    "epochs": 200,
    "batch_size": 128,
    "lr": 1e-2,
    "optimizer": "lbfgs",
    "print_every": 20,
}

# Quantum: Adam, small batch (slow per-sample circuit eval)
QUANTUM_OVERRIDES = {
    "epochs": 150,
    "batch_size": 16,
    "lr": 5e-3,
    "optimizer": "adam",
    "print_every": 15,
}

SEEDS = [0, 1]
MODES = ["classical", "baseline", "te"]


# ─── Experiment Runner ──────────────────────────────────────────────────────

def run_one(config, log_dir, seed, device):
    """Train a single model and return metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = QCPINNSolver(config, device=device)
    trainer = Trainer(model, config, log_dir, device=device)

    t0 = time.time()
    best_loss = trainer.train()
    train_time = time.time() - t0

    # Evaluate best checkpoint
    best_ckpt = os.path.join(log_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        model = QCPINNSolver.load_state(best_ckpt, device=device)

    # Use coarser grid for quantum models (10k point eval is too slow with circuit sim)
    grid = 30 if config["mode"] != "classical" else 100
    metrics = evaluate_heat_1d(model, grid_points=grid, device=device)
    plot_results(metrics, log_dir, title_prefix=f"{config['mode'].upper()} seed={seed} ")
    plot_loss_history(model.loss_history, log_dir,
                      title=f"heat_1d | {config['mode']} | seed={seed}")

    metrics["best_loss"] = best_loss
    metrics["train_time_sec"] = train_time
    metrics["total_params"] = model.count_parameters()
    metrics["loss_history"] = model.loss_history
    metrics["seed"] = seed

    # Save result.json (no large arrays)
    save_metrics = {
        "mode": config["mode"],
        "seed": seed,
        "best_loss": best_loss,
        "rel_l2_u": metrics["rel_l2_u"],
        "mse_u": metrics["mse_u"],
        "max_err_u": metrics["max_err_u"],
        "train_time_sec": train_time,
        "total_params": model.count_parameters(),
        "num_qubits": config.get("num_qubits"),
        "num_quantum_layers": config.get("num_quantum_layers"),
        "epochs": config["epochs"],
        "optimizer": config["optimizer"],
        "batch_size": config["batch_size"],
    }
    with open(os.path.join(log_dir, "result.json"), "w") as f:
        json.dump(save_metrics, f, indent=2, default=str)

    return metrics


def make_config(mode):
    """Build config with mode-appropriate overrides."""
    config = copy.deepcopy(COMMON_CONFIG)
    config["mode"] = mode
    if mode == "classical":
        config.update(CLASSICAL_OVERRIDES)
    else:
        config.update(QUANTUM_OVERRIDES)
    return config


def run_study(out_dir, device):
    """Run the full comparison study."""
    os.makedirs(out_dir, exist_ok=True)
    all_results = {}

    for mode in MODES:
        mode_results = []
        for seed in SEEDS:
            name = f"{mode}_seed{seed}"
            log_dir = os.path.join(out_dir, name)
            print(f"\n{'='*60}")
            print(f"  Running: {name}")
            print(f"{'='*60}")

            config = make_config(mode)

            metrics = run_one(config, log_dir, seed, device)
            mode_results.append(metrics)
            print(f"  -> rel_l2_u = {metrics['rel_l2_u']:.2f}%  |  "
                  f"mse_u = {metrics['mse_u']:.2e}  |  "
                  f"time = {metrics['train_time_sec']:.1f}s")

        all_results[mode] = mode_results

    return all_results


# ─── Plotting ───────────────────────────────────────────────────────────────

def plot_comparison_summary(all_results, out_dir):
    """Generate summary comparison plots."""
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Loss curves (mean ± std) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"classical": "#2196F3", "baseline": "#FF9800", "te": "#4CAF50"}
    labels = {"classical": "Classical PINN", "baseline": "Baseline QPINN", "te": "TE-QPINN"}

    for mode in MODES:
        histories = [r["loss_history"] for r in all_results[mode]]
        min_len = min(len(h) for h in histories)
        arr = np.array([h[:min_len] for h in histories])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        epochs = np.arange(min_len)

        ax.semilogy(epochs, mean, label=labels[mode], color=colors[mode], linewidth=2)
        ax.fill_between(epochs, np.maximum(mean - std, 1e-12), mean + std,
                        alpha=0.2, color=colors[mode])

    ax.set_xlabel("Epoch (L-BFGS iteration)", fontsize=13)
    ax.set_ylabel("Total Loss", fontsize=13)
    ax.set_title("1D Heat Equation: Training Convergence", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_comparison.pdf"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "loss_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- 2. Bar chart: relative L2 error ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(MODES))
    means = []
    stds = []
    for mode in MODES:
        errs = [r["rel_l2_u"] for r in all_results[mode]]
        means.append(np.mean(errs))
        stds.append(np.std(errs))

    bars = ax.bar(x_pos, means, yerr=stds, capsize=8,
                  color=[colors[m] for m in MODES], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[m] for m in MODES], fontsize=12)
    ax.set_ylabel("Relative L2 Error (%)", fontsize=13)
    ax.set_title("1D Heat Equation: Solution Accuracy", fontsize=15)

    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + s + 0.5,
                f'{m:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_comparison.pdf"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "accuracy_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- 3. MSE comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    means_mse = []
    stds_mse = []
    for mode in MODES:
        mses = [r["mse_u"] for r in all_results[mode]]
        means_mse.append(np.mean(mses))
        stds_mse.append(np.std(mses))

    bars = ax.bar(x_pos, means_mse, yerr=stds_mse, capsize=8,
                  color=[colors[m] for m in MODES], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[m] for m in MODES], fontsize=12)
    ax.set_ylabel("MSE", fontsize=13)
    ax.set_title("1D Heat Equation: Mean Squared Error", fontsize=15)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mse_comparison.pdf"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "mse_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # --- 4. Summary table ---
    summary = {}
    print("\n" + "=" * 85)
    print(f"{'Mode':<20} {'Params':>8} {'Rel L2 (%)':>14} {'MSE':>14} {'Time (s)':>10}")
    print("-" * 85)
    for mode in MODES:
        errs = [r["rel_l2_u"] for r in all_results[mode]]
        mses = [r["mse_u"] for r in all_results[mode]]
        times = [r["train_time_sec"] for r in all_results[mode]]
        params = all_results[mode][0]["total_params"]

        row = {
            "mode": mode,
            "label": labels[mode],
            "total_params": params,
            "rel_l2_mean": float(np.mean(errs)),
            "rel_l2_std": float(np.std(errs)),
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "time_mean": float(np.mean(times)),
            "seeds": len(errs),
        }
        summary[mode] = row
        print(f"{labels[mode]:<20} {params.get('total', '?'):>8} "
              f"{row['rel_l2_mean']:>8.2f} +/- {row['rel_l2_std']:<5.2f}"
              f"{row['mse_mean']:>12.2e}  {row['time_mean']:>8.1f}")
    print("=" * 85)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ─── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cpu")  # Quantum simulation is CPU-only in PennyLane
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("experiments", f"heat_te_comparison_{timestamp}")

    print(f"Output directory: {out_dir}")
    print(f"Common config: {json.dumps(COMMON_CONFIG, indent=2, default=str)}")
    print(f"Classical overrides: {json.dumps(CLASSICAL_OVERRIDES, indent=2)}")
    print(f"Quantum overrides: {json.dumps(QUANTUM_OVERRIDES, indent=2)}")

    all_results = run_study(out_dir, device)
    summary = plot_comparison_summary(all_results, out_dir)

    print(f"\nResults saved to: {out_dir}")
