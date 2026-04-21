#!/usr/bin/env python3
"""Noise Robustness Study for TE-QPINN Paper.

This script runs the core noise experiment:
  Phase 1 (fast): Load pre-trained clean models, evaluate at various noise levels
  Phase 2 (slow): Train noise-aware models, full cross-noise evaluation

The key hypothesis: TE-QPINN degrades more gracefully under noise than
standard QPINN (repeat/angle embedding), and noise-aware training with TE
can partially compensate for hardware errors.

Usage:
    # Phase 1 only (uses existing checkpoints, ~30 min)
    python scripts/noise_robustness.py --phase 1

    # Full study (Phase 1 + Phase 2 training, ~hours)
    python scripts/noise_robustness.py --phase 2

    # Custom noise levels
    python scripts/noise_robustness.py --phase 2 --train-noise 0.0 0.01 0.02 --test-noise 0.0 0.005 0.01 0.02 0.05
"""

import os
import sys
import gc
import json
import time
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcpinn.solver import QCPINNSolver
from qcpinn.trainer import Trainer
from qcpinn.evaluation import evaluate_helmholtz, plot_loss_history
from qcpinn.datasets import DATASET_REGISTRY, get_helmholtz_params


# ---------------------------------------------------------------------------
# Noise-swapped inference
# ---------------------------------------------------------------------------

def swap_noise_level(model, new_noise):
    """Rebuild model with different noise level, preserving trained weights."""
    config = dict(model.config)
    config["noise_strength"] = new_noise

    model_new = QCPINNSolver(config, device=model.device)

    old_state = model.state_dict()
    new_state = model_new.state_dict()
    for key in old_state:
        if key in new_state and old_state[key].shape == new_state[key].shape:
            new_state[key] = old_state[key].clone()
    model_new.load_state_dict(new_state)
    model_new.eval()
    return model_new


def evaluate_solution_only(model, grid_points=50, device=None):
    """Evaluate solution accuracy via forward-only passes (no autograd).

    Much faster than evaluate_helmholtz for noisy circuits because it
    avoids computing PDE residuals (second-order derivatives through
    the quantum circuit). For noise robustness, we care about solution
    accuracy under noise, not PDE residual quality.
    """
    from qcpinn.datasets import helmholtz_exact_u
    device = device or model.device
    a1, a2, _ = get_helmholtz_params(getattr(model, "config", {}))

    t = torch.linspace(-1, 1, grid_points, dtype=torch.float64, device=device)
    x = torch.linspace(-1, 1, grid_points, dtype=torch.float64, device=device)
    T, X = torch.meshgrid(t, x, indexing="ij")
    X_star = torch.stack([T.flatten(), X.flatten()], dim=1)

    with torch.no_grad():
        u_star = helmholtz_exact_u(X_star, a1=a1, a2=a2)
        u_pred = model(X_star)

    err_u = torch.norm(u_pred - u_star) / torch.norm(u_star) * 100
    mse_u = torch.mean((u_pred - u_star) ** 2)
    max_err_u = torch.max(torch.abs(u_pred - u_star))

    return {
        "rel_l2_u": err_u.item(),
        "mse_u": mse_u.item(),
        "max_err_u": max_err_u.item(),
    }


def evaluate_at_noise(model, noise_level, grid_points=50, device=None):
    """Evaluate a trained model at a specific noise level."""
    device = device or model.device
    if noise_level == 0.0 and model.config.get("noise_strength", 0.0) == 0.0:
        test_model = model
    else:
        test_model = swap_noise_level(model, noise_level)

    # Use forward-only evaluation (no autograd) for speed with noisy circuits
    metrics = evaluate_solution_only(test_model, grid_points=grid_points, device=device)

    if test_model is not model:
        del test_model
        gc.collect()

    return metrics


# ---------------------------------------------------------------------------
# Phase 1: Evaluate existing clean checkpoints under noise
# ---------------------------------------------------------------------------

def phase1_inference_noise(
    te_checkpoint,
    baseline_checkpoint,
    test_noise_levels,
    output_dir,
    device,
    grid_points=50,
):
    """Evaluate pre-trained clean models at various noise levels."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  PHASE 1: Inference-time noise robustness")
    print("  (Using pre-trained clean models, no new training)")
    print("=" * 70)

    # Load models
    print(f"\nLoading TE-QPINN: {te_checkpoint}")
    te_model = QCPINNSolver.load_state(te_checkpoint, device=device)
    te_model.eval()
    print(f"  Config: mode={te_model.config['mode']}, qubits={te_model.config['num_qubits']}, "
          f"layers={te_model.config['num_quantum_layers']}, params={te_model.count_parameters()['total']}")

    print(f"\nLoading Fixed-Angle QPINN (repeat mode): {baseline_checkpoint}")
    bl_model = QCPINNSolver.load_state(baseline_checkpoint, device=device)
    bl_model.eval()
    print(f"  Config: mode={bl_model.config['mode']}, qubits={bl_model.config['num_qubits']}, "
          f"layers={bl_model.config['num_quantum_layers']}, params={bl_model.count_parameters()['total']}")

    results = {"te_clean": {}, "baseline_clean": {}}

    for noise in test_noise_levels:
        print(f"\n  Testing at noise={noise:.4f}...")

        t0 = time.time()
        te_metrics = evaluate_at_noise(te_model, noise, grid_points, device)
        te_time = time.time() - t0
        results["te_clean"][str(noise)] = {
            "rel_l2_u": te_metrics["rel_l2_u"],
            "mse_u": te_metrics["mse_u"],
            "max_err_u": te_metrics["max_err_u"],
            "eval_time": te_time,
        }
        print(f"    TE-QPINN:      L2={te_metrics['rel_l2_u']:.2f}%  ({te_time:.1f}s)")

        t0 = time.time()
        bl_metrics = evaluate_at_noise(bl_model, noise, grid_points, device)
        bl_time = time.time() - t0
        results["baseline_clean"][str(noise)] = {
            "rel_l2_u": bl_metrics["rel_l2_u"],
            "mse_u": bl_metrics["mse_u"],
            "max_err_u": bl_metrics["max_err_u"],
            "eval_time": bl_time,
        }
        print(f"    Fixed-Angle QPINN: L2={bl_metrics['rel_l2_u']:.2f}%  ({bl_time:.1f}s)")

    # Save results
    with open(os.path.join(output_dir, "phase1_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Generate plot
    _plot_phase1(results, test_noise_levels, output_dir)

    del te_model, bl_model
    gc.collect()

    return results


# ---------------------------------------------------------------------------
# Phase 2: Train noise-aware models + full cross-noise evaluation
# ---------------------------------------------------------------------------

def phase2_noise_aware_training(
    train_noise_levels,
    test_noise_levels,
    epochs,
    output_dir,
    device,
    grid_points=50,
    seed=42,
):
    """Train noise-aware models and do full cross-noise evaluation."""
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  PHASE 2: Noise-aware training + cross-noise evaluation")
    print(f"  Train noise: {train_noise_levels}")
    print(f"  Test noise:  {test_noise_levels}")
    print(f"  Epochs: {epochs}")
    print("=" * 70)

    # Base config matching the paper / full_comparison_fixed settings
    base_cfg = {
        "problem": "helmholtz",
        "input_dim": 2,
        "output_dim": 1,
        "optimizer": "lbfgs",
        "lr": 1e-3,
        "epochs": epochs,
        "batch_size": 1000,
        "bc_weight": 10.0,
        "print_every": max(epochs // 20, 1),
        "val_every": 100,
        "fixed_collocation": True,
        "domain_lo": [-1.0, -1.0],
        "domain_hi": [1.0, 1.0],
        "num_qubits": 4,
        "num_quantum_layers": 5,
        "q_ansatz": "hea",
        "hidden_dim": 50,
        "seed": seed,
    }

    modes = ["repeat", "te"]
    trained_models = {}  # (mode, train_noise) -> model
    results = {}  # mode -> train_noise -> test_noise -> metrics

    # --- Train models ---
    for mode in modes:
        results[mode] = {}
        for train_noise in train_noise_levels:
            name = f"{mode}_train_noise{train_noise:.3f}"
            log_dir = os.path.join(output_dir, "training", name)

            print(f"\n{'='*60}")
            print(f"  Training: {name} ({epochs} epochs)")
            print(f"{'='*60}")

            cfg = dict(base_cfg)
            cfg["mode"] = mode
            cfg["noise_strength"] = train_noise
            if mode == "te":
                cfg["te_hidden_layers"] = 2
                cfg["te_width"] = 10

            # For noisy training, switch to Adam (L-BFGS does multiple
            # closure evals per step, each requiring per-sample loop through
            # default.mixed — a single L-BFGS epoch can take hours).
            # Adam does one forward+backward per step, much more feasible.
            # Also resample collocation points each step: keeping a fixed
            # 64-point batch would make the noisy run both brittle and
            # incomparable to the full-batch noiseless setting.
            if train_noise > 0:
                cfg["optimizer"] = "adam"
                cfg["lr"] = 5e-4
                cfg["batch_size"] = 64
                cfg["fixed_collocation"] = False
                cfg["print_every"] = max(epochs // 10, 1)

            torch.manual_seed(seed)
            np.random.seed(seed)

            model = QCPINNSolver(cfg, device=device)
            trainer = Trainer(model, cfg, log_dir, device=device)

            t0 = time.time()
            best_loss = trainer.train()
            train_time = time.time() - t0

            # Load best validation model
            val_path = os.path.join(log_dir, "best_val_model.pth")
            best_path = os.path.join(log_dir, "best_model.pth")
            ckpt = val_path if os.path.exists(val_path) else best_path
            best_model = QCPINNSolver.load_state(ckpt, device=device)
            best_model.eval()

            trained_models[(mode, train_noise)] = best_model
            plot_loss_history(model.loss_history, log_dir, title=f"{name}")

            print(f"  Trained in {train_time:.0f}s, best loss: {best_loss:.3e}")
            print(f"  Params: {model.count_parameters()}")

            del model, trainer
            gc.collect()

    # --- Cross-noise evaluation ---
    print("\n" + "=" * 70)
    print("  Cross-noise evaluation")
    print("=" * 70)

    for mode in modes:
        for train_noise in train_noise_levels:
            results[mode][str(train_noise)] = {}
            model = trained_models[(mode, train_noise)]

            for test_noise in test_noise_levels:
                label = f"{mode}/train={train_noise:.3f}/test={test_noise:.3f}"
                print(f"  {label}...", end=" ", flush=True)

                metrics = evaluate_at_noise(model, test_noise, grid_points, device)
                results[mode][str(train_noise)][str(test_noise)] = {
                    "rel_l2_u": metrics["rel_l2_u"],
                    "mse_u": metrics["mse_u"],
                    "max_err_u": metrics["max_err_u"],
                }
                print(f"L2={metrics['rel_l2_u']:.2f}%")

    # Save
    with open(os.path.join(output_dir, "phase2_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    _plot_phase2(results, train_noise_levels, test_noise_levels, output_dir)
    _save_latex_table(results, train_noise_levels, test_noise_levels, output_dir)

    # Cleanup
    for m in trained_models.values():
        del m
    gc.collect()

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_phase1(results, test_noise_levels, save_dir):
    """Plot Phase 1: clean-trained models tested at various noise levels."""
    fig, ax = plt.subplots(figsize=(10, 6))

    te_l2 = [results["te_clean"][str(n)]["rel_l2_u"] for n in test_noise_levels]
    bl_l2 = [results["baseline_clean"][str(n)]["rel_l2_u"] for n in test_noise_levels]

    ax.plot(test_noise_levels, te_l2, "D-", color="#d62728", linewidth=2,
            markersize=8, label="TE-QPINN (4Q/5L, trained clean)")
    ax.plot(test_noise_levels, bl_l2, "o--", color="#1f77b4", linewidth=2,
            markersize=8, label="Fixed-Angle QPINN (repeat, trained clean)")

    ax.set_xlabel("Test-time Depolarizing Noise Strength (p)", fontsize=13)
    ax.set_ylabel("Relative L2 Error (%)", fontsize=13)
    ax.set_title("Noise Robustness: Clean-Trained Models Under Noise", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(save_dir, f"phase1_noise_robustness.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: phase1_noise_robustness.pdf/png")


def _plot_phase2(results, train_noise_levels, test_noise_levels, save_dir):
    """Plot Phase 2: full cross-noise figure for the paper."""
    fig, ax = plt.subplots(figsize=(10, 6))

    styles = {
        ("repeat", 0.0):  {"color": "#1f77b4", "ls": "--", "marker": "o",
                           "label": "Fixed-Angle QPINN (repeat, trained clean)"},
        ("te", 0.0):      {"color": "#d62728", "ls": "--", "marker": "D",
                           "label": "TE-QPINN (trained clean)"},
    }
    # Dynamically add styles for noise-aware models
    noise_colors_repeat = ["#aec7e8", "#6baed6", "#2171b5"]
    noise_colors_te = ["#fcbba1", "#fb6a4a", "#cb181d"]
    ni = 0
    for tn in train_noise_levels:
        if tn > 0:
            c_idx = min(ni, len(noise_colors_repeat) - 1)
            styles[("repeat", tn)] = {
                "color": noise_colors_repeat[c_idx], "ls": "-", "marker": "s",
                "label": f"Fixed-Angle QPINN (repeat, noise-aware, p={tn})"
            }
            styles[("te", tn)] = {
                "color": noise_colors_te[c_idx], "ls": "-", "marker": "^",
                "label": f"TE-QPINN (noise-aware, p={tn})"
            }
            ni += 1

    for mode in results:
        for train_noise_str in results[mode]:
            train_noise = float(train_noise_str)
            key = (mode, train_noise)
            if key not in styles:
                continue
            style = styles[key]

            xs, ys = [], []
            for tn in test_noise_levels:
                tn_str = str(tn)
                if tn_str in results[mode][train_noise_str]:
                    xs.append(tn)
                    ys.append(results[mode][train_noise_str][tn_str]["rel_l2_u"])

            ax.plot(xs, ys, color=style["color"], linestyle=style["ls"],
                    marker=style["marker"], label=style["label"],
                    linewidth=1.5, markersize=7)

    ax.set_xlabel("Test-time Depolarizing Noise Strength (p)", fontsize=13)
    ax.set_ylabel("Relative L2 Error (%)", fontsize=13)
    ax.set_title("Cross-Noise Robustness: Helmholtz Equation", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=11)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(save_dir, f"cross_noise_robustness.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: cross_noise_robustness.pdf/png")


def _save_latex_table(results, train_noise_levels, test_noise_levels, save_dir):
    """Save results as LaTeX table."""
    rows = []
    for mode in results:
        mode_label = "TE-QPINN" if mode == "te" else "Fixed-Angle QPINN"
        for tn_train_str in results[mode]:
            tn_train = float(tn_train_str)
            train_label = "clean" if tn_train == 0 else f"p={tn_train}"
            for tn_test in test_noise_levels:
                tn_test_str = str(tn_test)
                if tn_test_str in results[mode][tn_train_str]:
                    m = results[mode][tn_train_str][tn_test_str]
                    rows.append((
                        mode_label, train_label,
                        f"{tn_test:.3f}",
                        f"{m['rel_l2_u']:.2f}",
                        f"{m['mse_u']:.2e}",
                    ))

    with open(os.path.join(save_dir, "noise_results_table.tex"), "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Cross-noise evaluation on Helmholtz equation}\n")
        f.write("\\label{tab:cross_noise}\n")
        f.write("\\begin{tabular}{llccc}\n\\toprule\n")
        f.write("Method & Training & Test Noise & Rel.~L2 (\\%) & MSE \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # Also save plain text
    with open(os.path.join(save_dir, "noise_results_table.txt"), "w") as f:
        f.write(f"{'Method':<15} {'Training':<12} {'Test Noise':<12} {'L2 (%)':<10} {'MSE':<12}\n")
        f.write("-" * 65 + "\n")
        for row in rows:
            f.write(f"{row[0]:<15} {row[1]:<12} {row[2]:<12} {row[3]:<10} {row[4]:<12}\n")

    print(f"  Saved: noise_results_table.tex, noise_results_table.txt")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results, phase_label=""):
    """Print a formatted summary table."""
    print(f"\n{'='*75}")
    print(f"  {phase_label} RESULTS SUMMARY")
    print(f"{'='*75}")

    if "te_clean" in results:
        # Phase 1 format
        print(f"  {'Noise':<10} {'TE-QPINN L2%':<15} {'Fixed-Angle L2%':<15} {'TE advantage':<15}")
        print(f"  {'-'*55}")
        for noise_str in results["te_clean"]:
            te_l2 = results["te_clean"][noise_str]["rel_l2_u"]
            bl_l2 = results["baseline_clean"][noise_str]["rel_l2_u"]
            ratio = bl_l2 / max(te_l2, 1e-10)
            print(f"  {noise_str:<10} {te_l2:<15.2f} {bl_l2:<15.2f} {ratio:<15.1f}x")
    else:
        # Phase 2 format
        for mode in results:
            print(f"\n  --- {mode.upper()} ---")
            for tn_train in results[mode]:
                print(f"  Trained at noise={tn_train}:")
                for tn_test in results[mode][tn_train]:
                    m = results[mode][tn_train][tn_test]
                    print(f"    test={tn_test}: L2={m['rel_l2_u']:.2f}%")

    print(f"{'='*75}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Noise Robustness Study for TE-QPINN")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="1=inference-only (fast), 2=full training+evaluation")
    parser.add_argument("--test-noise", nargs="+", type=float,
                        default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
                        help="Noise levels to test at")
    parser.add_argument("--train-noise", nargs="+", type=float,
                        default=[0.0, 0.01],
                        help="Noise levels to train at (Phase 2 only)")
    parser.add_argument("--epochs", type=int, default=300,
                        help="Training epochs for Phase 2 noise-aware models")
    parser.add_argument("--te-checkpoint", type=str,
                        default="experiments/full_comparison_fixed/te_qpinn_4q5l/best_val_model.pth",
                        help="Path to trained TE-QPINN checkpoint")
    parser.add_argument("--baseline-checkpoint", type=str,
                        default="experiments/full_comparison_fixed/baseline_qpinn_4q5l/best_val_model.pth",
                        help="Path to trained fixed-angle QPINN checkpoint (repeat mode)")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/noise_robustness",
                        help="Output directory")
    parser.add_argument("--grid-points", type=int, default=20,
                        help="Grid points per dim for evaluation (20x20=400 points)")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    device = torch.device("cpu")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    t_start = time.time()

    # Phase 1: Inference-only noise robustness
    phase1_results = phase1_inference_noise(
        te_checkpoint=args.te_checkpoint,
        baseline_checkpoint=args.baseline_checkpoint,
        test_noise_levels=args.test_noise,
        output_dir=output_dir,
        device=device,
        grid_points=args.grid_points,
    )
    print_summary(phase1_results, "PHASE 1")

    if args.phase >= 2:
        phase2_results = phase2_noise_aware_training(
            train_noise_levels=args.train_noise,
            test_noise_levels=args.test_noise,
            epochs=args.epochs,
            output_dir=output_dir,
            device=device,
            grid_points=args.grid_points,
            seed=args.seed,
        )
        print_summary(phase2_results, "PHASE 2")

    total_time = time.time() - t_start
    print(f"\nTotal wall time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()
