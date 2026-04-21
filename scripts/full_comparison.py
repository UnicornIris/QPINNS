#!/usr/bin/env python3
"""Full comparison: Classical PINN vs Fixed-Angle QPINN vs TE-QPINN on Helmholtz.

Runs all 3 methods at 2000 L-BFGS epochs on the Helmholtz equation.
Optionally includes Optuna HPO via --optuna-trials flag.

NOTE: batch_size=1000 (quantum) / 2000 (classical) to prevent overfitting
with fixed collocation.  Previous runs with batch_size=200 caused catastrophic
overfitting at high epochs (L2=365% for classical despite low training loss).

Usage:
    python scripts/full_comparison.py
    python scripts/full_comparison.py --epochs 2000 --optuna-trials 20 --optuna-epochs 100
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
from qcpinn.evaluation import evaluate_helmholtz, plot_results, plot_loss_history

try:
    import optuna
    from optuna.exceptions import TrialPruned
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


# ---------------------------------------------------------------------------
# Experiment runner (shared by fixed-config and Optuna-best retraining)
# ---------------------------------------------------------------------------

def run_experiment(config, label, output_dir):
    """Run a single experiment. Returns results dict."""
    log_dir = os.path.join(output_dir, label)

    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  mode={config['mode']}  qubits={config.get('num_qubits', 'N/A')}  "
          f"layers={config.get('num_quantum_layers', 'N/A')}  epochs={config['epochs']}")
    print(f"{'='*70}")

    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    model = QCPINNSolver(config)
    params = model.count_parameters()
    print(f"  Parameters: {params}")

    trainer = Trainer(model, config, log_dir)
    t0 = time.time()
    best_loss = trainer.train()
    train_time = time.time() - t0

    # Evaluate — prefer validation-best model (best generalization) over
    # training-best model (can overfit fixed collocation points)
    val_model_path = os.path.join(log_dir, "best_val_model.pth")
    train_model_path = os.path.join(log_dir, "best_model.pth")
    model_path = val_model_path if os.path.exists(val_model_path) else train_model_path
    best_model = QCPINNSolver.load_state(model_path)
    best_model.eval()
    metrics = evaluate_helmholtz(best_model, grid_points=100)
    print(f"  Evaluated model: {os.path.basename(model_path)}")

    print(f"\n  --- {label} Results ---")
    print(f"  Best loss:     {best_loss:.4e}")
    print(f"  Rel L2 (u):    {metrics['rel_l2_u']:.4f}%")
    print(f"  MSE (u):       {metrics['mse_u']:.4e}")
    print(f"  Max error (u): {metrics['max_err_u']:.4e}")
    print(f"  Train time:    {train_time:.1f}s")
    print(f"  Parameters:    {params['total']}")

    # Save plots
    plot_results(metrics, log_dir, title_prefix=f"{label}: ")
    plot_loss_history(model.loss_history, log_dir, title=f"{label} Loss History")

    results = {
        "label": label,
        "mode": config["mode"],
        "config": {k: str(v) if not isinstance(v, (int, float, str, bool, list)) else v
                   for k, v in config.items()},
        "best_loss": best_loss,
        "rel_l2_u": metrics["rel_l2_u"],
        "mse_u": metrics["mse_u"],
        "max_err_u": metrics["max_err_u"],
        "train_time": train_time,
        "params": params,
        "loss_history": model.loss_history,
        "val_history": trainer.val_history,
        "evaluated_model": os.path.basename(model_path),
    }
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    del model, trainer, best_model
    gc.collect()

    return results


# ---------------------------------------------------------------------------
# Base config shared by all experiments
# ---------------------------------------------------------------------------

def base_config(
    epochs,
    seed=42,
    helmholtz_a1=1.0,
    helmholtz_a2=4.0,
    helmholtz_lambda=1.0,
    output_activation="identity",
    output_scale=1.0,
    hard_bc=False,
    hard_bc_scale=10.0,
    postprocessor_type="mlp",
    spectral_norm_postprocessor=False,
    noise_augmentation_sigma=0.0,
    postprocessor_gain_penalty=0.0,
):
    return {
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
        "noise_strength": 0.0,
        "domain_lo": [-1.0, -1.0],
        "domain_hi": [1.0, 1.0],
        "seed": seed,
        "helmholtz_a1": helmholtz_a1,
        "helmholtz_a2": helmholtz_a2,
        "helmholtz_lambda": helmholtz_lambda,
        "output_activation": output_activation,
        "output_scale": output_scale,
        "hard_bc": hard_bc,
        "hard_bc_scale": hard_bc_scale,
        "postprocessor_type": postprocessor_type,
        "spectral_norm_postprocessor": spectral_norm_postprocessor,
        "noise_augmentation_sigma": noise_augmentation_sigma,
        "postprocessor_gain_penalty": postprocessor_gain_penalty,
    }


# ---------------------------------------------------------------------------
# Optuna HPO
# ---------------------------------------------------------------------------

def run_optuna_hpo(n_trials, optuna_epochs, output_dir, seed=42):
    """Run Optuna HPO and return best params dict."""
    if not HAS_OPTUNA:
        print("Optuna not installed — skipping HPO. pip install optuna>=3.0")
        return None

    hpo_dir = os.path.join(output_dir, "optuna_hpo")
    os.makedirs(hpo_dir, exist_ok=True)

    def objective(trial):
        te_width = trial.suggest_categorical("te_width", [8, 16, 32, 64, 128])
        te_hidden_layers = trial.suggest_int("te_hidden_layers", 1, 3)
        num_qubits = trial.suggest_categorical("num_qubits", [2, 4, 6])
        num_quantum_layers = trial.suggest_categorical("num_quantum_layers", [3, 5, 7])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
        bc_weight = trial.suggest_categorical("bc_weight", [1.0, 5.0, 10.0, 20.0])
        batch_size = trial.suggest_categorical("batch_size", [200, 500, 1000])

        trial_seed = seed + trial.number
        torch.manual_seed(trial_seed)
        np.random.seed(trial_seed)

        config = base_config(optuna_epochs, seed=trial_seed)
        config.update({
            "num_qubits": num_qubits,
            "num_quantum_layers": num_quantum_layers,
            "q_ansatz": "hea",
            "hidden_dim": hidden_dim,
            "mode": "te",
            "lr": lr,
            "batch_size": batch_size,
            "bc_weight": bc_weight,
            "te_hidden_layers": te_hidden_layers,
            "te_width": te_width,
            "print_every": max(optuna_epochs // 5, 1),
        })

        trial_dir = os.path.join(hpo_dir, f"trial_{trial.number:04d}")
        model = QCPINNSolver(config)
        trainer = Trainer(model, config, trial_dir)

        def trial_callback(epoch, loss_val):
            trial.report(loss_val, epoch)
            if trial.should_prune():
                raise TrialPruned()

        try:
            trainer.train(callback=trial_callback)
        except TrialPruned:
            del model, trainer
            gc.collect()
            raise

        metrics = evaluate_helmholtz(model, grid_points=100)
        rel_l2 = metrics["rel_l2_u"]

        del model, trainer
        gc.collect()
        return rel_l2

    print(f"\n{'='*70}")
    print(f"  OPTUNA HPO: {n_trials} trials x {optuna_epochs} epochs")
    print(f"{'='*70}")

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=min(10, n_trials))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)

    study = optuna.create_study(
        study_name="te_qpinn_helmholtz_hpo",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials)

    # Save Optuna outputs
    best_data = {
        "objective": study.best_value,
        "trial_number": study.best_trial.number,
        "params": study.best_params,
    }
    with open(os.path.join(hpo_dir, "best_params.json"), "w") as f:
        json.dump(best_data, f, indent=2)

    try:
        df = study.trials_dataframe()
        df.to_csv(os.path.join(hpo_dir, "trials.csv"), index=False)
    except ImportError:
        pass

    # Visualizations
    try:
        from optuna.visualization.matplotlib import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
            plot_slice,
        )
        for fname, fn in [
            ("optimization_history.png", plot_optimization_history),
            ("param_importances.png", plot_param_importances),
            ("parallel_coordinate.png", plot_parallel_coordinate),
            ("slice_plot.png", plot_slice),
        ]:
            try:
                result = fn(study)
                if hasattr(result, "figure"):
                    fig = result.figure
                elif hasattr(result, "flat"):
                    fig = result.flat[0].figure
                else:
                    fig = plt.gcf()
                fig.savefig(os.path.join(hpo_dir, fname), dpi=150, bbox_inches="tight")
                plt.close(fig)
            except Exception as e:
                print(f"  Warning: {fname}: {e}")
    except ImportError:
        pass

    # Print summary
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n  Optuna: {len(completed)} completed, {len(pruned)} pruned")
    print(f"  Best trial #{study.best_trial.number}: rel_l2={study.best_value:.4f}%")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study.best_params


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def make_comparison_plots(all_results, output_dir):
    """Create side-by-side comparison plots."""

    # --- Loss curves ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    colors = {"Classical PINN": "#2196F3", "Fixed-Angle QPINN (Repeat, 4Q/5L)": "#FF9800",
              "TE-QPINN (4Q/5L)": "#4CAF50", "Optuna TE-QPINN": "#E91E63"}

    for name, r in all_results.items():
        c = colors.get(name, "gray")
        axes[0].semilogy(r["loss_history"], label=name, linewidth=1.5, color=c)
    axes[0].set_xlabel("L-BFGS Epoch", fontsize=12)
    axes[0].set_ylabel("Total Loss", fontsize=12)
    axes[0].set_title("Training Convergence", fontsize=14)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # --- Bar chart: Rel L2 Error ---
    names = list(all_results.keys())
    l2_errors = [all_results[n]["rel_l2_u"] for n in names]
    bar_colors = [colors.get(n, "gray") for n in names]
    bars = axes[1].bar(range(len(names)), l2_errors, color=bar_colors)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axes[1].set_ylabel("Relative L2 Error (%)", fontsize=12)
    axes[1].set_title("Solution Accuracy", fontsize=14)
    for bar, val in zip(bars, l2_errors):
        axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

    # --- Bar chart: Parameter efficiency ---
    param_counts = [all_results[n]["params"]["total"] for n in names]
    efficiency = [l2 / max(p, 1) * 1000 for l2, p in zip(l2_errors, param_counts)]
    bars2 = axes[2].bar(range(len(names)), param_counts, color=bar_colors)
    axes[2].set_xticks(range(len(names)))
    axes[2].set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    axes[2].set_ylabel("Total Parameters", fontsize=12)
    axes[2].set_title("Model Size", fontsize=14)
    for bar, val, l2 in zip(bars2, param_counts, l2_errors):
        axes[2].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{val}\n({l2:.1f}%)', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "comparison.pdf"), dpi=150, bbox_inches="tight")
    plt.close()

    # --- Zoomed loss curves (last 50%) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in all_results.items():
        hist = r["loss_history"]
        mid = len(hist) // 2
        c = colors.get(name, "gray")
        ax.semilogy(range(mid, len(hist)), hist[mid:], label=name, linewidth=1.5, color=c)
    ax.set_xlabel("L-BFGS Epoch", fontsize=12)
    ax.set_ylabel("Total Loss", fontsize=12)
    ax.set_title("Late-Stage Convergence (last 50%)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "convergence_detail.png"), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full 4-way comparison on Helmholtz")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs for comparison runs")
    parser.add_argument("--optuna-trials", type=int, default=20, help="Optuna HPO trials")
    parser.add_argument("--optuna-epochs", type=int, default=100, help="Epochs per Optuna trial")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--helmholtz-a1", type=float, default=1.0,
                        help="Helmholtz exact-solution frequency in x1")
    parser.add_argument("--helmholtz-a2", type=float, default=4.0,
                        help="Helmholtz exact-solution frequency in x2")
    parser.add_argument("--helmholtz-lambda", type=float, default=1.0,
                        help="Lambda term in the forced Helmholtz operator")
    parser.add_argument("--output-activation", type=str, default="identity",
                        choices=["identity", "tanh"],
                        help="Optional bounded final activation for the PDE output")
    parser.add_argument("--output-scale", type=float, default=1.0,
                        help="Scale applied after the final output activation")
    parser.add_argument("--hard-bc", action="store_true",
                        help="Enforce hard zero Dirichlet boundaries for Helmholtz")
    parser.add_argument("--hard-bc-scale", type=float, default=10.0,
                        help="Boundary-envelope steepness for hard Helmholtz boundary enforcement")
    parser.add_argument("--postprocessor-type", type=str, default="mlp",
                        choices=["mlp", "linear"],
                        help="Classical readout type after the quantum layer")
    parser.add_argument("--spectral-norm-postprocessor", action="store_true",
                        help="Apply spectral normalization to the postprocessor layers")
    parser.add_argument("--noise-augmentation-sigma", type=float, default=0.0,
                        help="Add differentiable Gaussian noise after the quantum layer during training")
    parser.add_argument("--postprocessor-gain-penalty", type=float, default=0.0,
                        help="Penalty on the postprocessor gain proxy")
    parser.add_argument("--skip-optuna", action="store_true",
                        help="Skip Optuna HPO (3-way comparison only)")
    parser.add_argument("--output-dir", type=str, default="experiments/full_comparison",
                        help="Output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    optuna_str = "skipped" if args.skip_optuna else f"{args.optuna_trials} trials x {args.optuna_epochs} ep"
    print("=" * 70)
    print("  FULL COMPARISON: Helmholtz Equation")
    print(f"  Epochs: {args.epochs} | Optuna: {optuna_str}")
    print(f"  Batch size: 2000 (classical) / 1000 (quantum)")
    print(f"  Output: {output_dir}")
    print("=" * 70)

    all_results = {}
    t_start = time.time()

    # -----------------------------------------------------------------------
    # 1. Classical PINN (strong baseline, hidden_dim=50, ~2751 params)
    # -----------------------------------------------------------------------
    cfg = base_config(
        args.epochs,
        args.seed,
        args.helmholtz_a1,
        args.helmholtz_a2,
        args.helmholtz_lambda,
        args.output_activation,
        args.output_scale,
        args.hard_bc,
        args.hard_bc_scale,
        args.postprocessor_type,
        args.spectral_norm_postprocessor,
        args.noise_augmentation_sigma,
        args.postprocessor_gain_penalty,
    )
    cfg.update({
        "mode": "classical",
        "num_qubits": 4,
        "num_quantum_layers": 5,
        "q_ansatz": "hea",
        "hidden_dim": 50,
        "batch_size": 2000,  # classical is fast; use dense grid
    })
    all_results["Classical PINN"] = run_experiment(cfg, "classical_pinn", output_dir)

    # -----------------------------------------------------------------------
    # 2. TE-QPINN (4 qubits, 5 layers — paper default)
    # -----------------------------------------------------------------------
    cfg = base_config(
        args.epochs,
        args.seed,
        args.helmholtz_a1,
        args.helmholtz_a2,
        args.helmholtz_lambda,
        args.output_activation,
        args.output_scale,
        args.hard_bc,
        args.hard_bc_scale,
        args.postprocessor_type,
        args.spectral_norm_postprocessor,
        args.noise_augmentation_sigma,
        args.postprocessor_gain_penalty,
    )
    cfg.update({
        "mode": "te",
        "num_qubits": 4,
        "num_quantum_layers": 5,
        "q_ansatz": "hea",
        "hidden_dim": 50,
        "te_hidden_layers": 2,
        "te_width": 10,
    })
    all_results["TE-QPINN (4Q/5L)"] = run_experiment(cfg, "te_qpinn_4q5l", output_dir)

    # -----------------------------------------------------------------------
    # 3. Fixed-angle QPINN (4 qubits, 5 layers)
    #    Uses "repeat" mode: coordinates tiled to fill all qubits, NO classical
    #    preprocessor. This is the fixed-angle / repeated-coordinate quantum
    #    baseline used elsewhere in this repo.
    # -----------------------------------------------------------------------
    cfg = base_config(
        args.epochs,
        args.seed,
        args.helmholtz_a1,
        args.helmholtz_a2,
        args.helmholtz_lambda,
        args.output_activation,
        args.output_scale,
        args.hard_bc,
        args.hard_bc_scale,
        args.postprocessor_type,
        args.spectral_norm_postprocessor,
        args.noise_augmentation_sigma,
        args.postprocessor_gain_penalty,
    )
    cfg.update({
        "mode": "repeat",
        "num_qubits": 4,
        "num_quantum_layers": 5,
        "q_ansatz": "hea",
        "hidden_dim": 50,
    })
    all_results["Fixed-Angle QPINN (Repeat, 4Q/5L)"] = run_experiment(
        cfg, "baseline_qpinn_4q5l", output_dir
    )

    # -----------------------------------------------------------------------
    # 4. Optuna HPO → retrain best at full epochs (optional)
    # -----------------------------------------------------------------------
    if not args.skip_optuna:
        best_params = run_optuna_hpo(args.optuna_trials, args.optuna_epochs, output_dir, args.seed)

        if best_params:
            cfg = base_config(
                args.epochs,
                args.seed,
                args.helmholtz_a1,
                args.helmholtz_a2,
                args.helmholtz_lambda,
                args.output_activation,
                args.output_scale,
                args.hard_bc,
                args.hard_bc_scale,
                args.postprocessor_type,
                args.spectral_norm_postprocessor,
                args.noise_augmentation_sigma,
                args.postprocessor_gain_penalty,
            )
            cfg.update({
                "mode": "te",
                "num_qubits": best_params["num_qubits"],
                "num_quantum_layers": best_params["num_quantum_layers"],
                "q_ansatz": "hea",
                "hidden_dim": best_params["hidden_dim"],
                "lr": best_params["lr"],
                "batch_size": max(best_params["batch_size"], 500),  # enforce minimum
                "bc_weight": best_params["bc_weight"],
                "te_hidden_layers": best_params["te_hidden_layers"],
                "te_width": best_params["te_width"],
            })
            all_results["Optuna TE-QPINN"] = run_experiment(
                cfg, "optuna_te_qpinn", output_dir
            )

    total_time = time.time() - t_start

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("  SUMMARY TABLE")
    print("=" * 90)
    print(f"  {'Method':<28} {'Rel L2 (%)':<12} {'MSE':<12} {'Best Loss':<12} "
          f"{'Params':<8} {'Time (s)':<10}")
    print(f"  {'-'*86}")
    for name, r in all_results.items():
        print(f"  {name:<28} {r['rel_l2_u']:<12.4f} {r['mse_u']:<12.4e} "
              f"{r['best_loss']:<12.4e} {r['params']['total']:<8} {r['train_time']:<10.1f}")
    print(f"  {'-'*86}")
    print(f"  Total wall time: {total_time/3600:.2f} hours")
    print("=" * 90)

    # Save summary
    summary = {}
    for name, r in all_results.items():
        summary[name] = {k: v for k, v in r.items() if k != "loss_history"}
        summary[name]["final_loss"] = r["loss_history"][-1] if r["loss_history"] else None
    summary["total_time_hours"] = total_time / 3600
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Comparison plots
    make_comparison_plots(all_results, output_dir)
    print(f"\n  All results saved to {output_dir}/")


if __name__ == "__main__":
    main()
