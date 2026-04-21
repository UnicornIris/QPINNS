#!/usr/bin/env python3
"""Optuna-based hyperparameter optimization for TE-QPINN on Helmholtz equation.

Produces:
  - best_params.json: best hyperparameters + objective value
  - trials.csv: full study dataframe
  - optimization_history.png, param_importances.png, parallel_coordinate.png, slice_plot.png
  - Optional: retrain best configuration for 300 epochs with full evaluation

Usage:
  # Quick smoke test
  python scripts/optuna_hpo_helmholtz.py --n-trials 3 --epochs 20

  # Full HPO run
  python scripts/optuna_hpo_helmholtz.py --n-trials 50 --epochs 100

  # Retrain best configuration
  python scripts/optuna_hpo_helmholtz.py --retrain-best experiments/optuna_hpo/best_params.json
"""

import os
import sys
import gc
import json
import argparse
import time

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import optuna
    from optuna.exceptions import TrialPruned
except ImportError:
    print("Optuna is required. Install with: pip install 'qcpinn[hpo]' or pip install optuna>=3.0")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcpinn.solver import QCPINNSolver
from qcpinn.trainer import Trainer
from qcpinn.evaluation import evaluate_helmholtz


def create_objective(epochs, base_seed, output_dir):
    """Create an Optuna objective function for TE-QPINN Helmholtz optimization."""

    def objective(trial):
        # Sample hyperparameters
        te_width = trial.suggest_categorical("te_width", [8, 16, 32, 64, 128])
        te_hidden_layers = trial.suggest_int("te_hidden_layers", 1, 3)
        num_qubits = trial.suggest_categorical("num_qubits", [2, 4, 6])
        num_quantum_layers = trial.suggest_categorical("num_quantum_layers", [3, 5, 7])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
        bc_weight = trial.suggest_categorical("bc_weight", [1.0, 5.0, 10.0, 20.0])
        batch_size = trial.suggest_categorical("batch_size", [24, 48, 100, 200])

        # Deterministic seed per trial
        seed = base_seed + trial.number
        torch.manual_seed(seed)
        np.random.seed(seed)

        config = {
            "problem": "helmholtz",
            "input_dim": 2,
            "output_dim": 1,
            "num_qubits": num_qubits,
            "num_quantum_layers": num_quantum_layers,
            "q_ansatz": "hea",
            "hidden_dim": hidden_dim,
            "mode": "te",
            "optimizer": "lbfgs",
            "lr": lr,
            "epochs": epochs,
            "batch_size": batch_size,
            "bc_weight": bc_weight,
            "print_every": max(epochs // 5, 1),
            "fixed_collocation": True,
            "noise_strength": 0.0,
            "domain_lo": [-1.0, -1.0],
            "domain_hi": [1.0, 1.0],
            "te_hidden_layers": te_hidden_layers,
            "te_width": te_width,
        }

        trial_dir = os.path.join(output_dir, f"trial_{trial.number:04d}")

        model = QCPINNSolver(config)
        trainer = Trainer(model, config, trial_dir)

        def trial_callback(epoch, loss_val):
            trial.report(loss_val, epoch)
            if trial.should_prune():
                raise TrialPruned()

        try:
            trainer.train(callback=trial_callback)
        except TrialPruned:
            # Clean up before re-raising
            del model, trainer
            gc.collect()
            raise

        # Evaluate on test grid
        metrics = evaluate_helmholtz(model, grid_points=100)
        rel_l2_u = metrics["rel_l2_u"]

        # Clean up to free memory
        del model, trainer
        gc.collect()

        return rel_l2_u

    return objective


def save_visualizations(study, output_dir):
    """Save Optuna visualization plots."""
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
        plot_parallel_coordinate,
        plot_slice,
    )

    plots = {
        "optimization_history.png": plot_optimization_history,
        "param_importances.png": plot_param_importances,
        "parallel_coordinate.png": plot_parallel_coordinate,
        "slice_plot.png": plot_slice,
    }

    for filename, plot_fn in plots.items():
        try:
            result = plot_fn(study)
            # plot_slice returns an ndarray of Axes; others return a single Axes
            if hasattr(result, "figure"):
                fig = result.figure
            elif hasattr(result, "flat"):
                fig = result.flat[0].figure
            else:
                fig = plt.gcf()
            fig.savefig(
                os.path.join(output_dir, filename), dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
            print(f"  Saved {filename}")
        except Exception as e:
            print(f"  Warning: could not generate {filename}: {e}")


def print_study_summary(study):
    """Print a summary of the study results."""
    print("\n" + "=" * 70)
    print("STUDY SUMMARY")
    print("=" * 70)

    print(f"\nTotal trials: {len(study.trials)}")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"  Completed: {len(completed)}")
    print(f"  Pruned:    {len(pruned)}")
    print(f"  Failed:    {len(failed)}")

    if not completed:
        print("\nNo completed trials.")
        return

    print(f"\nBest trial (#{study.best_trial.number}):")
    print(f"  Objective (rel_l2_u): {study.best_value:.4f}%")
    print(f"  Parameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # Top 5 trials
    sorted_trials = sorted(completed, key=lambda t: t.value)
    print(f"\nTop {min(5, len(sorted_trials))} trials:")
    print(f"  {'#':<6} {'Objective':<12} {'Qubits':<8} {'Layers':<8} {'TE Width':<10} {'LR':<10}")
    print(f"  {'-'*54}")
    for t in sorted_trials[:5]:
        print(
            f"  {t.number:<6} {t.value:<12.4f} "
            f"{t.params.get('num_qubits', '?'):<8} "
            f"{t.params.get('num_quantum_layers', '?'):<8} "
            f"{t.params.get('te_width', '?'):<10} "
            f"{t.params.get('lr', 0):<10.2e}"
        )


def retrain_best(params_path, output_dir):
    """Retrain with the best parameters for 300 epochs with full evaluation."""
    with open(params_path, "r") as f:
        data = json.load(f)

    params = data["params"]
    print("\n" + "=" * 60)
    print("RETRAINING BEST CONFIGURATION (300 epochs)")
    print("=" * 60)
    for k, v in params.items():
        print(f"  {k}: {v}")

    torch.manual_seed(42)
    np.random.seed(42)

    config = {
        "problem": "helmholtz",
        "input_dim": 2,
        "output_dim": 1,
        "num_qubits": params["num_qubits"],
        "num_quantum_layers": params["num_quantum_layers"],
        "q_ansatz": "hea",
        "hidden_dim": params["hidden_dim"],
        "mode": "te",
        "optimizer": "lbfgs",
        "lr": params["lr"],
        "epochs": 300,
        "batch_size": params["batch_size"],
        "bc_weight": params["bc_weight"],
        "print_every": 10,
        "fixed_collocation": True,
        "noise_strength": 0.0,
        "domain_lo": [-1.0, -1.0],
        "domain_hi": [1.0, 1.0],
        "te_hidden_layers": params["te_hidden_layers"],
        "te_width": params["te_width"],
    }

    retrain_dir = os.path.join(output_dir, "retrain_best")
    model = QCPINNSolver(config)
    print(f"Parameters: {model.count_parameters()}")

    trainer = Trainer(model, config, retrain_dir)
    t0 = time.time()
    best_loss = trainer.train()
    train_time = time.time() - t0

    # Load best model for evaluation
    best_model = QCPINNSolver.load_state(os.path.join(retrain_dir, "best_model.pth"))
    metrics = evaluate_helmholtz(best_model, grid_points=100)

    print(f"\n--- Retrain Results ---")
    print(f"  Best loss:     {best_loss:.4e}")
    print(f"  Rel L2 (u):    {metrics['rel_l2_u']:.4f}%")
    print(f"  MSE (u):       {metrics['mse_u']:.4e}")
    print(f"  Max error (u): {metrics['max_err_u']:.4e}")
    print(f"  Train time:    {train_time:.1f}s")

    results = {
        "best_loss": best_loss,
        "metrics": metrics,
        "train_time": train_time,
        "config": config,
    }
    with open(os.path.join(retrain_dir, "retrain_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nRetrain results saved to {retrain_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Optuna HPO for TE-QPINN on Helmholtz equation"
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of HPO trials")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs per trial")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/optuna_hpo",
        help="Output directory",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///hpo.db) for resume capability",
    )
    parser.add_argument(
        "--retrain-best",
        type=str,
        default=None,
        help="Path to best_params.json to retrain with 300 epochs",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Retrain mode
    if args.retrain_best:
        retrain_best(args.retrain_best, args.output_dir)
        return

    print("=" * 60)
    print("OPTUNA HPO: TE-QPINN Helmholtz")
    print(f"  Trials:  {args.n_trials}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Seed:    {args.seed}")
    print(f"  Output:  {args.output_dir}")
    print(f"  Storage: {args.storage or 'in-memory'}")
    print("=" * 60)

    # Set up Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=10)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)

    study = optuna.create_study(
        study_name="te_qpinn_helmholtz_hpo",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    objective = create_objective(args.epochs, args.seed, args.output_dir)
    study.optimize(objective, n_trials=args.n_trials)

    # Save results
    print("\nSaving results...")

    # Best params
    best_data = {
        "objective": study.best_value,
        "trial_number": study.best_trial.number,
        "params": study.best_params,
    }
    with open(os.path.join(args.output_dir, "best_params.json"), "w") as f:
        json.dump(best_data, f, indent=2)
    print(f"  Saved best_params.json")

    # Trials CSV
    try:
        df = study.trials_dataframe()
        df.to_csv(os.path.join(args.output_dir, "trials.csv"), index=False)
        print(f"  Saved trials.csv")
    except ImportError:
        print("  Warning: pandas not installed, skipping trials.csv (pip install pandas)")

    # Visualizations
    save_visualizations(study, args.output_dir)

    # Summary
    print_study_summary(study)

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
