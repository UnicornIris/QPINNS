#!/usr/bin/env python3
"""Unified experiment runner for QCPINN research.

Runs controlled experiments comparing baseline vs TE embeddings
across noise levels and PDE types.

Usage:
    # Single experiment
    python -m qcpinn.run --problem helmholtz --mode te --epochs 5000
    
    # Full comparison suite
    python -m qcpinn.run --suite helmholtz_comparison
    
    # Noise study
    python -m qcpinn.run --suite noise_study --problem helmholtz
"""

import argparse
import os
import json
import time
import torch
import sys
from datetime import datetime

from qcpinn.solver import QCPINNSolver
from qcpinn.trainer import Trainer
from qcpinn.evaluation import (
    EVALUATORS, plot_results, plot_loss_history, plot_comparison
)
from qcpinn.datasets import DATASET_REGISTRY


def build_config(args) -> dict:
    """Build experiment config from CLI args."""
    ds_info = DATASET_REGISTRY.get(args.problem)
    if ds_info is None:
        raise ValueError(f"Unknown problem: {args.problem}. Available: {list(DATASET_REGISTRY.keys())}")
    _, in_dim, out_dim, dom_lo, dom_hi = ds_info

    return {
        "problem": args.problem,
        "mode": args.mode,
        "input_dim": in_dim,
        "output_dim": out_dim,
        "num_qubits": args.num_qubits,
        "num_quantum_layers": args.layers,
        "hidden_dim": args.hidden_dim,
        "q_ansatz": args.ansatz,
        "encoding": "angle",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "bc_weight": args.bc_weight,
        "grad_clip": args.grad_clip,
        "print_every": args.print_every,
        "noise_strength": args.noise,
        "shots": args.shots if args.shots > 0 else None,
        "qml_device": "default.qubit",
        "domain_lo": dom_lo,
        "domain_hi": dom_hi,
        "helmholtz_a1": args.helmholtz_a1,
        "helmholtz_a2": args.helmholtz_a2,
        "helmholtz_lambda": args.helmholtz_lambda,
        "te_hidden_layers": args.te_layers,
        "te_width": args.te_width,
        "output_activation": args.output_activation,
        "output_scale": args.output_scale,
        "hard_bc": args.hard_bc,
        "hard_bc_scale": args.hard_bc_scale,
        "postprocessor_type": args.postprocessor_type,
        "spectral_norm_postprocessor": args.spectral_norm_postprocessor,
        "noise_augmentation_sigma": args.noise_augmentation_sigma,
        "postprocessor_gain_penalty": args.postprocessor_gain_penalty,
    }


def run_single_experiment(config: dict, log_dir: str, device=None):
    """Run a single training experiment and return metrics."""
    device = device or torch.device("cpu")
    model = QCPINNSolver(config, device=device)
    trainer = Trainer(model, config, log_dir, device=device)

    t0 = time.time()
    best_loss = trainer.train()
    train_time = time.time() - t0

    # Evaluate the best saved checkpoint, not the last in-memory weights.
    best_val_ckpt = os.path.join(log_dir, "best_val_model.pth")
    best_train_ckpt = os.path.join(log_dir, "best_model.pth")
    best_ckpt = best_val_ckpt if os.path.exists(best_val_ckpt) else best_train_ckpt
    if os.path.exists(best_ckpt):
        model = QCPINNSolver.load_state(best_ckpt, device=device)

    # Evaluate
    metrics = {}
    evaluator = EVALUATORS.get(config["problem"])
    if evaluator is not None:
        metrics = evaluator(model, grid_points=100, device=device)
        plot_results(metrics, log_dir, title_prefix=f"{config['mode'].upper()} ")

    plot_loss_history(model.loss_history, log_dir,
                      title=f"{config['problem']} | {config['mode']} | noise={config.get('noise_strength', 0)}")

    metrics["best_loss"] = best_loss
    metrics["train_time_sec"] = train_time
    metrics["total_params"] = model.count_parameters()["total"]
    metrics["loss_history"] = model.loss_history
    metrics["config"] = config

    # Save metrics
    save_metrics = {k: v for k, v in metrics.items()
                    if not isinstance(v, (torch.Tensor, type(None)))}
    # Convert numpy arrays for JSON
    for k in list(save_metrics.keys()):
        v = save_metrics[k]
        if hasattr(v, 'tolist'):
            save_metrics[k] = v.tolist() if v.size < 100 else f"array({v.shape})"

    with open(os.path.join(log_dir, "metrics.json"), "w") as f:
        json.dump(save_metrics, f, indent=2, default=str)

    return metrics


# ---------------------------------------------------------------------------
# Predefined experiment suites
# ---------------------------------------------------------------------------

def suite_helmholtz_comparison(base_dir, device, args):
    """Compare baseline vs TE on Helmholtz at different qubit counts."""
    results = {}

    # Classical baseline does not depend on qubit count; run once.
    print(f"\n{'='*60}\nRunning: classical\n{'='*60}")
    config = build_config(args)
    config["mode"] = "classical"
    log_dir = os.path.join(base_dir, "classical")
    results["classical"] = run_single_experiment(config, log_dir, device)

    # Quantum baselines across qubit counts.
    for n_qubits in [2, 4]:
        for mode in ["baseline", "te"]:
            name = f"{mode}_q{n_qubits}"
            print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
            config = build_config(args)
            config["mode"] = mode
            config["num_qubits"] = n_qubits
            log_dir = os.path.join(base_dir, name)
            results[name] = run_single_experiment(config, log_dir, device)

    plot_comparison(results, base_dir)
    _print_summary(results)
    return results


def suite_noise_study(base_dir, device, args):
    """Core experiment: noise-aware vs noise-unaware training.
    
    For each noise level:
      - baseline_unaware: baseline trained at noise=0, tested at noise=p
      - baseline_aware:   baseline trained at noise=p, tested at noise=p
      - te_unaware:       TE trained at noise=0, tested at noise=p
      - te_aware:         TE trained at noise=p, tested at noise=p
    """
    results = {}
    noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]

    # First train noiseless models
    for mode in ["baseline", "te"]:
        name = f"{mode}_noiseless"
        print(f"\n{'='*60}\nTraining noiseless: {name}\n{'='*60}")
        config = build_config(args)
        config["mode"] = mode
        config["noise_strength"] = 0.0
        log_dir = os.path.join(base_dir, name)
        results[name] = run_single_experiment(config, log_dir, device)

    # For each noise level, train noise-aware models and test all
    for noise in noise_levels:
        if noise == 0.0:
            continue
        for mode in ["baseline", "te"]:
            name = f"{mode}_aware_p{noise:.3f}"
            print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
            config = build_config(args)
            config["mode"] = mode
            config["noise_strength"] = noise
            log_dir = os.path.join(base_dir, name)
            results[name] = run_single_experiment(config, log_dir, device)

    plot_comparison(results, base_dir)
    _print_summary(results)
    return results


def suite_shot_noise(base_dir, device, args):
    """Study effect of finite shots on baseline vs TE."""
    results = {}
    shot_counts = [None, 1024, 256, 64]

    for shots in shot_counts:
        for mode in ["baseline", "te"]:
            name = f"{mode}_shots{'inf' if shots is None else shots}"
            print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
            config = build_config(args)
            config["mode"] = mode
            config["shots"] = shots
            if shots is not None:
                config["qml_device"] = "default.qubit"  # PennyLane handles finite shots
            log_dir = os.path.join(base_dir, name)
            results[name] = run_single_experiment(config, log_dir, device)

    plot_comparison(results, base_dir)
    _print_summary(results)
    return results


def suite_multi_pde(base_dir, device, args):
    """Test TE across multiple PDE classes."""
    results = {}
    pdes = ["helmholtz", "wave", "klein_gordon", "heat_1d"]

    for pde in pdes:
        for mode in ["baseline", "te"]:
            name = f"{pde}_{mode}"
            print(f"\n{'='*60}\nRunning: {name}\n{'='*60}")
            config = build_config(args)
            config["problem"] = pde
            config["mode"] = mode
            ds_info = DATASET_REGISTRY[pde]
            _, in_dim, out_dim, dom_lo, dom_hi = ds_info
            config["input_dim"] = in_dim
            config["output_dim"] = out_dim
            config["domain_lo"] = dom_lo
            config["domain_hi"] = dom_hi
            log_dir = os.path.join(base_dir, name)
            results[name] = run_single_experiment(config, log_dir, device)

    plot_comparison(results, base_dir)
    _print_summary(results)
    return results


SUITES = {
    "helmholtz_comparison": suite_helmholtz_comparison,
    "noise_study": suite_noise_study,
    "shot_noise": suite_shot_noise,
    "multi_pde": suite_multi_pde,
}


def _print_summary(results):
    print("\n" + "=" * 80)
    print(f"{'Experiment':<30} {'Params':>8} {'Best Loss':>12} {'L2 Err%':>10} {'Time(s)':>10}")
    print("-" * 80)
    for name, r in results.items():
        params = r.get("total_params", "?")
        loss = r.get("best_loss", float("nan"))
        l2 = r.get("rel_l2_u", float("nan"))
        t = r.get("train_time_sec", 0)
        print(f"{name:<30} {params:>8} {loss:>12.3e} {l2:>9.2f}% {t:>10.1f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="QCPINN Experiment Runner")

    # Experiment selection
    parser.add_argument("--suite", type=str, default=None,
                        choices=list(SUITES.keys()),
                        help="Run a predefined experiment suite")
    parser.add_argument("--problem", type=str, default="helmholtz",
                        choices=list(DATASET_REGISTRY.keys()))
    parser.add_argument("--mode", type=str, default="te",
                        choices=["baseline", "te", "direct", "repeat", "classical"])

    # Architecture
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=50)
    parser.add_argument("--ansatz", type=str, default="hea",
                        choices=["hea", "layered_circuit", "alternating_tdcnot", "sim_circ_15"])
    parser.add_argument("--te-layers", type=int, default=2, help="TE hidden layers")
    parser.add_argument("--te-width", type=int, default=32, help="TE hidden width")
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
                        help="Enforce hard zero Dirichlet boundaries (currently Helmholtz only)")
    parser.add_argument("--hard-bc-scale", type=float, default=10.0,
                        help="Boundary-envelope steepness for hard Helmholtz boundary enforcement")
    parser.add_argument("--postprocessor-gain-penalty", type=float, default=0.0,
                        help="Penalty on the postprocessor gain proxy to reduce hardware fragility")
    parser.add_argument("--postprocessor-type", type=str, default="mlp",
                        choices=["mlp", "linear"],
                        help="Classical readout type after the quantum layer")
    parser.add_argument("--spectral-norm-postprocessor", action="store_true",
                        help="Apply spectral normalization to the postprocessor layers")
    parser.add_argument("--noise-augmentation-sigma", type=float, default=0.0,
                        help="Add differentiable Gaussian noise after the quantum layer during training")

    # Training
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lbfgs"])
    parser.add_argument("--bc-weight", type=float, default=10.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--print-every", type=int, default=100)

    # Noise
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Depolarizing noise strength (0=noiseless)")
    parser.add_argument("--shots", type=int, default=0,
                        help="Number of shots (0=analytic)")

    # Output
    parser.add_argument("--output-dir", type=str, default="./experiments")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.suite:
        base_dir = os.path.join(args.output_dir, f"{args.suite}_{timestamp}")
        os.makedirs(base_dir, exist_ok=True)
        print(f"Running suite: {args.suite}")
        print(f"Output: {base_dir}")
        SUITES[args.suite](base_dir, device, args)
    else:
        name = f"{args.problem}_{args.mode}_q{args.num_qubits}_n{args.noise}"
        log_dir = os.path.join(args.output_dir, f"{name}_{timestamp}")
        config = build_config(args)
        run_single_experiment(config, log_dir, device)


if __name__ == "__main__":
    main()
