#!/usr/bin/env python3
"""Warm-start noise rescue experiments for QCPINN.

This script is designed to answer a specific research question honestly:
can a strong clean checkpoint be adapted to noisy execution without
destroying its underlying PDE solution?

Protocol:
  1. Load a clean checkpoint
  2. Rebuild the model at a target train-time noise level
  3. Copy compatible weights from the clean checkpoint
  4. Fine-tune with resampled collocation points under noisy simulation
  5. Evaluate before/after across multiple test-time noise levels

Typical usage:
    python scripts/noise_rescue.py \
        --checkpoint experiments/full_comparison_fixed/te_qpinn_4q5l/best_val_model.pth \
        --train-noise 0.01 --epochs 40 --lr 1e-4 --batch-size 16 \
        --test-noise 0.0 0.001 0.005 0.01 0.02 0.05 \
        --output-dir experiments/noise_rescue/te_ft_p001
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcpinn.solver import QCPINNSolver
from qcpinn.trainer import Trainer
from qcpinn.evaluation import plot_loss_history
from scripts.noise_robustness import evaluate_at_noise


def _copy_matching_state(src_model: QCPINNSolver, dst_model: QCPINNSolver) -> None:
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    for key in src_state:
        if key in dst_state and src_state[key].shape == dst_state[key].shape:
            dst_state[key] = src_state[key].clone()
    dst_model.load_state_dict(dst_state)


def _load_weights_only(path: str, device: torch.device) -> QCPINNSolver:
    state = torch.load(path, map_location=device)
    model = QCPINNSolver(state["config"], device=device)
    model.load_state_dict(state["model_state_dict"])
    model.loss_history = state.get("loss_history", [])
    model.eval()
    return model


def _set_train_scope(model: QCPINNSolver, scope: str) -> None:
    for p in model.parameters():
        p.requires_grad = False

    if scope == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif scope == "quantum_post":
        for name, p in model.named_parameters():
            if name.startswith("quantum_layer.") or name.startswith("postprocessor."):
                p.requires_grad = True
    elif scope == "post_only":
        for name, p in model.named_parameters():
            if name.startswith("postprocessor."):
                p.requires_grad = True
    else:
        raise ValueError(f"Unknown train scope: {scope}")

    model._build_optimizer()


def _evaluate_grid(model: QCPINNSolver, test_noise_levels, device, grid_points):
    out = {}
    for noise in test_noise_levels:
        metrics = evaluate_at_noise(model, noise, grid_points=grid_points, device=device)
        out[str(noise)] = {
            "rel_l2_u": float(metrics["rel_l2_u"]),
            "mse_u": float(metrics["mse_u"]),
            "max_err_u": float(metrics["max_err_u"]),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Warm-start noisy fine-tuning rescue experiments")
    parser.add_argument("--checkpoint", required=True, help="Clean checkpoint to warm-start from")
    parser.add_argument("--train-noise", type=float, required=True, help="Noise strength during fine-tuning")
    parser.add_argument("--test-noise", nargs="+", type=float,
                        default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05],
                        help="Noise strengths used for evaluation")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--optimizer", choices=["adam", "lbfgs"], default="adam")
    parser.add_argument("--scope", choices=["full", "quantum_post", "post_only"], default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-points", type=int, default=20)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_model = QCPINNSolver.load_state(args.checkpoint, device=device)
    clean_model.eval()

    cfg = dict(clean_model.config)
    cfg["noise_strength"] = args.train_noise
    cfg["optimizer"] = args.optimizer
    cfg["lr"] = args.lr
    cfg["epochs"] = args.epochs
    cfg["batch_size"] = args.batch_size
    cfg["fixed_collocation"] = args.optimizer == "lbfgs"
    cfg["print_every"] = max(args.epochs // 10, 1)
    cfg["val_every"] = max(args.epochs // 4, 1)
    cfg["seed"] = args.seed

    ft_model = QCPINNSolver(cfg, device=device)
    _copy_matching_state(clean_model, ft_model)
    _set_train_scope(ft_model, args.scope)
    ft_model.eval()

    before_metrics = _evaluate_grid(ft_model, args.test_noise, device, args.grid_points)

    trainer = Trainer(ft_model, cfg, str(out_dir), device=device)
    t0 = time.time()
    best_loss = trainer.train()
    train_time = time.time() - t0

    best_val_path = out_dir / "best_val_model.pth"
    best_path = out_dir / "best_model.pth"
    chosen = best_val_path if best_val_path.exists() else best_path
    best_model = _load_weights_only(str(chosen), device=device)
    after_metrics = _evaluate_grid(best_model, args.test_noise, device, args.grid_points)

    plot_loss_history(best_model.loss_history, str(out_dir), title=out_dir.name)

    summary = {
        "checkpoint": args.checkpoint,
        "mode": clean_model.config.get("mode"),
        "train_scope": args.scope,
        "train_noise": args.train_noise,
        "test_noise_levels": args.test_noise,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "best_loss": float(best_loss),
        "train_time_sec": train_time,
        "selected_checkpoint": str(chosen),
        "before": before_metrics,
        "after": after_metrics,
    }

    with open(out_dir / "rescue_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"Rescue experiment complete: {out_dir}")
    print(f"Mode: {summary['mode']} | scope={args.scope} | train_noise={args.train_noise}")
    print(f"Best loss: {best_loss:.3e} | train_time={train_time:.1f}s")
    print("-" * 70)
    print(f"{'test_noise':<12} {'before_L2%':>12} {'after_L2%':>12}")
    for noise in args.test_noise:
        key = str(noise)
        print(f"{noise:<12.3f} {before_metrics[key]['rel_l2_u']:>12.2f} {after_metrics[key]['rel_l2_u']:>12.2f}")
    print("=" * 70)

    del clean_model, ft_model, trainer, best_model
    gc.collect()


if __name__ == "__main__":
    main()
