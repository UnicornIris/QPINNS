#!/usr/bin/env python3
"""Fine-tune or retrain a QCPINN checkpoint with architecture overrides."""

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

from qcpinn.evaluation import EVALUATORS, plot_loss_history, plot_results
from qcpinn.solver import QCPINNSolver
from qcpinn.trainer import Trainer


def _get_post_layers(module: torch.nn.Module):
    if isinstance(module, torch.nn.Linear):
        return [module]
    if isinstance(module, torch.nn.Sequential):
        return [m for m in module if isinstance(m, torch.nn.Linear)]
    return []


def _assign_linear(dst: torch.nn.Linear, src: torch.nn.Linear):
    with torch.no_grad():
        if hasattr(dst, "parametrizations") and hasattr(dst.parametrizations, "weight"):
            dst.parametrizations.weight.original.copy_(src.weight)
        else:
            dst.weight.copy_(src.weight)
        if dst.bias is not None and src.bias is not None and dst.bias.shape == src.bias.shape:
            dst.bias.copy_(src.bias)


def _copy_matching_weights(src_model: QCPINNSolver, dst_model: QCPINNSolver):
    src_state = src_model.state_dict()
    dst_state = dst_model.state_dict()
    for key, value in src_state.items():
        if key in dst_state and dst_state[key].shape == value.shape:
            dst_state[key] = value.clone()
    dst_model.load_state_dict(dst_state, strict=False)

    src_post = _get_post_layers(src_model.postprocessor)
    dst_post = _get_post_layers(dst_model.postprocessor)
    for src_layer, dst_layer in zip(src_post, dst_post):
        if src_layer.weight.shape == dst_layer.weight.shape:
            _assign_linear(dst_layer, src_layer)


def _set_train_scope(model: QCPINNSolver, scope: str):
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


def _load_weights_only(path: str, device: torch.device) -> QCPINNSolver:
    state = torch.load(path, map_location=device)
    model = QCPINNSolver(state["config"], device=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.loss_history = state.get("loss_history", [])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Fine-tune or retrain a checkpoint with config overrides")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scope", choices=["full", "quantum_post", "post_only"], default="full")
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimizer", choices=["adam", "lbfgs"], default="lbfgs")
    parser.add_argument("--fixed-collocation", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-every", type=int, default=25)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--postprocessor-type", choices=["mlp", "linear"], default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--spectral-norm-postprocessor", action="store_true")
    parser.add_argument("--noise-augmentation-sigma", type=float, default=None)
    parser.add_argument("--output-activation", choices=["identity", "tanh"], default=None)
    parser.add_argument("--output-scale", type=float, default=None)
    parser.add_argument("--hard-bc", action="store_true")
    parser.add_argument("--hard-bc-scale", type=float, default=None)
    parser.add_argument("--postprocessor-gain-penalty", type=float, default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = QCPINNSolver.load_state(args.checkpoint, device=device)
    base_model.eval()

    cfg = dict(base_model.config)
    cfg["epochs"] = args.epochs
    cfg["lr"] = args.lr
    cfg["batch_size"] = args.batch_size
    cfg["optimizer"] = args.optimizer
    cfg["fixed_collocation"] = args.fixed_collocation or args.optimizer == "lbfgs"
    cfg["seed"] = args.seed
    cfg["val_every"] = args.val_every
    cfg["print_every"] = args.print_every
    if args.postprocessor_type is not None:
        cfg["postprocessor_type"] = args.postprocessor_type
    if args.hidden_dim is not None:
        cfg["hidden_dim"] = args.hidden_dim
    if args.spectral_norm_postprocessor:
        cfg["spectral_norm_postprocessor"] = True
    if args.noise_augmentation_sigma is not None:
        cfg["noise_augmentation_sigma"] = args.noise_augmentation_sigma
    if args.output_activation is not None:
        cfg["output_activation"] = args.output_activation
    if args.output_scale is not None:
        cfg["output_scale"] = args.output_scale
    if args.hard_bc:
        cfg["hard_bc"] = True
    if args.hard_bc_scale is not None:
        cfg["hard_bc_scale"] = args.hard_bc_scale
    if args.postprocessor_gain_penalty is not None:
        cfg["postprocessor_gain_penalty"] = args.postprocessor_gain_penalty

    model = QCPINNSolver(cfg, device=device)
    if not args.no_warm_start:
        _copy_matching_weights(base_model, model)
    _set_train_scope(model, args.scope)
    model.eval()

    trainer = Trainer(model, cfg, str(out_dir), device=device)
    t0 = time.time()
    best_loss = trainer.train()
    train_time = time.time() - t0

    best_val_path = out_dir / "best_val_model.pth"
    best_path = out_dir / "best_model.pth"
    chosen = best_val_path if best_val_path.exists() else best_path
    best_model = _load_weights_only(str(chosen), device=device)

    metrics = {}
    evaluator = EVALUATORS.get(cfg["problem"])
    if evaluator is not None:
        metrics = evaluator(best_model, grid_points=100, device=device)
        plot_results(metrics, str(out_dir), title_prefix=f"{cfg['mode'].upper()} ")

    plot_loss_history(best_model.loss_history, str(out_dir), title=out_dir.name)

    summary = {
        "base_checkpoint": args.checkpoint,
        "warm_started": not args.no_warm_start,
        "train_scope": args.scope,
        "selected_checkpoint": str(chosen),
        "best_loss": float(best_loss),
        "train_time_sec": train_time,
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in metrics.items() if k != "u_exact" and k != "u_pred"},
        "config": cfg,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("=" * 72)
    print(f"Intervention training complete: {out_dir}")
    print(f"Warm start: {summary['warm_started']} | scope={args.scope}")
    if "rel_l2_u" in metrics:
        print(f"Clean rel_l2_u: {metrics['rel_l2_u']:.4f}%")
    print(f"Best loss: {best_loss:.3e} | train_time={train_time:.1f}s")
    print("=" * 72)

    del base_model, model, trainer, best_model
    gc.collect()


if __name__ == "__main__":
    main()
