#!/usr/bin/env python3
"""Diagnose postprocessor gain on a trained checkpoint."""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcpinn.datasets import DATASET_REGISTRY
from qcpinn.hardware import _build_eval_grid
from qcpinn.hardware import _load_model_for_inference
from qcpinn.solver import QCPINNSolver


def _spectral_norms(model: QCPINNSolver):
    modules = []
    if isinstance(model.postprocessor, torch.nn.Sequential):
        modules = [m for m in model.postprocessor if isinstance(m, torch.nn.Linear)]
    elif isinstance(model.postprocessor, torch.nn.Linear):
        modules = [model.postprocessor]

    sigmas = []
    for idx, layer in enumerate(modules):
        weight = layer.weight.detach()
        sigma = torch.linalg.matrix_norm(weight, ord=2).item()
        sigmas.append({"layer_index": idx, "shape": list(weight.shape), "sigma": sigma})

    total = 1.0
    for entry in sigmas:
        total *= entry["sigma"]
    return sigmas, total


def _local_gain_report(model: QCPINNSolver, X_star: torch.Tensor):
    entries = []
    for idx in range(X_star.shape[0]):
        x = X_star[idx : idx + 1]
        with torch.no_grad():
            q0 = model.extract_quantum_features(x)
        q = q0.detach().clone().requires_grad_(True)
        y_post = model.postprocessor(q)
        grad_post = torch.autograd.grad(y_post[0, 0], q, retain_graph=False)[0]
        gain_post = torch.linalg.vector_norm(grad_post).item()

        q2 = q0.detach().clone().requires_grad_(True)
        y_final = model.readout_quantum_features(x, q2)
        grad_final = torch.autograd.grad(y_final[0, 0], q2, retain_graph=False)[0]
        gain_final = torch.linalg.vector_norm(grad_final).item()

        entries.append({
            "point_index": idx,
            "x": x.detach().cpu().tolist()[0],
            "q_out": q0.detach().cpu().tolist()[0],
            "local_gain_postprocessor": gain_post,
            "local_gain_final_output": gain_final,
        })
    return entries


def main():
    parser = argparse.ArgumentParser(description="Analyze postprocessor spectral norms and local gains")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--grid", type=int, default=4)
    parser.add_argument("--grid-scheme", choices=["endpoints", "interior"], default="interior")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    model = _load_model_for_inference(args.checkpoint, device=device)
    model.eval()

    _, _, _, dom_lo, dom_hi = DATASET_REGISTRY[model.config["problem"]]
    X_star = _build_eval_grid(dom_lo, dom_hi, args.grid, args.grid_scheme, device)

    sigmas, total_sigma = _spectral_norms(model)
    local = _local_gain_report(model, X_star)

    worst_post = max(local, key=lambda x: x["local_gain_postprocessor"])
    worst_final = max(local, key=lambda x: x["local_gain_final_output"])
    summary = {
        "checkpoint": args.checkpoint,
        "config": model.config,
        "grid": args.grid,
        "grid_scheme": args.grid_scheme,
        "spectral_norms": sigmas,
        "spectral_norm_product_upper_bound": total_sigma,
        "max_local_gain_postprocessor": worst_post["local_gain_postprocessor"],
        "max_local_gain_final_output": worst_final["local_gain_final_output"],
        "worst_postprocessor_point": worst_post,
        "worst_final_output_point": worst_final,
        "points": local,
    }

    output_json = args.output_json
    if output_json is None:
        output_json = Path(args.checkpoint).with_name("postprocessor_gain_analysis.json")
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("Postprocessor Gain Analysis")
    print("=" * 72)
    print(f"Checkpoint: {args.checkpoint}")
    for entry in sigmas:
        print(f"Layer {entry['layer_index']}: sigma={entry['sigma']:.4f} shape={tuple(entry['shape'])}")
    print(f"Spectral-norm product upper bound: {total_sigma:.4f}")
    print(f"Max local postprocessor gain: {summary['max_local_gain_postprocessor']:.4f}")
    print(f"Max local final-output gain:  {summary['max_local_gain_final_output']:.4f}")
    print(f"Saved: {output_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
