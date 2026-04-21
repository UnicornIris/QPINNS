#!/usr/bin/env python3
"""Explain sparse hardware evaluations against the dense Helmholtz solution.

This script bridges the gap between:
  - dense 100x100 "solution_comparison.pdf" plots from clean simulator evaluation
  - tiny 4x4 / 8x8 hardware sampling grids used for real QPU inference

It produces:
  1. a plot showing the dense exact solution plus the sparse sampled points
  2. a summary JSON with raw and projected diagnostic metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcpinn.datasets import helmholtz_exact_u


def _rebuild_grid(dom_lo, dom_hi, grid_points, grid_scheme):
    axes = []
    for lo, hi in zip(dom_lo, dom_hi):
        if grid_scheme == "interior":
            axis = np.linspace(lo, hi, grid_points + 2)[1:-1]
        else:
            axis = np.linspace(lo, hi, grid_points)
        axes.append(axis)
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.reshape(-1) for m in mesh], axis=1)


def _helmholtz_boundary_factor(X, dom_lo, dom_hi):
    span = np.maximum(np.asarray(dom_hi) - np.asarray(dom_lo), 1e-8)
    factor = np.ones((X.shape[0],), dtype=float)
    for dim in range(X.shape[1]):
        coord = X[:, dim]
        factor *= 4.0 * (coord - dom_lo[dim]) * (dom_hi[dim] - coord) / (span[dim] ** 2)
    return factor


def main():
    parser = argparse.ArgumentParser(description="Create a sparse-grid hardware diagnostic report")
    parser.add_argument("--results", required=True, help="Path to hardware_results.json")
    parser.add_argument("--output-dir", default=None, help="Where to write the report (defaults to results dir)")
    args = parser.parse_args()

    results_path = Path(args.results)
    with open(results_path) as f:
        results = json.load(f)

    out_dir = Path(args.output_dir) if args.output_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_points = int(results["grid_points"])
    grid_scheme = results.get("grid_scheme", "endpoints")
    helmholtz_a1 = float(results.get("helmholtz_a1", 1.0))
    helmholtz_a2 = float(results.get("helmholtz_a2", 4.0))
    X_sparse_raw = results.get("X_star")
    dom_lo = np.array([-1.0, -1.0])
    dom_hi = np.array([1.0, 1.0])
    if X_sparse_raw is None:
        X_sparse = _rebuild_grid(dom_lo, dom_hi, grid_points, grid_scheme)
    else:
        X_sparse = np.asarray(X_sparse_raw, dtype=float)

    u_sim = np.asarray(results["u_sim"]).reshape(-1)
    u_hw = np.asarray(results["u_hw"]).reshape(-1)
    if "u_exact" in results:
        u_exact_sparse = np.asarray(results["u_exact"]).reshape(-1)
    else:
        X_t = torch.tensor(X_sparse, dtype=torch.float64)
        u_exact_sparse = helmholtz_exact_u(X_t, a1=helmholtz_a1, a2=helmholtz_a2).numpy().reshape(-1)

    boundary_factor = _helmholtz_boundary_factor(X_sparse, dom_lo, dom_hi)
    u_hw_clip = np.clip(u_hw, -1.0, 1.0)
    u_hw_bc = u_hw * boundary_factor
    u_hw_clip_bc = u_hw_clip * boundary_factor

    exact_norm = np.linalg.norm(u_exact_sparse)
    diagnostics = {
        "rel_l2_hw_vs_exact_raw": float(np.linalg.norm(u_hw - u_exact_sparse) / exact_norm * 100),
        "rel_l2_hw_vs_exact_clip": float(np.linalg.norm(u_hw_clip - u_exact_sparse) / exact_norm * 100),
        "rel_l2_hw_vs_exact_bc": float(np.linalg.norm(u_hw_bc - u_exact_sparse) / exact_norm * 100),
        "rel_l2_hw_vs_exact_clip_bc": float(np.linalg.norm(u_hw_clip_bc - u_exact_sparse) / exact_norm * 100),
    }

    dense_axis = np.linspace(dom_lo[0], dom_hi[0], 100)
    T, X = np.meshgrid(dense_axis, dense_axis, indexing="ij")
    X_dense = np.stack([T.reshape(-1), X.reshape(-1)], axis=1)
    u_exact_dense = helmholtz_exact_u(
        torch.tensor(X_dense, dtype=torch.float64), a1=helmholtz_a1, a2=helmholtz_a2
    ).numpy().reshape(100, 100)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    im = axes[0].imshow(
        u_exact_dense,
        cmap="RdBu_r",
        extent=[dom_lo[1], dom_hi[1], dom_lo[0], dom_hi[0]],
        origin="lower",
        aspect="auto",
    )
    axes[0].scatter(X_sparse[:, 1], X_sparse[:, 0], c="k", s=20)
    axes[0].set_title("Dense Exact + Sampled Grid")
    axes[0].set_xlabel("x2")
    axes[0].set_ylabel("x1")
    plt.colorbar(im, ax=axes[0])

    for ax, values, title in [
        (axes[1], u_exact_sparse, "Sparse Exact"),
        (axes[2], u_sim, "Sparse Simulator"),
        (axes[3], u_hw, "Sparse Hardware"),
    ]:
        sc = ax.scatter(
            X_sparse[:, 1],
            X_sparse[:, 0],
            c=values,
            cmap="RdBu_r",
            s=120,
            edgecolors="k",
        )
        ax.set_xlim(dom_lo[1], dom_hi[1])
        ax.set_ylim(dom_lo[0], dom_hi[0])
        ax.set_title(title)
        ax.set_xlabel("x2")
        ax.set_ylabel("x1")
        plt.colorbar(sc, ax=ax)

    plt.tight_layout()
    plt.savefig(out_dir / "hardware_grid_report.pdf", dpi=160, bbox_inches="tight")
    plt.close()

    report = {
        "results_file": str(results_path),
        "grid_points": grid_points,
        "grid_scheme": grid_scheme,
        "helmholtz_a1": helmholtz_a1,
        "helmholtz_a2": helmholtz_a2,
        "diagnostics": diagnostics,
        "sample_points": X_sparse.tolist(),
        "u_exact_sparse": u_exact_sparse.tolist(),
        "u_sim_sparse": u_sim.tolist(),
        "u_hw_sparse": u_hw.tolist(),
        "u_hw_clip_sparse": u_hw_clip.tolist(),
        "u_hw_bc_sparse": u_hw_bc.tolist(),
        "u_hw_clip_bc_sparse": u_hw_clip_bc.tolist(),
    }
    with open(out_dir / "hardware_grid_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved {out_dir / 'hardware_grid_report.pdf'}")
    print(f"Saved {out_dir / 'hardware_grid_report.json'}")
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
