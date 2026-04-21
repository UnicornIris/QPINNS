"""Evaluation and visualization utilities."""

import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Optional

from qcpinn.pde import helmholtz_operator, wave_operator, klein_gordon_operator, heat_1d_operator
from qcpinn.datasets import (
    helmholtz_exact_u,
    helmholtz_exact_f,
    wave_exact_u,
    kg_exact_u,
    heat_1d_exact_u,
    get_helmholtz_params,
)


def evaluate_helmholtz(model, grid_points=100, device="cpu"):
    """Evaluate Helmholtz model on a regular grid. Returns metrics dict."""
    a1, a2, lam = get_helmholtz_params(getattr(model, "config", {}))
    t = torch.linspace(-1, 1, grid_points, dtype=torch.float64, device=device)
    x = torch.linspace(-1, 1, grid_points, dtype=torch.float64, device=device)
    T, X = torch.meshgrid(t, x, indexing="ij")
    X_star = torch.stack([T.flatten(), X.flatten()], dim=1)

    with torch.no_grad():
        u_star = helmholtz_exact_u(X_star, a1=a1, a2=a2)
        f_star = helmholtz_exact_f(X_star, a1=a1, a2=a2, lam=lam)

    u_pred, f_pred = helmholtz_operator(model, X_star[:, 0:1], X_star[:, 1:2], lam=lam)
    u_pred = u_pred.detach()
    f_pred = f_pred.detach()

    err_u = torch.norm(u_pred - u_star) / torch.norm(u_star) * 100
    err_f = torch.norm(f_pred - f_star) / (torch.norm(f_star) + 1e-10) * 100
    mse_u = torch.mean((u_pred - u_star) ** 2)
    max_err_u = torch.max(torch.abs(u_pred - u_star))

    return {
        "rel_l2_u": err_u.item(),
        "rel_l2_f": err_f.item(),
        "mse_u": mse_u.item(),
        "max_err_u": max_err_u.item(),
        "X_star": X_star.cpu().numpy(),
        "u_star": u_star.cpu().numpy(),
        "u_pred": u_pred.cpu().numpy(),
        "f_star": f_star.cpu().numpy(),
        "f_pred": f_pred.cpu().numpy(),
        "grid_shape": (grid_points, grid_points),
    }


def evaluate_wave(model, grid_points=100, device="cpu"):
    """Evaluate wave equation model on a regular grid."""
    t = torch.linspace(0, 1, grid_points, dtype=torch.float64, device=device)
    x = torch.linspace(0, 1, grid_points, dtype=torch.float64, device=device)
    T, X = torch.meshgrid(t, x, indexing="ij")
    X_star = torch.stack([T.flatten(), X.flatten()], dim=1)

    with torch.no_grad():
        u_star = wave_exact_u(X_star)

    u_pred, _ = wave_operator(model, X_star[:, 0:1], X_star[:, 1:2])
    u_pred = u_pred.detach()

    err_u = torch.norm(u_pred - u_star) / torch.norm(u_star) * 100
    mse_u = torch.mean((u_pred - u_star) ** 2)
    max_err_u = torch.max(torch.abs(u_pred - u_star))

    return {
        "rel_l2_u": err_u.item(),
        "mse_u": mse_u.item(),
        "max_err_u": max_err_u.item(),
        "X_star": X_star.cpu().numpy(),
        "u_star": u_star.cpu().numpy(),
        "u_pred": u_pred.cpu().numpy(),
        "grid_shape": (grid_points, grid_points),
    }


def evaluate_klein_gordon(model, grid_points=100, device="cpu"):
    """Evaluate Klein-Gordon model on a regular grid."""
    t = torch.linspace(0, 1, grid_points, dtype=torch.float64, device=device)
    x = torch.linspace(0, 1, grid_points, dtype=torch.float64, device=device)
    T, X = torch.meshgrid(t, x, indexing="ij")
    X_star = torch.stack([T.flatten(), X.flatten()], dim=1)

    with torch.no_grad():
        u_star = kg_exact_u(X_star)

    u_pred, _ = klein_gordon_operator(model, X_star[:, 0:1], X_star[:, 1:2])
    u_pred = u_pred.detach()

    err_u = torch.norm(u_pred - u_star) / torch.norm(u_star) * 100
    mse_u = torch.mean((u_pred - u_star) ** 2)
    max_err_u = torch.max(torch.abs(u_pred - u_star))

    return {
        "rel_l2_u": err_u.item(),
        "mse_u": mse_u.item(),
        "max_err_u": max_err_u.item(),
        "X_star": X_star.cpu().numpy(),
        "u_star": u_star.cpu().numpy(),
        "u_pred": u_pred.cpu().numpy(),
        "grid_shape": (grid_points, grid_points),
    }


def evaluate_heat_1d(model, grid_points=100, device="cpu"):
    """Evaluate 1D heat equation model on a regular grid."""
    t = torch.linspace(0, 1, grid_points, dtype=torch.float64, device=device)
    x = torch.linspace(-1, 1, grid_points, dtype=torch.float64, device=device)
    T, X = torch.meshgrid(t, x, indexing="ij")
    X_star = torch.stack([T.flatten(), X.flatten()], dim=1)

    with torch.no_grad():
        u_star = heat_1d_exact_u(X_star)

    u_pred, _ = heat_1d_operator(model, X_star[:, 0:1], X_star[:, 1:2])
    u_pred = u_pred.detach()

    norm_u_star = torch.norm(u_star)
    if norm_u_star < 1e-10:
        norm_u_star = torch.tensor(1.0)
    err_u = torch.norm(u_pred - u_star) / norm_u_star * 100
    mse_u = torch.mean((u_pred - u_star) ** 2)
    max_err_u = torch.max(torch.abs(u_pred - u_star))

    return {
        "rel_l2_u": err_u.item(),
        "mse_u": mse_u.item(),
        "max_err_u": max_err_u.item(),
        "X_star": X_star.cpu().numpy(),
        "u_star": u_star.cpu().numpy(),
        "u_pred": u_pred.cpu().numpy(),
        "grid_shape": (grid_points, grid_points),
    }


EVALUATORS = {
    "helmholtz": evaluate_helmholtz,
    "wave": evaluate_wave,
    "klein_gordon": evaluate_klein_gordon,
    "heat_1d": evaluate_heat_1d,
}


def plot_results(metrics: dict, save_dir: str, title_prefix: str = ""):
    """Generate publication-quality result plots."""
    os.makedirs(save_dir, exist_ok=True)
    gs = metrics["grid_shape"]

    u_star = metrics["u_star"].reshape(gs)
    u_pred = metrics["u_pred"].reshape(gs)
    u_err = np.abs(u_star - u_pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    im0 = axes[0].imshow(u_star, cmap="RdBu_r", extent=[-1, 1, -1, 1], origin="lower", aspect="auto")
    axes[0].set_title(f"{title_prefix}Exact u(x)")
    axes[0].set_xlabel("x₁"); axes[0].set_ylabel("x₂")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u_pred, cmap="RdBu_r", extent=[-1, 1, -1, 1], origin="lower", aspect="auto")
    axes[1].set_title(f"{title_prefix}Predicted u(x)")
    axes[1].set_xlabel("x₁"); axes[1].set_ylabel("x₂")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(u_err, cmap="hot", extent=[-1, 1, -1, 1], origin="lower", aspect="auto")
    axes[2].set_title(f"|Error| (L2={metrics['rel_l2_u']:.2f}%)")
    axes[2].set_xlabel("x₁"); axes[2].set_ylabel("x₂")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "solution_comparison.pdf"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss_history(loss_history: list, save_dir: str, title: str = ""):
    """Plot training loss curve."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(loss_history, linewidth=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Loss")
    ax.set_title(title or "Training Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_history.pdf"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(results: Dict[str, dict], save_dir: str):
    """Plot comparison across multiple experiment conditions.
    
    Args:
        results: dict mapping experiment_name → metrics dict (must include 'loss_history')
    """
    os.makedirs(save_dir, exist_ok=True)

    # Loss comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, r in results.items():
        if "loss_history" in r:
            ax.semilogy(r["loss_history"], label=name, linewidth=1.0)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Total Loss", fontsize=12)
    ax.set_title("Training Convergence Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_comparison.pdf"), dpi=150, bbox_inches="tight")
    plt.close()

    # Error bar chart
    names = []
    l2_errors = []
    param_counts = []
    for name, r in results.items():
        if "rel_l2_u" in r:
            names.append(name)
            l2_errors.append(r["rel_l2_u"])
            param_counts.append(r.get("total_params", 0))

    if names:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = ax1.bar(names, l2_errors, color=colors)
        ax1.set_ylabel("Relative L2 Error (%)", fontsize=12)
        ax1.set_title("Solution Accuracy", fontsize=14)
        ax1.tick_params(axis='x', rotation=30)
        for bar, val in zip(bars, l2_errors):
            ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

        if any(p > 0 for p in param_counts):
            ax2.bar(names, param_counts, color=colors)
            ax2.set_ylabel("Parameter Count", fontsize=12)
            ax2.set_title("Model Complexity", fontsize=14)
            ax2.tick_params(axis='x', rotation=30)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "accuracy_comparison.pdf"), dpi=150, bbox_inches="tight")
        plt.close()
