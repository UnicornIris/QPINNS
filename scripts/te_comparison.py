#!/usr/bin/env python3
"""Fair QPINN vs TE-QPINN comparison on Helmholtz.

Compares a standard QPINN (repeat encoding) against TE-QPINN, both using all
4 qubits. The only variable is fixed vs learned encoding:

  - repeat: tiles (x, y) → (x, y, x, y) for AngleEmbedding — 0 learned encoding params
  - te:     TE network maps (x, y) → 4 learned angles — ~1,284 learned encoding params

Runs 4 experiments: {repeat, te} x {clean, noisy p=0.01}, 1000 epochs each.
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
from qcpinn.evaluation import EVALUATORS, plot_results, plot_loss_history


# ─── Config ─────────────────────────────────────────────────────────────────

SEED = 42

BASE = {
    "problem": "helmholtz",
    "input_dim": 2,
    "output_dim": 1,
    "num_qubits": 4,
    "num_quantum_layers": 3,
    "hidden_dim": 30,
    "q_ansatz": "hea",
    "encoding": "angle",
    "bc_weight": 10.0,
    "grad_clip": 1.0,
    "shots": None,
    "qml_device": "default.qubit",
    "domain_lo": [-1.0, -1.0],
    "domain_hi": [1.0, 1.0],
    "te_hidden_layers": 2,
    "te_width": 32,
    "optimizer": "adam",
    "lr": 5e-3,
    "print_every": 50,
}

# Experiment matrix: repeat vs TE, clean vs noisy
RUNS = [
    {"name": "repeat_clean", "mode": "repeat", "noise_strength": 0.0,  "epochs": 1000, "batch_size": 32},
    {"name": "te_clean",     "mode": "te",     "noise_strength": 0.0,  "epochs": 1000, "batch_size": 32},
    {"name": "repeat_noisy", "mode": "repeat", "noise_strength": 0.01, "epochs": 1000, "batch_size": 32},
    {"name": "te_noisy",     "mode": "te",     "noise_strength": 0.01, "epochs": 1000, "batch_size": 32},
]


def run_one(config, log_dir, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = QCPINNSolver(config, device=device)
    trainer = Trainer(model, config, log_dir, device=device)
    t0 = time.time()
    best_loss = trainer.train()
    train_time = time.time() - t0

    best_ckpt = os.path.join(log_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        model = QCPINNSolver.load_state(best_ckpt, device=device)

    evaluator = EVALUATORS[config["problem"]]
    metrics = evaluator(model, grid_points=30, device=device)
    plot_results(metrics, log_dir,
                 title_prefix=f"{config['mode'].upper()} noise={config.get('noise_strength', 0)} ")
    plot_loss_history(model.loss_history, log_dir,
                      title=f"{config['problem']} | {config['mode']} | noise={config.get('noise_strength', 0)}")

    metrics["best_loss"] = best_loss
    metrics["train_time_sec"] = train_time
    metrics["total_params"] = model.count_parameters()
    metrics["loss_history"] = model.loss_history

    save_m = {
        "mode": config["mode"], "noise_strength": config.get("noise_strength", 0.0),
        "seed": seed, "best_loss": best_loss, "rel_l2_u": metrics["rel_l2_u"],
        "mse_u": metrics["mse_u"], "max_err_u": metrics["max_err_u"],
        "train_time_sec": train_time, "total_params": model.count_parameters(),
        "epochs": config["epochs"], "batch_size": config["batch_size"],
    }
    with open(os.path.join(log_dir, "result.json"), "w") as f:
        json.dump(save_m, f, indent=2, default=str)
    return metrics


def make_plots(results, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    colors = {"repeat": "#FF9800", "te": "#4CAF50"}
    labels_map = {"repeat": "Standard QPINN", "te": "TE-QPINN"}

    # 1. Bar chart: clean comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    clean_names = ["repeat_clean", "te_clean"]
    clean_labels = ["Standard QPINN\n(noiseless)", "TE-QPINN\n(noiseless)"]
    clean_errs = [results[n]["rel_l2_u"] for n in clean_names]
    clean_colors = ["#FF9800", "#4CAF50"]
    bars = ax.bar(range(len(clean_errs)), clean_errs, color=clean_colors,
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(clean_errs)))
    ax.set_xticklabels(clean_labels, fontsize=11)
    ax.set_ylabel("Relative L2 Error (%)", fontsize=13)
    ax.set_title("Helmholtz: Solution Accuracy (Noiseless)", fontsize=15)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_noiseless.pdf"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "accuracy_noiseless.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Noise resilience bar chart (side-by-side)
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(2)
    width = 0.35
    repeat_errs = [results["repeat_clean"]["rel_l2_u"], results["repeat_noisy"]["rel_l2_u"]]
    te_errs = [results["te_clean"]["rel_l2_u"], results["te_noisy"]["rel_l2_u"]]

    bars1 = ax.bar(x - width/2, repeat_errs, width, label="Standard QPINN",
                   color="#FF9800", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, te_errs, width, label="TE-QPINN",
                   color="#4CAF50", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Noiseless (p=0)", "Noisy (p=0.01)"], fontsize=12)
    ax.set_ylabel("Relative L2 Error (%)", fontsize=13)
    ax.set_title("Helmholtz: Noise Resilience — Standard vs TE", fontsize=15)
    ax.legend(fontsize=12)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "noise_resilience.pdf"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "noise_resilience.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Degradation ratio
    fig, ax = plt.subplots(figsize=(7, 5))
    repeat_deg = results["repeat_noisy"]["rel_l2_u"] / max(results["repeat_clean"]["rel_l2_u"], 0.01)
    te_deg = results["te_noisy"]["rel_l2_u"] / max(results["te_clean"]["rel_l2_u"], 0.01)
    bars = ax.bar(["Standard QPINN", "TE-QPINN"], [repeat_deg, te_deg],
                  color=["#FF9800", "#4CAF50"], edgecolor="black", linewidth=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No degradation")
    ax.set_ylabel("Error Degradation Ratio\n(noisy / noiseless)", fontsize=13)
    ax.set_title("Noise Impact: Lower = More Resilient", fontsize=15)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{bar.get_height():.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "degradation_ratio.pdf"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "degradation_ratio.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 4. Loss curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Noiseless
    for name in ["repeat_clean", "te_clean"]:
        if name in results and "loss_history" in results[name]:
            mode = name.split("_")[0]
            c = colors.get(mode, "#999999")
            axes[0].semilogy(results[name]["loss_history"], label=labels_map.get(mode, name),
                           color=c, linewidth=1.5)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Total Loss")
    axes[0].set_title("Noiseless Training"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    # Noisy
    for name in ["repeat_noisy", "te_noisy"]:
        if name in results and "loss_history" in results[name]:
            mode = name.split("_")[0]
            c = colors.get(mode, "#999999")
            axes[1].semilogy(results[name]["loss_history"], label=labels_map.get(mode, name),
                           color=c, linewidth=1.5)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Total Loss")
    axes[1].set_title("Noisy Training (p=0.01)"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.suptitle("Helmholtz: Training Convergence", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.pdf"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "loss_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 5. Parameter comparison table
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('off')
    repeat_params = results["repeat_clean"]["total_params"]
    te_params = results["te_clean"]["total_params"]
    table_data = [
        ["", "Standard QPINN (repeat)", "TE-QPINN"],
        ["Qubits used", "All 4", "All 4"],
        ["Encoding", "Fixed: (x,y,x,y)", "Learned: TE network"],
        ["Encoding params", "0", str(te_params.get("te", 0))],
        ["Quantum params", str(repeat_params.get("quantum_layer", "?")),
         str(te_params.get("quantum_layer", "?"))],
        ["Postprocessor params", str(repeat_params.get("postprocessor", "?")),
         str(te_params.get("postprocessor", "?"))],
        ["Total params", str(repeat_params.get("total", "?")),
         str(te_params.get("total", "?"))],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    for i in range(len(table_data[0])):
        table[0, i].set_facecolor('#E8E8E8')
        table[0, i].set_text_props(fontweight='bold')
    plt.title("Parameter Comparison (Fair Setup)", fontsize=13, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "param_comparison.pdf"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "param_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    device = torch.device("cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("experiments", f"te_comparison_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Output: {out_dir}\n")
    results = {}

    for run_spec in RUNS:
        name = run_spec["name"]
        log_dir = os.path.join(out_dir, name)
        print(f"\n{'='*60}")
        print(f"  Running: {name} (mode={run_spec['mode']}, noise={run_spec.get('noise_strength', 0)}, "
              f"epochs={run_spec['epochs']}, batch={run_spec['batch_size']})")
        print(f"{'='*60}")

        config = copy.deepcopy(BASE)
        config.update(run_spec)
        del config["name"]

        metrics = run_one(config, log_dir, SEED, device)
        results[name] = metrics
        print(f"  -> L2={metrics['rel_l2_u']:.2f}%  MSE={metrics['mse_u']:.2e}  "
              f"Time={metrics['train_time_sec']:.1f}s")

    # Plots
    make_plots(results, out_dir)

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Run':<22} {'Params':>7} {'Rel L2 (%)':>12} {'MSE':>12} {'Time (s)':>10}")
    print("-" * 80)
    for run_spec in RUNS:
        name = run_spec["name"]
        if name in results:
            r = results[name]
            p = r["total_params"].get("total", "?") if isinstance(r["total_params"], dict) else r["total_params"]
            print(f"{name:<22} {p:>7} {r['rel_l2_u']:>11.2f}% {r['mse_u']:>12.2e} {r['train_time_sec']:>10.1f}")
    print("=" * 80)

    # Degradation analysis
    repeat_deg = results["repeat_noisy"]["rel_l2_u"] / max(results["repeat_clean"]["rel_l2_u"], 0.01)
    te_deg = results["te_noisy"]["rel_l2_u"] / max(results["te_clean"]["rel_l2_u"], 0.01)
    print(f"\nNoise Degradation (p=0.01):")
    print(f"  Standard QPINN: {results['repeat_clean']['rel_l2_u']:.2f}% -> {results['repeat_noisy']['rel_l2_u']:.2f}% ({repeat_deg:.2f}x)")
    print(f"  TE-QPINN:       {results['te_clean']['rel_l2_u']:.2f}% -> {results['te_noisy']['rel_l2_u']:.2f}% ({te_deg:.2f}x)")
    if te_deg < repeat_deg:
        print(f"  -> TE is {repeat_deg/te_deg:.1f}x more noise-resilient than standard QPINN")
    else:
        print(f"  -> Standard QPINN degraded less ({repeat_deg:.2f}x vs {te_deg:.2f}x)")

    summary = {name: {"rel_l2_u": r["rel_l2_u"], "mse_u": r["mse_u"],
                       "best_loss": r["best_loss"], "train_time_sec": r["train_time_sec"],
                       "total_params": r["total_params"]}
               for name, r in results.items()}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {out_dir}")
