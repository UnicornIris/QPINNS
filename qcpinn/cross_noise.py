"""Cross-noise evaluation: test models trained at one noise level under other noise levels.

This is the core experiment for the paper. The protocol:

  1. Train models at noise=0 and noise=p for both fixed-angle repeat QPINN and TE
  2. For each trained model, evaluate at ALL noise levels
  3. Compare degradation curves: does TE degrade less?

The key plot: x-axis = test noise, y-axis = L2 error,
              lines = {repeat_clean, repeat_aware, te_clean, te_aware}
"""

import os
import json
import time
import torch
import numpy as np
from typing import List, Dict, Optional

from qcpinn.solver import QCPINNSolver
from qcpinn.trainer import Trainer
from qcpinn.evaluation import EVALUATORS, plot_loss_history
from qcpinn.datasets import DATASET_REGISTRY


def _swap_noise(model: QCPINNSolver, new_noise: float) -> QCPINNSolver:
    """Create a copy of a model operating at a different noise level.
    
    Rebuilds the quantum layer with new noise but preserves all trained weights.
    """
    config = dict(model.config)
    config["noise_strength"] = new_noise
    
    model_new = QCPINNSolver(config, device=model.device)
    
    # Transfer all compatible weights
    old_state = model.state_dict()
    new_state = model_new.state_dict()
    for key in old_state:
        if key in new_state and old_state[key].shape == new_state[key].shape:
            new_state[key] = old_state[key].clone()
    model_new.load_state_dict(new_state)
    model_new.eval()
    return model_new


def run_cross_noise_study(
    problem: str = "helmholtz",
    train_noise_levels: List[float] = None,
    test_noise_levels: List[float] = None,
    modes: List[str] = None,
    epochs: int = 5000,
    num_qubits: int = 4,
    num_layers: int = 2,
    ansatz: str = "hea",
    batch_size: int = 64,
    lr: float = 5e-4,
    bc_weight: float = 10.0,
    output_dir: str = "./experiments/cross_noise",
    device: Optional[torch.device] = None,
):
    """Run the full cross-noise evaluation study.
    
    Args:
        problem: PDE to solve
        train_noise_levels: Noise levels to train at
        test_noise_levels: Noise levels to evaluate at
        modes: ["repeat", "te"] typically
        epochs: Training epochs per model
        output_dir: Where to save everything
        
    Returns:
        results: Nested dict[mode][train_noise][test_noise] → metrics
    """
    device = device or torch.device("cpu")
    os.makedirs(output_dir, exist_ok=True)
    
    if train_noise_levels is None:
        train_noise_levels = [0.0, 0.01]
    if test_noise_levels is None:
        test_noise_levels = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    if modes is None:
        modes = ["repeat", "te"]
    
    ds_info = DATASET_REGISTRY[problem]
    _, in_dim, out_dim, dom_lo, dom_hi = ds_info
    
    base_config = {
        "problem": problem,
        "input_dim": in_dim,
        "output_dim": out_dim,
        "num_qubits": num_qubits,
        "num_quantum_layers": num_layers,
        "hidden_dim": 50,
        "q_ansatz": ansatz,
        "encoding": "angle",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "optimizer": "lbfgs",
        "bc_weight": bc_weight,
        "grad_clip": 1.0,
        "print_every": 100,
        "shots": None,
        "qml_device": "default.qubit",
        "domain_lo": dom_lo,
        "domain_hi": dom_hi,
        "te_hidden_layers": 2,
        "te_width": 10,
    }
    
    results = {}  # results[mode][train_noise][test_noise] = metrics
    
    # Phase 1: Train models
    print("=" * 70)
    print("PHASE 1: Training models")
    print("=" * 70)
    
    trained_models = {}  # (mode, train_noise) → checkpoint_path
    
    for mode in modes:
        for train_noise in train_noise_levels:
            name = f"{mode}_train{train_noise:.3f}"
            print(f"\n{'='*60}\nTraining: {name} ({epochs} epochs)\n{'='*60}")
            
            config = dict(base_config)
            config["mode"] = mode
            config["noise_strength"] = train_noise
            
            log_dir = os.path.join(output_dir, "training", name)
            
            model = QCPINNSolver(config, device=device)
            trainer = Trainer(model, config, log_dir, device=device)
            
            t0 = time.time()
            best_loss = trainer.train()
            train_time = time.time() - t0
            
            ckpt_path = os.path.join(log_dir, "best_model.pth")
            trained_models[(mode, train_noise)] = ckpt_path
            
            print(f"  Trained in {train_time:.0f}s, best loss: {best_loss:.3e}")
    
    # Phase 2: Cross-evaluate
    print("\n" + "=" * 70)
    print("PHASE 2: Cross-noise evaluation")
    print("=" * 70)
    
    evaluator = EVALUATORS.get(problem)
    
    for mode in modes:
        results[mode] = {}
        for train_noise in train_noise_levels:
            results[mode][train_noise] = {}
            ckpt_path = trained_models[(mode, train_noise)]
            model = QCPINNSolver.load_state(ckpt_path, device=device)
            model.eval()
            
            for test_noise in test_noise_levels:
                name = f"{mode}_train{train_noise:.3f}_test{test_noise:.3f}"
                print(f"  Evaluating: {name}")
                
                if test_noise == train_noise and test_noise == 0.0:
                    # Use the model directly
                    test_model = model
                else:
                    # Swap noise level
                    test_model = _swap_noise(model, test_noise)
                
                if evaluator:
                    metrics = evaluator(test_model, grid_points=50, device=device)
                else:
                    metrics = {"rel_l2_u": float("nan")}
                
                metrics["train_noise"] = train_noise
                metrics["test_noise"] = test_noise
                metrics["mode"] = mode
                results[mode][train_noise][test_noise] = {
                    k: v for k, v in metrics.items()
                    if not isinstance(v, np.ndarray)
                }
    
    # Phase 3: Generate plots and tables
    print("\n" + "=" * 70)
    print("PHASE 3: Generating figures")
    print("=" * 70)
    
    _plot_cross_noise(results, test_noise_levels, output_dir, problem)
    _save_results_table(results, train_noise_levels, test_noise_levels, output_dir)
    
    return results


def _plot_cross_noise(results, test_noise_levels, save_dir, problem):
    """Generate the main cross-noise figure for the paper."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color/style map
    styles = {
        ("repeat", 0.0): {"color": "#1f77b4", "ls": "--", "marker": "o", "label": "Fixed-angle repeat (trained noiseless)"},
        ("repeat", 0.01): {"color": "#1f77b4", "ls": "-", "marker": "s", "label": "Fixed-angle repeat (noise-aware)"},
        ("te", 0.0): {"color": "#d62728", "ls": "--", "marker": "^", "label": "TE (trained noiseless)"},
        ("te", 0.01): {"color": "#d62728", "ls": "-", "marker": "D", "label": "TE (noise-aware)"},
    }
    
    for mode in results:
        for train_noise in results[mode]:
            key = (mode, train_noise)
            if key not in styles:
                # Generate style for unexpected train noise levels
                color = "#2ca02c" if mode == "te" else "#ff7f0e"
                style = {"color": color, "ls": "-.", "marker": "x",
                         "label": f"{mode} (train_noise={train_noise})"}
            else:
                style = styles[key]
            
            xs = []
            ys = []
            for tn in test_noise_levels:
                if tn in results[mode][train_noise]:
                    xs.append(tn)
                    ys.append(results[mode][train_noise][tn].get("rel_l2_u", float("nan")))
            
            ax.plot(xs, ys, color=style["color"], linestyle=style["ls"],
                    marker=style["marker"], label=style["label"], linewidth=1.5, markersize=7)
    
    ax.set_xlabel("Test Noise Strength (p)", fontsize=13)
    ax.set_ylabel("Relative L2 Error (%)", fontsize=13)
    ax.set_title(f"Noise Robustness: {problem.replace('_', ' ').title()}", fontsize=14)
    ax.legend(fontsize=11, loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cross_noise_main.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(save_dir, "cross_noise_main.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: cross_noise_main.pdf")


def _save_results_table(results, train_noise_levels, test_noise_levels, save_dir):
    """Save results as a LaTeX-ready table and JSON."""
    rows = []
    header = ["Mode", "Train Noise", "Test Noise", "L2 Error (%)", "MSE", "Max Error"]
    
    for mode in results:
        for tn_train in results[mode]:
            for tn_test in results[mode][tn_train]:
                m = results[mode][tn_train][tn_test]
                rows.append([
                    mode,
                    f"{tn_train:.3f}",
                    f"{tn_test:.3f}",
                    f"{m.get('rel_l2_u', float('nan')):.2f}",
                    f"{m.get('mse_u', float('nan')):.2e}",
                    f"{m.get('max_err_u', float('nan')):.4f}",
                ])
    
    # Save as plain text table
    with open(os.path.join(save_dir, "results_table.txt"), "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join(row) + "\n")
    
    # Save as LaTeX table
    with open(os.path.join(save_dir, "results_table.tex"), "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write(f"\\caption{{Cross-noise evaluation results}}\n")
        f.write("\\begin{tabular}{ll" + "r" * 4 + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\\end{table}\n")
    
    # Full JSON
    # Convert float keys to strings for JSON
    json_results = {}
    for mode in results:
        json_results[mode] = {}
        for tn_train in results[mode]:
            json_results[mode][str(tn_train)] = {}
            for tn_test in results[mode][tn_train]:
                json_results[mode][str(tn_train)][str(tn_test)] = results[mode][tn_train][tn_test]
    
    with open(os.path.join(save_dir, "all_results.json"), "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"  Saved: results_table.txt, results_table.tex, all_results.json")


def main():
    """CLI for cross-noise study."""
    import argparse
    parser = argparse.ArgumentParser(description="Cross-noise evaluation study")
    parser.add_argument("--problem", type=str, default="helmholtz")
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--num-qubits", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default="./experiments/cross_noise")
    parser.add_argument("--train-noise", nargs="+", type=float, default=[0.0, 0.01])
    parser.add_argument("--test-noise", nargs="+", type=float,
                        default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05])
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run_cross_noise_study(
        problem=args.problem,
        train_noise_levels=args.train_noise,
        test_noise_levels=args.test_noise,
        epochs=args.epochs,
        num_qubits=args.num_qubits,
        num_layers=args.layers,
        output_dir=args.output_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
