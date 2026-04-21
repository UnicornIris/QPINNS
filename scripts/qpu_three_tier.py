import json
import time
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

from qcpinn.solver import QCPINNSolver
from qcpinn.trainer import Trainer
from qcpinn.evaluation import evaluate_helmholtz


def log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"{now} | {msg}")


# =========================
# CONFIG BUILDERS
# =========================
def base_config():
    return {
        "problem": "helmholtz",
        "input_dim": 2,
        "output_dim": 1,
        "optimizer": "adam",
        "lr": 1e-3,
        "epochs": 400,
        "batch_size": 1000,
        "bc_weight": 10.0,
        "print_every": 50,
        "val_every": 100,
        "fixed_collocation": True,
        "domain_lo": [-1.0, -1.0],
        "domain_hi": [1.0, 1.0],
        "helmholtz_a1": 1.0,
        "helmholtz_a2": 1.0,
        "helmholtz_lambda": 1.0,
        "output_activation": "tanh",
        "hard_bc": True,
        "seed": 42,
        "noise_strength": 0.0,
    }


def build_te_config():
    cfg = base_config()
    cfg.update({
        "mode": "te",
        "num_qubits": 4,
        "num_quantum_layers": 3,
        "q_ansatz": "hea",
        "hidden_dim": 50,
        "te_hidden_layers": 2,
        "te_width": 10,
    })
    return cfg


def build_repeat_config():
    cfg = base_config()
    cfg.update({
        "mode": "repeat",
        "num_qubits": 4,
        "num_quantum_layers": 3,
        "q_ansatz": "hea",
        "hidden_dim": 50,
    })
    return cfg


# =========================
# TRAINING
# =========================
def train_model(config, label):
    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))

    model = QCPINNSolver(config)

    log("============================================================")
    log(
        f"Training {label} | problem={config['problem']} | "
        f"mode={config['mode']} | qubits={config['num_qubits']} | "
        f"noise={config.get('noise_strength', 0.0)}"
    )
    log(f"Parameters: {model.count_parameters()}")
    log("============================================================")

    log_dir = os.path.join("runs", label)
    trainer = Trainer(model, config, log_dir=log_dir)
    trainer.train()

    model.eval()
    final_metrics = evaluate_helmholtz(model, grid_points=32)

    # Pull best val L2 from trainer's val_history
    best_val = float("inf")
    if trainer.val_history:
        best_val = min(v for _, v in trainer.val_history) / 100.0  # convert % → fraction

    return model, final_metrics, best_val


# =========================
# RUN ONE EXPERIMENT
# =========================
def run_experiment(config, label):
    train_start = time.time()
    model, metrics, best_val = train_model(config, label)
    train_time = time.time() - train_start

    result = {
        "label": label,
        "mode": config["mode"],
        "config": config,
        "rel_l2_u": float(metrics["rel_l2_u"]),
        "rel_l2_f": float(metrics.get("rel_l2_f", float("nan"))),
        "mse_u": float(metrics.get("mse_u", float("nan"))),
        "max_err_u": float(metrics.get("max_err_u", float("nan"))),
        "best_val_l2": float(best_val) if best_val < float("inf") else None,
        "train_time_sec": train_time,
        "timestamp": time.time(),
    }

    if result["best_val_l2"] is not None:
        log(
            f"Done {label} | final_rel_l2={100.0 * result['rel_l2_u']:.2f}% | "
            f"best_val={100.0 * result['best_val_l2']:.2f}%"
        )
    else:
        log(f"Done {label} | final_rel_l2={100.0 * result['rel_l2_u']:.2f}%")

    return result


# =========================
# TIERS
# =========================
def run_tier1():
    log("=== Tier 1: TE Repeats ===")
    results = []
    for i in range(3):
        cfg = build_te_config()
        cfg["seed"] = 42 + i  # different seed per run
        results.append(run_experiment(cfg, f"TE_run_{i+1}"))
    return results


def run_tier2():
    log("=== Tier 2: Repeat Baseline ===")
    results = []
    for i in range(2):
        cfg = build_repeat_config()
        cfg["seed"] = 42 + i
        results.append(run_experiment(cfg, f"Repeat_run_{i+1}"))
    return results


def run_tier3():
    log("=== Tier 3: Ablations ===")
    results = []

    cfg = build_te_config()
    cfg["output_activation"] = "identity"
    results.append(run_experiment(cfg, "TE_no_tanh"))

    cfg = build_te_config()
    cfg["batch_size"] = 2000
    results.append(run_experiment(cfg, "TE_high_res"))

    return results


# =========================
# MASTER
# =========================
def run_all():
    results = []
    results += run_tier1()
    results += run_tier2()
    results += run_tier3()

    with open("qpu_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


# =========================
# PLOTTING
# =========================
def plot_results(results):
    labels = [r["label"] for r in results]
    rel_l2_pct = [100.0 * r["rel_l2_u"] for r in results]

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    plt.bar(x, rel_l2_pct)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Relative L2 Error (%)")
    plt.title("QPINN Experiment Results")
    plt.tight_layout()
    plt.savefig("qpu_rel_l2.png", dpi=160)
    plt.close()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    results = run_all()

    log("=== SUMMARY ===")
    for r in results:
        if r["best_val_l2"] is not None:
            log(
                f"{r['label']}: "
                f"rel_l2_u={100.0 * r['rel_l2_u']:.2f}%, "
                f"best_val={100.0 * r['best_val_l2']:.2f}%"
            )
        else:
            log(f"{r['label']}: rel_l2_u={100.0 * r['rel_l2_u']:.2f}%")

    plot_results(results)