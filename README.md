# Noise-Aware Trainable Embeddings for Quantum PINNs

This repository investigates whether quantum Physics-Informed Neural Networks (QPINNs) can bridge the gap between clean simulation and real quantum hardware. The central question is: can **trainable input embeddings** make QPINNs more robust to hardware noise than conventional fixed-angle QPINNs?

The short answer is yes — with an important caveat about *which* problems survive the jump to real QPUs.

---

## Table of Contents

1. [Background](#background)
2. [Overview](#overview)
3. [How to Use This Codebase](#how-to-use-this-codebase)
4. [IonQ Hardware / Simulator Runs](#ionq-hardware--simulator-runs)
5. [Evaluation Modes](#evaluation-modes)
6. [Scientific Narrative](#scientific-narrative)
7. [Repository Structure](#repository-structure)

---

## Background

**Physics-Informed Neural Networks (PINNs)** are neural networks trained to solve partial differential equations (PDEs) by incorporating the PDE residual directly into the loss function. A standard PINN learns a function `u(x)` that satisfies both the equation and boundary conditions without any labeled solution data.

**Quantum PINNs (QPINNs)** replace the neural network core with a parameterized quantum circuit. The circuit takes input coordinates, applies learned quantum gates, and returns measurement outcomes that are fed into a small classical postprocessor to produce the final PDE solution. The appeal is that quantum circuits can express certain function classes efficiently, but in practice they are fragile: noise on real quantum hardware degrades circuit outputs and can ruin the solution quality.

**Trainable Embeddings (TE)** are a learned preprocessing step applied to the input coordinates before they enter the quantum circuit. Rather than encoding coordinates directly as fixed rotation angles, the embedding transforms them in a way that can be optimized jointly with the rest of the model. The hypothesis is that a well-trained embedding can steer the circuit into a regime that is less sensitive to hardware noise.

This repo tests that hypothesis on the **Helmholtz PDE** — a standard elliptic benchmark — using both simulated and real IonQ quantum hardware.

---

## Overview

The work is organized around two complementary Helmholtz benchmarks:

**Hard Helmholtz benchmark** (`sin(πx₁) sin(4πx₂)`) — a higher-frequency target used to expose and diagnose the simulator-to-hardware gap. Clean simulation results are strong, but real-QPU performance degrades due to high readout sensitivity in the classical postprocessor. This benchmark now functions as a **failure analysis and diagnosis** case, useful for understanding *why* hardware runs fail before investing in QPU time.

**Hardware-matched Helmholtz benchmark** (`sin(πx₁) sin(πx₂)`) — a lower-frequency target chosen after diagnosing why the harder benchmark fails on hardware. This is the **primary positive result**: TE-QPINN runs on a real IonQ QPU with high correlation to the exact solution, while the repeat-QPINN baseline fails under the same conditions.

---

## How to Use This Codebase

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are PyTorch, PennyLane (for quantum circuits), and optionally the IonQ SDK if you plan to run on real hardware or the IonQ cloud simulator.

### Step 2: Run the tests

Verify your environment is set up correctly before running any experiments:

```bash
python tests/test_all.py
```

All tests should pass on a clean install. If any fail, check your PennyLane version and PyTorch installation first.

### Step 3: Train a clean TE-QPINN

The simplest entry point. This trains a Trainable Embedding QPINN on the hardware-matched Helmholtz benchmark in clean (noiseless) simulation:

```bash
python -m qcpinn.run --problem helmholtz --mode te --epochs 500
```

Output is saved to `experiments/` by default. Key files produced:
- `best_val_model.pth` — best checkpoint by validation loss, used for hardware evaluation
- `solution_comparison.pdf` — predicted vs. exact solution on a dense grid
- `training_curves.pdf` — loss curves over training

### Step 4: Run the full comparison (optional)

To reproduce the clean-simulation comparison between classical PINN, repeat-QPINN, and TE-QPINN on the harder Helmholtz benchmark:

```bash
python scripts/full_comparison.py \
  --epochs 1000 \
  --skip-optuna \
  --output-dir experiments/full_comparison_fixed
```

The `--skip-optuna` flag bypasses hyperparameter search and uses the defaults from the paper. Remove it if you want to re-run the search (this takes significantly longer).

### Step 5: Diagnose readout gain

Before committing to a QPU run, check whether your trained model is hardware-viable by analyzing the gain of its classical postprocessor:

```bash
python scripts/analyze_postprocessor_gain.py \
  --checkpoint experiments/helmholtz_a11_te_4q3l_hardbc/best_val_model.pth
```

This reports the spectral upper bound and max local gain. As a rule of thumb:
- **Gain < 20:** generally safe to run on hardware
- **Gain 20–60:** borderline; results may be degraded
- **Gain > 60:** high risk of QPU failure; consider retraining with a lower-frequency target or regularizing the postprocessor

### Step 6: Run the noise robustness study (optional)

To reproduce the synthetic-noise robustness curves:

```bash
python scripts/noise_robustness.py
```

This sweeps depolarizing noise levels and measures how quickly each model's solution quality degrades. Results are written to `experiments/noise_robustness/`.

### Step 7: Hardware evaluation

See the [IonQ Hardware / Simulator Runs](#ionq-hardware--simulator-runs) section below. Always run the noisy simulator first to estimate QPU performance before submitting a real-hardware job.

---

## IonQ Hardware / Simulator Runs

Set your API key before running any hardware or cloud-simulator commands:

```bash
export IONQ_API_KEY=your_key_here
```

### Noisy simulator run

This uses IonQ's `forte-1` noise model in simulation — no QPU time is consumed. Use this to validate your checkpoint before a real-hardware run:

```bash
python -m qcpinn.hardware \
  --checkpoint experiments/helmholtz_a11_te_4q3l_hardbc/best_val_model.pth \
  --backend simulator \
  --noise-model forte-1 \
  --grid 4 \
  --grid-scheme interior \
  --shots 512 \
  --output-activation-override tanh \
  --transpile-optimization-level 1 \
  --output-dir experiments/ionq_forte1_helmholtz_a11_te_4q3l_tanh_opt1
```

### Real QPU run

Submits jobs to the IonQ `forte-1` QPU. This consumes QPU credits and may queue:

```bash
python -m qcpinn.hardware \
  --checkpoint experiments/helmholtz_a11_te_4q3l_hardbc/best_val_model.pth \
  --backend qpu \
  --grid 4 \
  --grid-scheme interior \
  --shots 512 \
  --error-mitigation debias \
  --output-activation-override tanh \
  --transpile-optimization-level 1 \
  --output-dir experiments/ionq_qpu_helmholtz_a11_te_4q3l_tanh_debias512_opt1
```

**Note on `--error-mitigation debias`:** This applies IonQ's debiasing error mitigation, which runs additional calibration circuits. It increases shot count and cost but substantially improves result quality on real hardware. It is strongly recommended for QPU runs.

**Note on `--grid 4 --grid-scheme interior`:** Hardware evaluations use a sparse 4×4 interior grid (16 points total) rather than the dense 100×100 grid used in simulation. This keeps QPU cost manageable. See [Evaluation Modes](#evaluation-modes) for what metrics are reported.

---

## Evaluation Modes

Two distinct evaluation styles are used, and their metrics are **not directly comparable**.

### Dense simulator evaluation

Used for clean training results reported in the paper.

- Grid: 100×100
- Metrics: `rel_l2_u`, `rel_l2_f`, `mse_u`, `max_err_u`
- Output: `solution_comparison.pdf`

### Sparse hardware evaluation

Used for noisy-provider and real-QPU runs. Dense grids are impractical at hardware costs.

- Grid: 4×4 or 8×8
- Metrics:
  - `rel_l2_sim_vs_exact` — how well the clean simulator matches the exact solution
  - `rel_l2_hw_vs_exact` — how well the hardware result matches the exact solution
  - `rel_l2_sim_vs_hw` — how much the hardware result deviates from the clean simulator
  - `correlation` — Pearson correlation between hardware outputs and exact solution (key summary metric)
  - `mae_sim_vs_hw` — mean absolute error between simulator and hardware outputs
- Output: `hardware_comparison.pdf`, `scatter_sim_vs_hw.pdf`

---

## Scientific Narrative

### Model architecture

All QPINN variants follow the same forward path:

```
coordinates
  → optional trainable embedding / preprocessing
  → quantum circuit
  → measured quantum features
  → classical postprocessor (MLP)
  → PDE solution u(x)
```

The classical postprocessor MLP is present in **all** variants — it is not the novelty here. The novelty is the trainable embedding at the input stage, and the finding that the *gain* of the postprocessor is what determines whether a model survives real hardware noise.

### What "gain" means

After the quantum circuit produces measurement outcomes, those outcomes are passed through a classical MLP. If hardware noise slightly perturbs the quantum outputs, a high-gain MLP will amplify that perturbation into a large error in the final PDE prediction.

Two diagnostic quantities are reported:

- **Spectral upper bound** — the worst-case amplification that the MLP can produce, computed from its weight matrices (a global, conservative bound)
- **Max local final-output gain** — the actual measured sensitivity at the worst point sampled from the input domain (tighter and more informative for practice)

On the hard benchmark, both are large (~125 and ~70 respectively), directly predicting QPU failure. On the hardware-matched benchmark, both drop sharply, and QPU performance becomes strong.

### Why the diagnosis is predictive

The gain analysis was performed on the hard benchmark *before* running the hardware-matched benchmark. Switching to the lower-frequency target was specifically motivated by the prediction that lower gain → better QPU results. The real-QPU results confirmed this prediction, establishing gain analysis as a useful screening tool for hardware viability.

---

## Repository Structure

### Code structure

```
qcpinn/
├── circuits.py          # Parameterized quantum circuit definitions
├── datasets.py          # PDE domain sampling and boundary conditions
├── evaluation.py        # Dense and sparse evaluation routines
├── hardware.py          # IonQ cloud/QPU interface
├── pde.py               # Helmholtz and other PDE residual definitions
├── run.py               # Main training entry point
├── solver.py            # Classical PINN solver (baseline)
└── trainer.py           # Training loop, loss, optimizer

scripts/
├── analyze_postprocessor_gain.py     # Gain diagnosis tool
├── fine_tune_checkpoint.py           # Fine-tune an existing checkpoint
├── full_comparison.py                # Multi-model comparison script
├── hardware_grid_report.py           # Summarize hardware evaluation results
├── noise_rescue.py                   # Noise rescue / constraint experiments
├── noise_robustness.py               # Synthetic noise sweep
└── spatial_smooth_hardware_results.py  # Post-processing for sparse hardware outputs

tests/
└── test_all.py
```
