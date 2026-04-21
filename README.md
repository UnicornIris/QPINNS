# Noise-Aware Trainable Embeddings for Quantum PINNs

This repo studies when a quantum PINN can survive the jump from clean simulation to noisy hardware.

The main result is now split into two parts:

- a **hard Helmholtz benchmark** that exposes the simulator-to-hardware gap and lets us diagnose why QPU performance fails
- a **hardware-matched Helmholtz benchmark** that validates the diagnosis and produces a strong real-IonQ QPU result

## Current Headline Results

### Harder Helmholtz benchmark: `sin(pi x1) sin(4 pi x2)`

- Clean TE-QPINN was improved to about `1.38%` relative L2:
  - `experiments/constraint_rescue/te_hardbc_clean_ft_e200`
- The same model had very large readout sensitivity:
  - spectral upper bound about `125`
  - max local final-output gain about `70`
- On real QPU, this benchmark was not reliable enough for a positive hardware claim.

This benchmark is now best understood as a **failure-analysis / diagnosis** case.

### Hardware-matched Helmholtz benchmark: `sin(pi x1) sin(pi x2)`

- Clean TE-QPINN, `4 qubits / 3 layers`: about `1.70%` relative L2
  - `experiments/helmholtz_a11_te_4q3l_hardbc`
- IonQ `forte-1` noisy simulator:
  - TE-QPINN: `31.18%` hardware-vs-exact, correlation `0.982`
  - Repeat QPINN: `95.07%` hardware-vs-exact, correlation `0.454`
- Real IonQ QPU:
  - TE-QPINN: `33.13%` hardware-vs-exact, correlation `0.970`
  - Repeat QPINN: `94.98%` hardware-vs-exact, correlation `0.406`

This is the main positive hardware result in the repo.

## What Changed Scientifically

The original question was: can trainable embeddings make QPINNs more noise-robust than standard fixed-angle QPINNs?

The answer from the repo is now:

1. **Yes, in simulation and noisy simulation, TE-QPINN is clearly more robust than repeat QPINN.**
2. **The hard benchmark fails on real hardware for a concrete reason:** the classical readout after the quantum circuit amplifies hardware perturbations too strongly.
3. **That diagnosis is predictive.** When the Helmholtz target is changed to a lower-frequency mode, the measured gain drops sharply and the QPU result becomes strong.

So the repo is no longer just a generic noise study. It is now a **simulation-to-hardware diagnosis and validation project**.

## Key Folders

### Old hard benchmark

- `experiments/full_comparison_fixed`
  - original clean classical / repeat / TE comparison on the harder Helmholtz problem
- `experiments/noise_robustness`
  - synthetic-noise robustness study on the harder Helmholtz problem
- `experiments/constraint_rescue/te_hardbc_clean_ft_e200`
  - best clean TE checkpoint on the harder problem
  - includes gain diagnosis showing why the hard benchmark is fragile on QPU

### New hardware-matched benchmark

- `experiments/helmholtz_a11_te_4q2l_hardbc`
  - clean TE-QPINN on `sin(pi x1) sin(pi x2)`, 2 layers
- `experiments/helmholtz_a11_te_4q3l_hardbc`
  - clean TE-QPINN on `sin(pi x1) sin(pi x2)`, 3 layers
- `experiments/helmholtz_a11_full_comparison`
  - clean classical PINN and repeat-QPINN comparison on the new benchmark
- `experiments/ionq_forte1_helmholtz_a11_te_4q3l_tanh_opt1`
  - TE-QPINN on IonQ `forte-1` noisy simulator
- `experiments/ionq_forte1_helmholtz_a11_repeat_4q3l_tanh_opt1`
  - repeat-QPINN on IonQ `forte-1` noisy simulator
- `experiments/ionq_qpu_helmholtz_a11_te_4q3l_tanh_debias512_opt1`
  - final TE-QPINN real-QPU result
- `experiments/ionq_qpu_helmholtz_a11_repeat_4q3l_tanh_debias512_opt1`
  - matched repeat-QPINN real-QPU result

### Notes

- `experiments/archive`
  - local archive bucket for one-off demos, aborted branches, and low-value clutter not central to the paper story
- `notes/apr07_a11_results_summary.txt`
  - short plain-text summary of the current `(1,1)` benchmark results

## Model Structure

For the quantum models, the forward path is:

```text
coordinates
  -> optional trainable embedding / preprocessing
  -> quantum circuit
  -> measured quantum features
  -> classical postprocessor
  -> PDE solution u(x)
```

Important:

- the **postprocessor exists in all QPINN variants**
- the novelty here is **not** that a postprocessor exists
- the important result is that we measured how sensitive that readout became and used that to explain and predict hardware behavior

## What “Gain” Means

The repo now includes gain-analysis tooling for the classical readout after the quantum layer.

Two useful quantities:

- **spectral upper bound**
  - worst-case amplification bound for the readout MLP
- **max local final-output gain**
  - actual measured sensitivity at the worst sampled point

Intuition:

- if hardware noise perturbs the quantum outputs a little,
- and the readout gain is large,
- the final PDE prediction can move a lot

That is what broke the harder benchmark and what improved dramatically on the lower-frequency benchmark.

## Quick Start

```bash
# Run tests
python tests/test_all.py

# Clean TE training on Helmholtz
python -m qcpinn.run --problem helmholtz --mode te --epochs 500

# Harder Helmholtz comparison
python scripts/full_comparison.py

# Noise robustness study
python scripts/noise_robustness.py

#New useful one
python scripts/full_comparison.py --epochs 1000 --skip-optuna --output-dir experiments/full_comparison_fixed
```

## IonQ Hardware / Simulator Commands

Set your API key first:

```bash
export IONQ_API_KEY=your_key_here
```

Example noisy-simulator run:

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

Example real-QPU run:

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

## Evaluation Modes

There are two different evaluation styles in this repo.

### Dense simulator evaluation

Used for clean training results.

- evaluates on a dense grid like `100x100`
- reports:
  - `rel_l2_u`
  - `rel_l2_f`
  - `mse_u`
  - `max_err_u`
- writes figures like `solution_comparison.pdf`

### Sparse hardware evaluation

Used for noisy-provider and real-QPU runs.

- evaluates on a sparse grid like `4x4` or `8x8`
- reports:
  - `rel_l2_sim_vs_exact`
  - `rel_l2_hw_vs_exact`
  - `rel_l2_sim_vs_hw`
  - `correlation`
  - `mae_sim_vs_hw`
- writes:
  - `hardware_comparison.pdf`
  - `scatter_sim_vs_hw.pdf`

These are not the same metric families, and that is intentional: dense evaluation is cheap locally, while hardware evaluation is expensive and only practical on sparse grids.

## Project Structure

```text
qcpinn/
├── circuits.py
├── datasets.py
├── evaluation.py
├── hardware.py
├── pde.py
├── run.py
├── solver.py
└── trainer.py

scripts/
├── analyze_postprocessor_gain.py
├── fine_tune_checkpoint.py
├── full_comparison.py
├── hardware_grid_report.py
├── noise_rescue.py
├── noise_robustness.py
└── spatial_smooth_hardware_results.py

tests/
└── test_all.py
```

## Current Bottom Line

If you want the shortest accurate summary of the repo:

- **Old hard Helmholtz benchmark:** useful for stress-testing and diagnosing the simulator-to-hardware gap
- **New lower-frequency Helmholtz benchmark:** produces a strong real-IonQ QPU result
- **TE-QPINN beats repeat-QPINN** on clean sim, noisy sim, and real QPU for the hardware-matched benchmark
