import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths to your experiment results
baseline_path = "experiments/helmholtz_baseline_q4_n0.0_20260307_200340/metrics.json"
te_path = "experiments/helmholtz_te_q4_n0.0_20260308_150224/metrics.json"

# Check files exist
if not os.path.exists(baseline_path):
    raise FileNotFoundError(f"Baseline metrics not found: {baseline_path}")

if not os.path.exists(te_path):
    raise FileNotFoundError(f"TE metrics not found: {te_path}")

# Load metrics
with open(baseline_path, "r") as f:
    baseline_data = json.load(f)

with open(te_path, "r") as f:
    te_data = json.load(f)

# Extract loss histories
loss_baseline = np.array(baseline_data["loss_history"])
loss_te = np.array(te_data["loss_history"])

# Create iteration index
iters_baseline = np.arange(len(loss_baseline))
iters_te = np.arange(len(loss_te))

# Plot
plt.figure(figsize=(8, 5))

plt.plot(iters_baseline, loss_baseline, label="Baseline QCPINN", color="blue", linewidth=1)
plt.plot(iters_te, loss_te, label="TE-QCPINN", color="red", linewidth=1)

# Log scale like research papers
plt.yscale("log")

plt.xlabel("Iteration")
plt.ylabel("Training Loss")
plt.title("Training Loss Comparison (Helmholtz)")
plt.legend()

plt.grid(True, alpha=0.3)

# Save figure
output_file = "loss_comparison.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300)

print(f"Saved plot to: {output_file}")

# Show plot
plt.show()