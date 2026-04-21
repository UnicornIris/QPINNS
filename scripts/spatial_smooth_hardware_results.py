#!/usr/bin/env python3
"""Apply Helmholtz-compatible spatial smoothing to an existing hardware_results.json."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcpinn.datasets import DATASET_REGISTRY
from qcpinn.hardware import _spatial_smooth_helmholtz


def main():
    parser = argparse.ArgumentParser(description="Spatially smooth an existing hardware_results.json")
    parser.add_argument("--results", required=True, help="Path to hardware_results.json")
    parser.add_argument("--basis", type=int, default=3, help="Number of sine basis functions per dimension")
    parser.add_argument("--output-json", default=None, help="Optional output path")
    args = parser.parse_args()

    results_path = Path(args.results)
    data = json.load(open(results_path))

    problem = data.get("problem", "helmholtz")
    _, _, _, dom_lo, dom_hi = DATASET_REGISTRY[problem]
    X_star = np.asarray(data["X_star"], dtype=float)
    u_hw = np.asarray(data["u_hw"], dtype=float)
    u_sim = np.asarray(data["u_sim"], dtype=float)
    u_exact = np.asarray(data["u_exact"], dtype=float) if "u_exact" in data else None

    u_smooth, coeffs = _spatial_smooth_helmholtz(
        X_star,
        u_hw,
        dom_lo,
        dom_hi,
        basis_per_dim=args.basis,
    )

    out = {
        "results_path": str(results_path),
        "basis": int(args.basis),
        "u_hw_smooth": u_smooth.tolist(),
        "coeffs": coeffs.tolist(),
        "mae_sim_vs_hw_smooth": float(np.mean(np.abs(u_sim - u_smooth))),
        "rel_l2_sim_vs_hw_smooth": float(
            np.linalg.norm(u_sim - u_smooth) / (np.linalg.norm(u_sim) + 1e-10)
        ),
        "correlation_smooth": float(np.corrcoef(u_sim.flatten(), u_smooth.flatten())[0, 1]),
    }
    if u_exact is not None:
        exact_norm = np.linalg.norm(u_exact)
        if exact_norm > 1e-10:
            out["rel_l2_hw_smooth_vs_exact"] = float(np.linalg.norm(u_smooth - u_exact) / exact_norm * 100)

    output_json = Path(args.output_json) if args.output_json else results_path.with_name("hardware_results_smoothed.json")
    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"Saved {output_json}")


if __name__ == "__main__":
    main()
