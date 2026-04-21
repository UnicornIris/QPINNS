"""IonQ hardware interface for inference validation.

This module handles running a TRAINED model's forward pass on IonQ hardware.
Training on real hardware is impractical (~1000s per iteration), so the workflow is:

    1. Train on simulator (with or without simulated noise)
    2. Load trained model
    3. Replace the PennyLane device with IonQ
    4. Run inference-only forward passes on a grid
    5. Compare simulator vs hardware predictions

Requirements:
    - pip install pennylane-qiskit qiskit-ionq  (already in qcpinn-modern env)
    - IONQ_API_KEY in .env file or environment variable

Usage:
    # IonQ simulator with Aria-1 noise model (free, realistic noise)
    python -m qcpinn.hardware --checkpoint path/to/best_model.pth --backend simulator --noise-model aria-1

    # Real IonQ QPU with debiasing and resumable partial saves
    python -m qcpinn.hardware --checkpoint path/to/best_model.pth --backend qpu --shots 512 --error-mitigation debias --resume
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pennylane as qml
import torch

from qcpinn.solver import QCPINNSolver
from qcpinn.datasets import get_helmholtz_params


_PARTIAL_RESULTS_FILE = "hardware_partial_results.json"
_RUN_CONFIG_FILE = "hardware_run_config.json"


def _load_env():
    """Load .env file if it exists."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


def _json_default(value):
    """Best-effort JSON serializer for numpy / torch values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _load_partial_results(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _build_eval_grid(dom_lo, dom_hi, grid_points: int, grid_scheme: str, device: torch.device):
    """Build a small evaluation grid for simulator/QPU inference."""
    axes = []
    for lo, hi in zip(dom_lo, dom_hi):
        if grid_scheme == "endpoints":
            axis = torch.linspace(lo, hi, grid_points, dtype=torch.float64, device=device)
        elif grid_scheme == "interior":
            axis = torch.linspace(lo, hi, grid_points + 2, dtype=torch.float64, device=device)[1:-1]
        else:
            raise ValueError(f"Unknown grid_scheme: {grid_scheme}")
        axes.append(axis)
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.flatten() for m in mesh], dim=1)


def _sine_basis_design_matrix(
    X_star: np.ndarray,
    dom_lo,
    dom_hi,
    basis_per_dim: int,
) -> np.ndarray:
    """Build a 2D sine-basis design matrix on the rectangular domain."""
    if X_star.shape[1] != 2:
        raise ValueError("Spatial smoothing is currently implemented only for 2D grids")

    lo = np.asarray(dom_lo, dtype=float)
    hi = np.asarray(dom_hi, dtype=float)
    span = np.maximum(hi - lo, 1e-12)
    S = (X_star - lo) / span

    cols = []
    for k in range(1, basis_per_dim + 1):
        for l in range(1, basis_per_dim + 1):
            cols.append(
                np.sin(k * np.pi * S[:, 0:1]) * np.sin(l * np.pi * S[:, 1:2])
            )
    return np.concatenate(cols, axis=1)


def _spatial_smooth_helmholtz(
    X_star: np.ndarray,
    u_values: np.ndarray,
    dom_lo,
    dom_hi,
    basis_per_dim: int = 4,
    ridge: float = 1e-6,
):
    """Fit a smooth Helmholtz-compatible sine basis to sparse hardware outputs."""
    Phi = _sine_basis_design_matrix(X_star, dom_lo, dom_hi, basis_per_dim)
    y = np.asarray(u_values, dtype=float).reshape(-1, 1)
    eye = np.eye(Phi.shape[1], dtype=float)
    coeffs = np.linalg.solve(Phi.T @ Phi + ridge * eye, Phi.T @ y)
    u_smooth = Phi @ coeffs
    return u_smooth.reshape(u_values.shape), coeffs.reshape(-1, 1)


def _normalize_error_mitigation(error_mitigation: Optional[str]):
    """Map CLI strings to qiskit-ionq error mitigation settings."""
    if error_mitigation in (None, "", "none"):
        return None

    from qiskit_ionq.constants import ErrorMitigation

    mapping = {
        "debias": ErrorMitigation.DEBIASING,
        "debiasing": ErrorMitigation.DEBIASING,
        "off": ErrorMitigation.NO_DEBIASING,
        "none": None,
    }
    if error_mitigation not in mapping:
        raise ValueError(
            f"Unknown error mitigation mode: {error_mitigation}. "
            "Expected one of: none, debias, debiasing, off."
        )
    return mapping[error_mitigation]


def _get_ionq_backend(
    backend_name,
    noise_model=None,
    shots: Optional[int] = None,
    error_mitigation: Optional[str] = None,
    transpile_optimization_level: int = 1,
    extra_metadata: Optional[dict] = None,
):
    """Create an IonQ backend via qiskit-ionq provider.

    For QPU, queries the API for the first available QPU target
    (e.g. qpu.forte-1) since the generic 'ionq_qpu' doesn't work.
    """
    from qiskit_ionq import IonQProvider
    import requests

    provider = IonQProvider()

    mitigation = _normalize_error_mitigation(error_mitigation)
    if backend_name == "qpu":
        # Find the first available QPU via the API
        api_key = os.environ.get("IONQ_API_KEY", "")
        headers = {"Authorization": f"apiKey {api_key}"}
        resp = requests.get("https://api.ionq.co/v0.3/backends", headers=headers, timeout=30)
        qpu_target = None
        if resp.ok:
            for b in resp.json():
                if b["backend"].startswith("qpu.") and b["status"] == "available":
                    qpu_target = b["backend"]
                    break
        if qpu_target is None:
            raise RuntimeError("No available IonQ QPU found. Check https://cloud.ionq.com/backends")
        print(f"  Selected QPU: {qpu_target}")
        qiskit_backend = provider.get_backend(qpu_target)
        backend_options = {}
        if shots is not None:
            backend_options["shots"] = shots
        if mitigation is not None:
            backend_options["error_mitigation"] = mitigation
        if backend_options:
            qiskit_backend.set_options(**backend_options)
    else:
        qiskit_backend = provider.get_backend("ionq_simulator")
        backend_options = {}
        if noise_model:
            backend_options["noise_model"] = noise_model
        if shots is not None:
            backend_options["shots"] = shots
        if backend_options:
            qiskit_backend.set_options(**backend_options)

    return qiskit_backend


def _load_model_for_inference(checkpoint_path: str, device: torch.device) -> QCPINNSolver:
    """Load a model checkpoint for inference.

    Some fine-tuning workflows save optimizer state for a subset of parameters
    (for example, when only the quantum/post layers are trainable). In that
    case the full `load_state` roundtrip can fail on optimizer restoration even
    though the model weights themselves are valid. For hardware inference we
    only need the weights and config.
    """
    try:
        return QCPINNSolver.load_state(checkpoint_path, device=device)
    except ValueError as exc:
        if "parameter group" not in str(exc):
            raise
        state = torch.load(checkpoint_path, map_location=device)
        model = QCPINNSolver(state["config"], device=device)
        model.load_state_dict(state["model_state_dict"])
        model.loss_history = state.get("loss_history", [])
        return model


def _rebuild_model_with_config_overrides(
    model: QCPINNSolver,
    overrides: dict,
    device: torch.device,
) -> QCPINNSolver:
    """Rebuild a model with config overrides while preserving trained weights."""
    if not overrides:
        return model

    new_cfg = dict(model.config)
    new_cfg.update(overrides)
    new_model = QCPINNSolver(new_cfg, device=device)
    new_model.load_state_dict(model.state_dict(), strict=False)
    new_model.loss_history = list(model.loss_history)
    new_model.eval()
    return new_model


def hardware_inference(
    checkpoint_path: str,
    grid_points: int = 10,
    shots: int = 1024,
    backend: str = "simulator",
    noise_model: Optional[str] = None,
    api_key: Optional[str] = None,
    device: Optional[torch.device] = None,
    save_dir: Optional[str] = None,
    error_mitigation: Optional[str] = None,
    resume: bool = False,
    grid_scheme: str = "endpoints",
    output_activation_override: Optional[str] = None,
    output_scale_override: Optional[float] = None,
    hard_bc_override: bool = False,
    hard_bc_scale_override: Optional[float] = None,
    transpile_optimization_level: int = 1,
    spatial_smooth: bool = False,
    spatial_basis: int = 3,
):
    """Run inference on IonQ hardware and compare with simulator prediction.

    Args:
        checkpoint_path: Path to a trained model checkpoint (.pth)
        grid_points: Number of grid points per dimension (keep small for hardware!)
        shots: Shots per circuit evaluation
        backend: "simulator" (free) or "qpu" (real hardware)
        noise_model: IonQ noise model for simulator, e.g. "aria-1", "harmony"
        api_key: IonQ API key. If None, reads from .env or IONQ_API_KEY env var.
        device: torch device
        save_dir: Where to save results (defaults to checkpoint directory)
        error_mitigation: IonQ QPU mitigation mode ("none" or "debias")
        resume: Resume from a partial point-by-point run in save_dir
        grid_scheme: "endpoints" or "interior" sampling on the eval grid
        output_activation_override: Optional inference-time override for output activation
        output_scale_override: Optional inference-time override for output scaling
        hard_bc_override: Enable Helmholtz hard zero-boundary projection at inference time
        hard_bc_scale_override: Optional inference-time override for boundary-envelope steepness
        transpile_optimization_level: Qiskit transpiler optimization level passed to
            the PennyLane-Qiskit device. IonQ recommends 0-1 for QIS circuits.
        spatial_smooth: Fit a smooth sine basis to the final hardware outputs
        spatial_basis: Number of sine basis functions per dimension for smoothing

    Returns:
        Dictionary with simulator predictions, hardware predictions, and comparison metrics
    """
    # Load API key from .env if needed
    _load_env()
    if api_key:
        os.environ["IONQ_API_KEY"] = api_key

    if "IONQ_API_KEY" not in os.environ:
        raise RuntimeError(
            "IONQ_API_KEY not set. Either:\n"
            "  1. Create a .env file with IONQ_API_KEY=your_key\n"
            "  2. export IONQ_API_KEY=your_key\n"
            "  3. Pass --api-key your_key on CLI\n"
            "Get your key at: https://cloud.ionq.com"
        )

    device = device or torch.device("cpu")
    save_dir = Path(save_dir or os.path.dirname(checkpoint_path))
    save_dir.mkdir(parents=True, exist_ok=True)
    partial_path = save_dir / _PARTIAL_RESULTS_FILE
    run_config_path = save_dir / _RUN_CONFIG_FILE
    run_tag = save_dir.name or "hardware_run"

    if backend == "qpu" and error_mitigation in {"debias", "debiasing"} and shots < 500:
        raise ValueError("IonQ debiasing requires at least 500 shots. Increase --shots or disable debiasing.")

    # Load the trained model (simulator version)
    print(f"Loading checkpoint: {checkpoint_path}")
    model_sim = _load_model_for_inference(checkpoint_path, device=device)
    override_cfg = {}
    if output_activation_override is not None:
        override_cfg["output_activation"] = output_activation_override
    if output_scale_override is not None:
        override_cfg["output_scale"] = output_scale_override
    if hard_bc_override:
        override_cfg["hard_bc"] = True
    if hard_bc_scale_override is not None:
        override_cfg["hard_bc_scale"] = hard_bc_scale_override
    model_sim = _rebuild_model_with_config_overrides(model_sim, override_cfg, device=device)
    model_sim.eval()
    config = model_sim.config
    problem = config["problem"]
    helmholtz_a1, helmholtz_a2, helmholtz_lam = get_helmholtz_params(config)

    # Build evaluation grid
    from qcpinn.datasets import DATASET_REGISTRY
    ds_info = DATASET_REGISTRY.get(problem)
    if ds_info:
        _, _, _, dom_lo, dom_hi = ds_info
    else:
        dom_lo, dom_hi = [-1., -1.], [1., 1.]

    X_star = _build_eval_grid(dom_lo, dom_hi, grid_points, grid_scheme, device)

    # Get simulator predictions
    print("Running simulator inference...")
    with torch.no_grad():
        u_sim = model_sim(X_star).cpu().numpy()

    # Get exact solution if available
    from qcpinn.datasets import helmholtz_exact_u, wave_exact_u, kg_exact_u, heat_1d_exact_u
    exact_fns = {
        "helmholtz": lambda X: helmholtz_exact_u(X, a1=helmholtz_a1, a2=helmholtz_a2), "wave": wave_exact_u,
        "klein_gordon": kg_exact_u, "heat_1d": heat_1d_exact_u,
    }
    u_exact = None
    if problem in exact_fns:
        with torch.no_grad():
            u_exact = exact_fns[problem](X_star).cpu().numpy()

    # Build IonQ backend
    backend_label = f"IonQ {backend}"
    if noise_model:
        backend_label += f" ({noise_model})"
    print(f"Setting up {backend_label} (shots={shots})...")

    run_config = {
        "checkpoint_path": checkpoint_path,
        "backend": backend,
        "noise_model": noise_model,
        "shots": shots,
        "grid_points": grid_points,
        "grid_scheme": grid_scheme,
        "error_mitigation": error_mitigation or "none",
        "run_tag": run_tag,
        "problem": problem,
        "helmholtz_a1": helmholtz_a1,
        "helmholtz_a2": helmholtz_a2,
        "helmholtz_lambda": helmholtz_lam,
        "output_activation": config.get("output_activation", "identity"),
        "output_scale": float(config.get("output_scale", 1.0)),
        "hard_bc": bool(config.get("hard_bc", False)),
        "hard_bc_scale": float(config.get("hard_bc_scale", 10.0)),
        "transpile_optimization_level": int(transpile_optimization_level),
        "spatial_smooth": bool(spatial_smooth),
        "spatial_basis": int(spatial_basis),
    }
    _save_json(run_config_path, run_config)

    ionq_backend = _get_ionq_backend(
        backend,
        noise_model,
        shots=shots,
        error_mitigation=error_mitigation,
        transpile_optimization_level=transpile_optimization_level,
    )

    # Rebuild the quantum layer with IonQ device
    config_hw = dict(config)
    config_hw["qml_device"] = "qiskit.remote"
    config_hw["shots"] = shots
    config_hw["qiskit_backend"] = ionq_backend
    config_hw["diff_method"] = "parameter-shift"
    config_hw["noise_strength"] = 0.0  # Real noise comes from hardware
    config_hw["transpile_optimization_level"] = int(transpile_optimization_level)

    model_hw = QCPINNSolver(config_hw, device=device)
    # Copy all trained weights from simulator model
    sim_state = model_sim.state_dict()
    hw_state = model_hw.state_dict()
    for key in sim_state:
        if key in hw_state and sim_state[key].shape == hw_state[key].shape:
            hw_state[key] = sim_state[key]
    model_hw.load_state_dict(hw_state)
    model_hw.eval()

    # Run hardware inference point-by-point
    print(f"Running hardware inference on {X_star.shape[0]} points...")
    output_dim = config.get("output_dim", 1)
    u_hw = np.zeros((X_star.shape[0], output_dim))
    times = []
    start_idx = 0

    if resume:
        partial = _load_partial_results(partial_path)
        if partial:
            completed = int(partial.get("completed_points", 0))
            saved_u = np.asarray(partial.get("u_hw", []), dtype=float)
            if saved_u.size:
                rows = min(completed, saved_u.shape[0], X_star.shape[0])
                u_hw[:rows] = saved_u[:rows]
                times = list(partial.get("times", []))[:rows]
                start_idx = rows
                print(f"Resuming from point {start_idx + 1}/{X_star.shape[0]}")

    for i in range(start_idx, X_star.shape[0]):
        t0 = time.time()
        with torch.no_grad():
            u_i = model_hw(X_star[i:i+1]).cpu().numpy()
        dt = time.time() - t0
        u_hw[i] = u_i
        times.append(dt)

        _save_json(
            partial_path,
            {
                **run_config,
                "completed_points": i + 1,
                "u_hw": u_hw[: i + 1],
                "times": times,
            },
        )

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  Point {i+1}/{X_star.shape[0]} | dt={dt:.1f}s | "
                  f"sim={u_sim[i, 0]:.4f} hw={u_hw[i, 0]:.4f}")

    # Compute comparison metrics
    diff = np.abs(u_sim - u_hw)
    mae = np.mean(diff)
    max_diff = np.max(diff)
    correlation = np.corrcoef(u_sim.flatten(), u_hw.flatten())[0, 1] if len(u_sim) > 1 else 0.0
    rel_diff = np.linalg.norm(u_sim - u_hw) / (np.linalg.norm(u_sim) + 1e-10)

    results = {
        "backend": backend,
        "noise_model": noise_model,
        "shots": shots,
        "grid_points": grid_points,
        "grid_scheme": grid_scheme,
        "total_circuits": X_star.shape[0],
        "run_tag": run_tag,
        "error_mitigation": error_mitigation or "none",
        "helmholtz_a1": helmholtz_a1,
        "helmholtz_a2": helmholtz_a2,
        "helmholtz_lambda": helmholtz_lam,
        "output_activation": config.get("output_activation", "identity"),
        "output_scale": float(config.get("output_scale", 1.0)),
        "hard_bc": bool(config.get("hard_bc", False)),
        "hard_bc_scale": float(config.get("hard_bc_scale", 10.0)),
        "transpile_optimization_level": int(transpile_optimization_level),
        "total_time_sec": sum(times),
        "avg_time_per_point_sec": float(np.mean(times)),
        "mae_sim_vs_hw": float(mae),
        "max_diff_sim_vs_hw": float(max_diff),
        "rel_l2_sim_vs_hw": float(rel_diff),
        "correlation": float(correlation),
        "u_sim": u_sim.tolist(),
        "u_hw": u_hw.tolist(),
        "X_star": X_star.cpu().numpy().tolist(),
        "times": times,
    }

    # Add exact solution comparison if available
    if u_exact is not None:
        exact_norm = np.linalg.norm(u_exact)
        results["exact_solution_l2_norm"] = float(exact_norm)
        if exact_norm > 1e-10:
            rel_l2_sim = np.linalg.norm(u_sim - u_exact) / exact_norm * 100
            rel_l2_hw = np.linalg.norm(u_hw - u_exact) / exact_norm * 100
            results["rel_l2_sim_vs_exact"] = float(rel_l2_sim)
            results["rel_l2_hw_vs_exact"] = float(rel_l2_hw)
            results["u_exact"] = u_exact.tolist()
        else:
            results["exact_metric_warning"] = (
                "Exact solution norm is ~0 on this evaluation grid; "
                "relative L2 vs exact is undefined."
            )

    if spatial_smooth:
        if problem != "helmholtz":
            raise ValueError("Spatial smoothing is currently only supported for Helmholtz")
        u_hw_smooth, coeffs = _spatial_smooth_helmholtz(
            X_star.cpu().numpy(),
            u_hw,
            dom_lo,
            dom_hi,
            basis_per_dim=spatial_basis,
        )
        diff_s = np.abs(u_sim - u_hw_smooth)
        results["u_hw_smooth"] = u_hw_smooth.tolist()
        results["spatial_smoothing_basis"] = int(spatial_basis)
        results["spatial_smoothing_coeffs"] = coeffs.tolist()
        results["mae_sim_vs_hw_smooth"] = float(np.mean(diff_s))
        results["max_diff_sim_vs_hw_smooth"] = float(np.max(diff_s))
        results["rel_l2_sim_vs_hw_smooth"] = float(
            np.linalg.norm(u_sim - u_hw_smooth) / (np.linalg.norm(u_sim) + 1e-10)
        )
        results["correlation_smooth"] = float(
            np.corrcoef(u_sim.flatten(), u_hw_smooth.flatten())[0, 1]
        ) if len(u_sim) > 1 else 0.0
        if u_exact is not None and "exact_solution_l2_norm" in results and results["exact_solution_l2_norm"] > 1e-10:
            exact_norm = results["exact_solution_l2_norm"]
            results["rel_l2_hw_smooth_vs_exact"] = float(
                np.linalg.norm(u_hw_smooth - u_exact) / exact_norm * 100
            )

    # Save results
    _save_json(save_dir / "hardware_results.json", results)
    if partial_path.exists():
        partial_path.unlink()

    # Summary
    print(f"\n{'='*60}")
    print(f"Hardware Inference Summary ({backend_label})")
    print(f"{'='*60}")
    print(f"  Grid:                {grid_points}x{grid_points} = {X_star.shape[0]} points")
    print(f"  Shots per circuit:   {shots}")
    print(f"  Total time:          {sum(times):.1f}s")
    print(f"  Avg time per point:  {np.mean(times):.1f}s")
    print(f"  MAE (sim vs hw):     {mae:.6f}")
    print(f"  Max diff:            {max_diff:.6f}")
    print(f"  Rel L2 diff:         {rel_diff:.4f}")
    print(f"  Correlation:         {correlation:.6f}")
    if "rel_l2_sim_vs_exact" in results:
        print(f"  Sim vs exact L2:     {results['rel_l2_sim_vs_exact']:.2f}%")
        print(f"  HW vs exact L2:      {results['rel_l2_hw_vs_exact']:.2f}%")
    elif "exact_metric_warning" in results:
        print("  Exact comparison:    skipped (exact solution is ~0 on this grid)")
    print(f"  Results saved to:    {save_dir}")

    # Plot comparison
    _plot_hardware_comparison(u_sim, u_hw, u_exact, grid_points,
                              dom_lo, dom_hi, results, str(save_dir), backend_label)

    return results


def _plot_hardware_comparison(u_sim, u_hw, u_exact, grid_points,
                               dom_lo, dom_hi, metrics, save_dir, backend_label):
    """Plot simulator vs hardware predictions side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gs = (grid_points, grid_points)
    u_s = u_sim.reshape(gs)
    u_h = u_hw.reshape(gs)
    u_d = np.abs(u_s - u_h)
    extent = [dom_lo[1], dom_hi[1], dom_lo[0], dom_hi[0]]

    n_plots = 4 if u_exact is not None else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))

    vmin = min(u_s.min(), u_h.min())
    vmax = max(u_s.max(), u_h.max())
    if u_exact is not None:
        u_e = u_exact.reshape(gs)
        vmin = min(vmin, u_e.min())
        vmax = max(vmax, u_e.max())

    idx = 0
    if u_exact is not None:
        im = axes[idx].imshow(u_e, cmap="RdBu_r", extent=extent, origin="lower",
                               aspect="auto", vmin=vmin, vmax=vmax)
        axes[idx].set_title("Exact Solution")
        axes[idx].set_xlabel("x"); axes[idx].set_ylabel("t")
        plt.colorbar(im, ax=axes[idx])
        idx += 1

    im0 = axes[idx].imshow(u_s, cmap="RdBu_r", extent=extent, origin="lower",
                             aspect="auto", vmin=vmin, vmax=vmax)
    axes[idx].set_title("Simulator Prediction")
    axes[idx].set_xlabel("x"); axes[idx].set_ylabel("t")
    plt.colorbar(im0, ax=axes[idx])
    idx += 1

    im1 = axes[idx].imshow(u_h, cmap="RdBu_r", extent=extent, origin="lower",
                             aspect="auto", vmin=vmin, vmax=vmax)
    axes[idx].set_title(f"{backend_label}\n({metrics['shots']} shots)")
    axes[idx].set_xlabel("x"); axes[idx].set_ylabel("t")
    plt.colorbar(im1, ax=axes[idx])
    idx += 1

    im2 = axes[idx].imshow(u_d, cmap="hot", extent=extent, origin="lower", aspect="auto")
    axes[idx].set_title(f"|Sim - HW| (MAE={metrics['mae_sim_vs_hw']:.4f})")
    axes[idx].set_xlabel("x"); axes[idx].set_ylabel("t")
    plt.colorbar(im2, ax=axes[idx])

    plt.suptitle(f"Simulator vs Hardware (corr={metrics['correlation']:.4f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hardware_comparison.pdf"), dpi=150, bbox_inches="tight")
    plt.close()

    # Scatter plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(u_sim.flatten(), u_hw.flatten(), alpha=0.5, s=20)
    lims = [min(u_sim.min(), u_hw.min()), max(u_sim.max(), u_hw.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="y=x")
    ax.set_xlabel("Simulator u(x)", fontsize=12)
    ax.set_ylabel("Hardware u(x)", fontsize=12)
    ax.set_title(f"Sim vs {backend_label} (r={metrics['correlation']:.4f})")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scatter_sim_vs_hw.pdf"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """CLI entry point for hardware inference."""
    import argparse
    parser = argparse.ArgumentParser(description="Run trained QCPINN on IonQ hardware")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--grid", type=int, default=10,
                        help="Grid points per dimension (10x10=100 circuits)")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--backend", type=str, default="simulator",
                        choices=["simulator", "qpu"],
                        help="IonQ backend")
    parser.add_argument("--noise-model", type=str, default=None,
                        choices=["aria-1", "aria-2", "forte-1", "forte-enterprise-1", "harmony"],
                        help="IonQ noise model for simulator (e.g. aria-1)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="IonQ API key (or use .env / IONQ_API_KEY env var)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--grid-scheme", type=str, default="endpoints",
                        choices=["endpoints", "interior"],
                        help="Whether to sample the eval grid on the domain endpoints or interior points")
    parser.add_argument("--error-mitigation", type=str, default="none",
                        choices=["none", "debias", "debiasing", "off"],
                        help="IonQ QPU error mitigation mode")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from hardware_partial_results.json if present")
    parser.add_argument("--output-activation-override", type=str, default=None,
                        choices=["identity", "tanh"],
                        help="Optional inference-time override for the model output activation")
    parser.add_argument("--output-scale-override", type=float, default=None,
                        help="Optional inference-time override for output scaling")
    parser.add_argument("--hard-bc-override", action="store_true",
                        help="Enable hard zero-boundary projection at inference time (Helmholtz only)")
    parser.add_argument("--hard-bc-scale-override", type=float, default=None,
                        help="Optional inference-time override for boundary-envelope steepness")
    parser.add_argument("--transpile-optimization-level", type=int, default=1,
                        choices=[0, 1, 2, 3],
                        help="Qiskit transpiler optimization level for IonQ runs")
    parser.add_argument("--spatial-smooth", action="store_true",
                        help="Fit a Helmholtz-compatible sine basis to the hardware outputs")
    parser.add_argument("--spatial-basis", type=int, default=3,
                        help="Number of sine basis functions per dimension for spatial smoothing")

    args = parser.parse_args()
    hardware_inference(
        checkpoint_path=args.checkpoint,
        grid_points=args.grid,
        shots=args.shots,
        backend=args.backend,
        noise_model=args.noise_model,
        api_key=args.api_key,
        save_dir=args.output_dir,
        error_mitigation=args.error_mitigation,
        resume=args.resume,
        grid_scheme=args.grid_scheme,
        output_activation_override=args.output_activation_override,
        output_scale_override=args.output_scale_override,
        hard_bc_override=args.hard_bc_override,
        hard_bc_scale_override=args.hard_bc_scale_override,
        transpile_optimization_level=args.transpile_optimization_level,
        spatial_smooth=args.spatial_smooth,
        spatial_basis=args.spatial_basis,
    )


if __name__ == "__main__":
    main()
