#!/usr/bin/env python3
"""Comprehensive test suite for the qcpinn package.

Run: conda run -n qcpinn-modern python tests/test_all.py
"""

import sys
import time
import traceback
import torch
import numpy as np

sys.path.insert(0, ".")

PASSED = 0
FAILED = 0

def test(name):
    def decorator(fn):
        def wrapper():
            global PASSED, FAILED
            try:
                t0 = time.time()
                fn()
                dt = time.time() - t0
                print(f"  PASS  {name} ({dt:.2f}s)")
                PASSED += 1
            except Exception as e:
                print(f"  FAIL  {name}: {e}")
                traceback.print_exc()
                FAILED += 1
        return wrapper
    return decorator


# =========================================================================
# Embedding tests
# =========================================================================

@test("TrainableEmbedding: output shape")
def test_te_shape():
    from qcpinn.embedding import TrainableEmbedding
    te = TrainableEmbedding(in_dim=2, n_qubits=4, hidden_layers=2, width=32)
    x = torch.randn(8, 2, dtype=torch.float64)
    out = te(x)
    assert out.shape == (8, 4), f"Expected (8,4), got {out.shape}"

@test("TrainableEmbedding: phi(x)*x structure")
def test_te_bounded():
    from qcpinn.embedding import TrainableEmbedding
    te = TrainableEmbedding(in_dim=2, n_qubits=4)
    x = torch.randn(100, 2, dtype=torch.float64)
    out = te(x)
    # Verify: output = phi(x) * x_cycled (Berger Eq. 11)
    phi = te.net(x)
    x_cycled = x[:, [0, 1, 0, 1]]
    assert torch.allclose(out, phi * x_cycled), "Output does not match phi(x)*x structure"
    # Output is finite
    assert torch.isfinite(out).all(), "Output contains non-finite values"

@test("TrainableEmbedding: gradients flow")
def test_te_gradients():
    from qcpinn.embedding import TrainableEmbedding
    te = TrainableEmbedding(in_dim=2, n_qubits=4)
    x = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
    out = te(x)
    out.sum().backward()
    assert x.grad is not None, "No gradient on input"
    for p in te.parameters():
        assert p.grad is not None, "No gradient on params"


# =========================================================================
# Quantum circuit tests
# =========================================================================

@test("QuantumLayer: angle encoding forward")
def test_ql_angle():
    from qcpinn.circuits import QuantumLayer
    config = {"num_qubits": 2, "num_quantum_layers": 1, "q_ansatz": "hea",
              "encoding": "angle", "noise_strength": 0.0}
    ql = QuantumLayer(config)
    x = torch.randn(3, 2, dtype=torch.float64) * 0.5
    out = ql(x)
    assert out.shape == (3, 2), f"Expected (3,2), got {out.shape}"
    assert torch.all(out.abs() <= 1.0), "Expectation values outside [-1,1]"

@test("QuantumLayer: TE encoding forward")
def test_ql_te():
    from qcpinn.circuits import QuantumLayer
    config = {"num_qubits": 2, "num_quantum_layers": 1, "q_ansatz": "hea",
              "encoding": "angle", "noise_strength": 0.0}
    ql = QuantumLayer(config)
    angles = torch.randn(3, 2, dtype=torch.float64)
    out = ql.forward_te(angles)
    assert out.shape == (3, 2)

@test("QuantumLayer: noisy circuit runs")
def test_ql_noisy():
    from qcpinn.circuits import QuantumLayer
    config = {"num_qubits": 2, "num_quantum_layers": 1, "q_ansatz": "hea",
              "encoding": "angle", "noise_strength": 0.05}
    ql = QuantumLayer(config)
    x = torch.randn(3, 2, dtype=torch.float64) * 0.5
    out = ql(x)
    assert out.shape == (3, 2)

@test("QuantumLayer: all ansatze work")
def test_all_ansatze():
    from qcpinn.circuits import QuantumLayer, ANSATZ_REGISTRY
    for name in ANSATZ_REGISTRY:
        config = {"num_qubits": 4, "num_quantum_layers": 1, "q_ansatz": name,
                  "encoding": "angle", "noise_strength": 0.0}
        ql = QuantumLayer(config)
        x = torch.randn(2, 4, dtype=torch.float64) * 0.5
        out = ql(x)
        assert out.shape == (2, 4), f"Ansatz {name}: Expected (2,4), got {out.shape}"


# =========================================================================
# Solver tests
# =========================================================================

@test("QCPINNSolver: classical mode")
def test_solver_classical():
    from qcpinn.solver import QCPINNSolver
    config = {"problem": "helmholtz", "mode": "classical", "input_dim": 2,
              "output_dim": 1, "num_qubits": 2, "num_quantum_layers": 1,
              "hidden_dim": 20, "q_ansatz": "hea", "domain_lo": [-1, -1],
              "domain_hi": [1, 1]}
    m = QCPINNSolver(config)
    x = torch.randn(5, 2, dtype=torch.float64)
    u = m(x)
    assert u.shape == (5, 1)
    u.sum().backward()

@test("QCPINNSolver: baseline mode")
def test_solver_baseline():
    from qcpinn.solver import QCPINNSolver
    config = {"problem": "helmholtz", "mode": "baseline", "input_dim": 2,
              "output_dim": 1, "num_qubits": 2, "num_quantum_layers": 1,
              "hidden_dim": 20, "q_ansatz": "hea", "domain_lo": [-1, -1],
              "domain_hi": [1, 1], "noise_strength": 0.0}
    m = QCPINNSolver(config)
    x = torch.randn(3, 2, dtype=torch.float64)
    u = m(x)
    assert u.shape == (3, 1)

@test("QCPINNSolver: TE mode - no double counting params")
def test_solver_te_params():
    from qcpinn.solver import QCPINNSolver
    config = {"problem": "helmholtz", "mode": "te", "input_dim": 2,
              "output_dim": 1, "num_qubits": 2, "num_quantum_layers": 1,
              "hidden_dim": 20, "q_ansatz": "hea", "domain_lo": [-1, -1],
              "domain_hi": [1, 1], "noise_strength": 0.0,
              "te_hidden_layers": 2, "te_width": 32}
    m = QCPINNSolver(config)
    counts = m.count_parameters()
    # Verify no double-counting: total must equal sum of components
    component_sum = sum(v for k, v in counts.items() if k != "total")
    assert counts["total"] == component_sum, \
        f"Param count mismatch: total={counts['total']} vs sum={component_sum}"

@test("QCPINNSolver: save and load roundtrip")
def test_solver_save_load():
    from qcpinn.solver import QCPINNSolver
    import tempfile, os
    config = {"problem": "helmholtz", "mode": "te", "input_dim": 2,
              "output_dim": 1, "num_qubits": 2, "num_quantum_layers": 1,
              "hidden_dim": 20, "q_ansatz": "hea", "domain_lo": [-1, -1],
              "domain_hi": [1, 1], "noise_strength": 0.0,
              "te_hidden_layers": 2, "te_width": 32}
    m = QCPINNSolver(config)
    x = torch.randn(3, 2, dtype=torch.float64)
    with torch.no_grad():
        u1 = m(x)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pth")
        m.save_state(path)
        m2 = QCPINNSolver.load_state(path)
        with torch.no_grad():
            u2 = m2(x)
    
    assert torch.allclose(u1, u2, atol=1e-10), "Save/load changed predictions"


# =========================================================================
# PDE operator tests
# =========================================================================

@test("Helmholtz operator: exact solution has zero residual")
def test_helmholtz_pde():
    from qcpinn.pde import helmholtz_operator
    from qcpinn.datasets import helmholtz_exact_u, helmholtz_exact_f
    
    class ExactModel(torch.nn.Module):
        def forward(self, x):
            return helmholtz_exact_u(x)
    
    model = ExactModel()
    x1 = torch.randn(50, 1, dtype=torch.float64, requires_grad=True) * 0.5
    x2 = torch.randn(50, 1, dtype=torch.float64, requires_grad=True) * 0.5
    u, r = helmholtz_operator(model, x1, x2)
    
    f_exact = helmholtz_exact_f(torch.cat([x1, x2], dim=1))
    residual_err = (r - f_exact).abs().max()
    assert residual_err < 1e-6, f"Residual error {residual_err:.2e} too large"


# =========================================================================
# Dataset tests
# =========================================================================

@test("Datasets: all registered datasets work")
def test_datasets():
    from qcpinn.datasets import DATASET_REGISTRY
    for name, (ds_fn, in_dim, out_dim, lo, hi) in DATASET_REGISTRY.items():
        bcs, res = ds_fn("cpu")
        X_r, f_r = res.sample(16)
        assert X_r.shape == (16, in_dim), f"{name} residual X shape wrong"
        assert f_r.shape[0] == 16, f"{name} residual f shape wrong"
        for bc in bcs:
            X_bc, u_bc = bc.sample(8)
            assert X_bc.shape == (8, in_dim), f"{name} BC X shape wrong"


# =========================================================================
# Training integration test
# =========================================================================

@test("Trainer: 20 epochs of Helmholtz TE converges without error")
def test_training_integration():
    import tempfile
    from qcpinn.solver import QCPINNSolver
    from qcpinn.trainer import Trainer
    
    config = {"problem": "helmholtz", "mode": "te", "input_dim": 2,
              "output_dim": 1, "num_qubits": 2, "num_quantum_layers": 1,
              "hidden_dim": 20, "q_ansatz": "hea", "encoding": "angle",
              "epochs": 20, "batch_size": 32, "lr": 5e-4, "optimizer": "adam",
              "bc_weight": 10.0, "grad_clip": 1.0, "print_every": 10,
              "noise_strength": 0.0, "shots": None, "qml_device": "default.qubit",
              "domain_lo": [-1, -1], "domain_hi": [1, 1],
              "te_hidden_layers": 2, "te_width": 32}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model = QCPINNSolver(config)
        trainer = Trainer(model, config, tmpdir)
        best = trainer.train()
        assert best > 0, "Best loss should be positive"
        assert len(model.loss_history) == 21, f"Expected 21 history entries, got {len(model.loss_history)}"
        assert not any(np.isnan(model.loss_history)), "NaN in loss history"


@test("Trainer: noisy training doesn't crash")
def test_noisy_training():
    import tempfile
    from qcpinn.solver import QCPINNSolver
    from qcpinn.trainer import Trainer
    
    config = {"problem": "helmholtz", "mode": "te", "input_dim": 2,
              "output_dim": 1, "num_qubits": 2, "num_quantum_layers": 1,
              "hidden_dim": 20, "q_ansatz": "hea", "encoding": "angle",
              "epochs": 10, "batch_size": 32, "lr": 5e-4, "optimizer": "adam",
              "bc_weight": 10.0, "grad_clip": 1.0, "print_every": 5,
              "noise_strength": 0.01, "shots": None, "qml_device": "default.qubit",
              "domain_lo": [-1, -1], "domain_hi": [1, 1],
              "te_hidden_layers": 2, "te_width": 32}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model = QCPINNSolver(config)
        trainer = Trainer(model, config, tmpdir)
        best = trainer.train()
        assert best > 0


# =========================================================================
# Evaluation tests
# =========================================================================

@test("Evaluation: evaluate_helmholtz returns all metrics")
def test_eval_helmholtz():
    from qcpinn.solver import QCPINNSolver
    from qcpinn.evaluation import evaluate_helmholtz
    
    config = {"problem": "helmholtz", "mode": "classical", "input_dim": 2,
              "output_dim": 1, "num_qubits": 2, "num_quantum_layers": 1,
              "hidden_dim": 20, "q_ansatz": "hea", "domain_lo": [-1, -1],
              "domain_hi": [1, 1]}
    m = QCPINNSolver(config)
    metrics = evaluate_helmholtz(m, grid_points=10)
    required = ["rel_l2_u", "rel_l2_f", "mse_u", "max_err_u", "u_star", "u_pred", "grid_shape"]
    for k in required:
        assert k in metrics, f"Missing key: {k}"


# =========================================================================
# Run all
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("QCPINN Test Suite")
    print("=" * 60)
    
    t0 = time.time()
    
    # Embedding
    test_te_shape()
    test_te_bounded()
    test_te_gradients()
    
    # Circuits
    test_ql_angle()
    test_ql_te()
    test_ql_noisy()
    test_all_ansatze()
    
    # Solver
    test_solver_classical()
    test_solver_baseline()
    test_solver_te_params()
    test_solver_save_load()
    
    # PDE
    test_helmholtz_pde()
    
    # Datasets
    test_datasets()
    
    # Training integration
    test_training_integration()
    test_noisy_training()
    
    # Evaluation
    test_eval_helmholtz()
    
    dt = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Results: {PASSED} passed, {FAILED} failed ({dt:.1f}s)")
    print(f"{'='*60}")
    
    sys.exit(1 if FAILED > 0 else 0)
