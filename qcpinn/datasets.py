"""Dataset generators for PDE benchmarks.

Each function returns (bcs_samplers, residual_sampler) following
the collocation-based approach of Raissi et al. (2019).
"""

import torch
import math
from functools import partial
from typing import Callable, List, Tuple


class Sampler:
    """Generates random collocation points in a rectangular domain."""

    def __init__(self, dim: int, coords: torch.Tensor, func: Callable,
                 name: str = "", device="cpu"):
        self.dim = dim
        self.coords = coords.to(device)
        self.func = func
        self.name = name
        self.device = device

    def sample(self, N: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rand = torch.rand(N, self.dim, dtype=torch.float64, device=self.device)
        x = self.coords[0:1, :] + (self.coords[1:2, :] - self.coords[0:1, :]) * rand
        y = self.func(x)
        return x, y


# ---------------------------------------------------------------------------
# Helmholtz:  nabla^2 u + lambda*u = f   on [-1, 1]^2
# ---------------------------------------------------------------------------

def get_helmholtz_params(config=None):
    """Extract Helmholtz frequency parameters from a config-like object."""
    config = config or {}
    a1 = float(config.get("helmholtz_a1", 1.0))
    a2 = float(config.get("helmholtz_a2", 4.0))
    lam = float(config.get("helmholtz_lambda", 1.0))
    return a1, a2, lam


def _helmholtz_u(x, a1=1, a2=4):
    return torch.sin(a1 * math.pi * x[:, 0:1]) * torch.sin(a2 * math.pi * x[:, 1:2])

def _helmholtz_f(x, a1=1, a2=4, lam=1.0):
    u_xx = -((a1 * math.pi) ** 2) * torch.sin(a1 * math.pi * x[:, 0:1]) * torch.sin(a2 * math.pi * x[:, 1:2])
    u_yy = -((a2 * math.pi) ** 2) * torch.sin(a1 * math.pi * x[:, 0:1]) * torch.sin(a2 * math.pi * x[:, 1:2])
    return u_xx + u_yy + lam * _helmholtz_u(x, a1, a2)


def helmholtz_dataset(device="cpu", a1=1, a2=4, lam=1.0):
    """Returns ([bc_samplers], residual_sampler) for Helmholtz on [-1,1]^2."""
    bc_fn = partial(_helmholtz_u, a1=a1, a2=a2)
    res_fn = partial(_helmholtz_f, a1=a1, a2=a2, lam=lam)
    bc_coords = [
        torch.tensor([[-1., -1.], [1., -1.]], dtype=torch.float64),  # bottom
        torch.tensor([[1., -1.], [1.,  1.]], dtype=torch.float64),   # right
        torch.tensor([[1.,  1.], [-1., 1.]], dtype=torch.float64),   # top
        torch.tensor([[-1., 1.], [-1., -1.]], dtype=torch.float64),  # left
    ]
    bcs = [Sampler(2, c, bc_fn, f"BC_{i}", device) for i, c in enumerate(bc_coords)]
    dom = torch.tensor([[-1., -1.], [1., 1.]], dtype=torch.float64)
    res = Sampler(2, dom, res_fn, "Residual", device)
    return bcs, res


# ---------------------------------------------------------------------------
# Wave:  u_tt - c^2 u_xx = 0   on [0,1] x [0,1]
# ---------------------------------------------------------------------------

def _wave_u(x, c=2.0):
    """Exact: u(t,x) = sin(pi*x)*cos(c*pi*t)."""
    return torch.sin(math.pi * x[:, 1:2]) * torch.cos(c * math.pi * x[:, 0:1])

def _wave_f(x, c=2.0):
    """Forcing = 0 for the standard wave equation."""
    return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)


def wave_dataset(device="cpu"):
    """Returns ([bc_samplers], residual_sampler) for wave on [0,1]^2."""
    bc_coords = [
        torch.tensor([[0., 0.], [1., 0.]], dtype=torch.float64),  # bottom (t-axis, x=0)
        torch.tensor([[0., 1.], [1., 1.]], dtype=torch.float64),  # top    (t-axis, x=1)
        torch.tensor([[0., 0.], [0., 1.]], dtype=torch.float64),  # left   (x-axis, t=0) — IC
    ]
    bcs = [Sampler(2, c, _wave_u, f"BC_{i}", device) for i, c in enumerate(bc_coords)]
    dom = torch.tensor([[0., 0.], [1., 1.]], dtype=torch.float64)
    res = Sampler(2, dom, _wave_f, "Residual", device)
    return bcs, res


# ---------------------------------------------------------------------------
# 1D Heat:  u_t = D*u_xx  on [-1,1] x [0,1]
# Exact: u(x,t) = sin(pi*x) * exp(-D*pi^2*t),  D = 0.01/pi
# ---------------------------------------------------------------------------

_HEAT_D = 0.01 / math.pi

def _heat_1d_u(x):
    """Exact: u(t,x) = sin(pi*x) * exp(-D*pi^2*t)."""
    t, xc = x[:, 0:1], x[:, 1:2]
    return torch.sin(math.pi * xc) * torch.exp(-_HEAT_D * math.pi**2 * t)

def _heat_1d_f(x):
    """Forcing = 0 for the standard heat equation."""
    return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)


def heat_1d_dataset(device="cpu"):
    """Returns ([bc_samplers], residual_sampler) for 1D heat on [0,1]x[-1,1].

    Coordinates: (t, x) with t in [0,1], x in [-1,1].
    BCs: u(t, -1) = 0, u(t, 1) = 0 (Dirichlet)
    IC: u(0, x) = sin(pi*x)
    """
    bc_coords = [
        # Left boundary: x = -1, t in [0,1]
        torch.tensor([[0., -1.], [1., -1.]], dtype=torch.float64),
        # Right boundary: x = 1, t in [0,1]
        torch.tensor([[0., 1.], [1., 1.]], dtype=torch.float64),
        # Initial condition: t = 0, x in [-1,1]
        torch.tensor([[0., -1.], [0., 1.]], dtype=torch.float64),
    ]
    bcs = [Sampler(2, c, _heat_1d_u, f"BC_{i}", device) for i, c in enumerate(bc_coords)]
    dom = torch.tensor([[0., -1.], [1., 1.]], dtype=torch.float64)
    res = Sampler(2, dom, _heat_1d_f, "Residual", device)
    return bcs, res


# ---------------------------------------------------------------------------
# Klein-Gordon:  u_tt + alpha*u_xx + beta*u + gamma*u^k = 0  on [0,1]^2
# ---------------------------------------------------------------------------

def _kg_exact(x):
    """sin(pi*x)*cos(pi*t) — approximate, residual computed numerically."""
    return torch.sin(math.pi * x[:, 1:2]) * torch.cos(math.pi * x[:, 0:1])

def _kg_forcing(x, alpha=-1.0, beta=0.0, gamma=1.0, k=3):
    u = _kg_exact(x)
    t, xc = x[:, 0:1], x[:, 1:2]
    u_tt = -(math.pi ** 2) * torch.sin(math.pi * xc) * torch.cos(math.pi * t)
    u_xx = -(math.pi ** 2) * torch.sin(math.pi * xc) * torch.cos(math.pi * t)
    return u_tt + alpha * u_xx + beta * u + gamma * u ** k


def klein_gordon_dataset(device="cpu"):
    bc_coords = [
        torch.tensor([[0., 0.], [1., 0.]], dtype=torch.float64),
        torch.tensor([[0., 1.], [1., 1.]], dtype=torch.float64),
        torch.tensor([[0., 0.], [0., 1.]], dtype=torch.float64),
    ]
    bcs = [Sampler(2, c, _kg_exact, f"BC_{i}", device) for i, c in enumerate(bc_coords)]
    dom = torch.tensor([[0., 0.], [1., 1.]], dtype=torch.float64)
    res = Sampler(2, dom, _kg_forcing, "Residual", device)
    return bcs, res


# ---------------------------------------------------------------------------
# Poisson:  -nabla^2 u = f  on [0,2]x[0,1]
# From Berger et al. (2025), FEniCS benchmark
# ---------------------------------------------------------------------------

def _poisson_f(x):
    """Source term f = 10*exp(-((x-0.5)^2 + (y-0.5)^2)/0.02)."""
    return 10.0 * torch.exp(-((x[:, 0:1] - 0.5)**2 + (x[:, 1:2] - 0.5)**2) / 0.02)

def _poisson_bc_dirichlet(x):
    """u = 0 on Dirichlet boundary (left x=0, right x=2)."""
    return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)

def _poisson_bc_neumann(x):
    """g = sin(5*x) on Neumann boundary (top y=1, bottom y=0)."""
    return torch.sin(5.0 * x[:, 0:1])


def poisson_dataset(device="cpu"):
    """Returns ([bc_samplers], residual_sampler) for Poisson on [0,2]x[0,1].

    BCs: Dirichlet u=0 on left (x=0) and right (x=2)
         Neumann du/dn = sin(5x) on top (y=1) and bottom (y=0)
    """
    bc_coords = [
        # Left: x=0, y in [0,1] (Dirichlet)
        torch.tensor([[0., 0.], [0., 1.]], dtype=torch.float64),
        # Right: x=2, y in [0,1] (Dirichlet)
        torch.tensor([[2., 0.], [2., 1.]], dtype=torch.float64),
    ]
    bcs = [Sampler(2, c, _poisson_bc_dirichlet, f"BC_Dir_{i}", device) for i, c in enumerate(bc_coords)]
    # Note: Neumann BCs are harder to enforce; for now use Dirichlet only
    dom = torch.tensor([[0., 0.], [2., 1.]], dtype=torch.float64)
    res = Sampler(2, dom, _poisson_f, "Residual", device)
    return bcs, res


# ---------------------------------------------------------------------------
# Burgers:  u_t + u*u_x = nu*u_xx   on [-1,1]x[0,0.95]
# From Berger et al. (2025)
# ---------------------------------------------------------------------------

def _burgers_u(x, nu=0.01 / math.pi):
    """Exact: u(t,x) = -sin(pi*x).  IC at t=0."""
    return -torch.sin(math.pi * x[:, 1:2])

def _burgers_bc_left(x):
    """u(t, -1) = 0."""
    return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)

def _burgers_bc_right(x):
    """u(t, 1) = 0."""
    return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)

def _burgers_ic(x):
    """u(0, x) = -sin(pi*x)."""
    return -torch.sin(math.pi * x[:, 1:2])

def _burgers_f(x):
    """Forcing = 0."""
    return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)


def burgers_dataset(device="cpu"):
    """Returns ([bc_samplers], residual_sampler) for Burgers on [0,0.95]x[-1,1]."""
    bc_coords = [
        # Left boundary: x = -1, t in [0,0.95]
        torch.tensor([[0., -1.], [0.95, -1.]], dtype=torch.float64),
        # Right boundary: x = 1, t in [0,0.95]
        torch.tensor([[0., 1.], [0.95, 1.]], dtype=torch.float64),
        # Initial condition: t = 0, x in [-1,1]
        torch.tensor([[0., -1.], [0., 1.]], dtype=torch.float64),
    ]
    bcs = [
        Sampler(2, bc_coords[0], _burgers_bc_left, "BC_left", device),
        Sampler(2, bc_coords[1], _burgers_bc_right, "BC_right", device),
        Sampler(2, bc_coords[2], _burgers_ic, "IC", device),
    ]
    dom = torch.tensor([[0., -1.], [0.95, 1.]], dtype=torch.float64)
    res = Sampler(2, dom, _burgers_f, "Residual", device)
    return bcs, res


# Dataset registry
DATASET_REGISTRY = {
    "helmholtz": (helmholtz_dataset, 2, 1, [-1., -1.], [1., 1.]),
    "wave":      (wave_dataset, 2, 1, [0., 0.], [1., 1.]),
    "klein_gordon": (klein_gordon_dataset, 2, 1, [0., 0.], [1., 1.]),
    "heat_1d":   (heat_1d_dataset, 2, 1, [0., -1.], [1., 1.]),
    "poisson":   (poisson_dataset, 2, 1, [0., 0.], [2., 1.]),
    "burgers":   (burgers_dataset, 2, 1, [0., -1.], [0.95, 1.]),
}

# Public aliases for exact solutions (used by evaluation.py)
helmholtz_exact_u = _helmholtz_u
helmholtz_exact_f = _helmholtz_f
wave_exact_u = _wave_u
kg_exact_u = _kg_exact
heat_1d_exact_u = _heat_1d_u
burgers_exact_u = _burgers_u
