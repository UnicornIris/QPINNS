"""PDE operators for physics-informed loss computation.

Each operator returns (u_pred, residual) where residual=0 means the PDE is satisfied.
"""

import torch


def helmholtz_operator(model, x1, x2, lam=1.0):
    """Helmholtz: nabla^2 u + lambda*u = f on [-1,1]^2."""
    x1 = x1.requires_grad_(True)
    x2 = x2.requires_grad_(True)

    u = model(torch.cat((x1, x2), dim=1))

    u_x1 = torch.autograd.grad(u, x1, torch.ones_like(u), create_graph=True)[0]
    u_x2 = torch.autograd.grad(u, x2, torch.ones_like(u), create_graph=True)[0]
    u_x1x1 = torch.autograd.grad(u_x1, x1, torch.ones_like(u_x1), create_graph=True)[0]
    u_x2x2 = torch.autograd.grad(u_x2, x2, torch.ones_like(u_x2), create_graph=True)[0]

    residual = u_x1x1 + u_x2x2 + lam * u
    return u, residual


def wave_operator(model, t, x, c=2.0):
    """Wave equation: u_tt - c^2 u_xx = 0."""
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)

    u = model(torch.cat((t, x), dim=1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    residual = u_tt - c ** 2 * u_xx
    return u, residual


def klein_gordon_operator(model, t, x, alpha=-1.0, beta=0.0, gamma=1.0, k=3):
    """Klein-Gordon: u_tt + alpha*u_xx + beta*u + gamma*u^k = 0."""
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)

    u = model(torch.cat((t, x), dim=1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, torch.ones_like(u_t), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    residual = u_tt + alpha * u_xx + beta * u + gamma * u ** k
    return u, residual


def heat_1d_operator(model, t, x, D=0.01 / 3.141592653589793):
    """1D Heat equation: u_t - D*u_xx = 0 on [-1,1] x [0,1].

    Exact solution: u(x,t) = sin(pi*x) * exp(-D*pi^2*t).
    """
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)

    u = model(torch.cat((t, x), dim=1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    residual = u_t - D * u_xx
    return u, residual


def diffusion_operator(model, t, x, y, D=0.01, v_x=1.0, v_y=1.0):
    """2D convection-diffusion: u_t + v_x*u_x + v_y*u_y - D*(u_xx + u_yy) = 0."""
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)

    u = model(torch.cat((t, x, y), dim=1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]

    residual = u_t + v_x * u_x + v_y * u_y - D * (u_xx + u_yy)
    return u, residual


def poisson_operator(model, x1, x2):
    """Poisson equation: -nabla^2 u = f  on [0,2]x[0,1].

    Following Berger et al. (2025) benchmark from the FEniCS example.
    """
    x1 = x1.requires_grad_(True)
    x2 = x2.requires_grad_(True)

    u = model(torch.cat((x1, x2), dim=1))

    u_x1 = torch.autograd.grad(u, x1, torch.ones_like(u), create_graph=True)[0]
    u_x2 = torch.autograd.grad(u, x2, torch.ones_like(u), create_graph=True)[0]
    u_x1x1 = torch.autograd.grad(u_x1, x1, torch.ones_like(u_x1), create_graph=True)[0]
    u_x2x2 = torch.autograd.grad(u_x2, x2, torch.ones_like(u_x2), create_graph=True)[0]

    residual = -(u_x1x1 + u_x2x2)  # = f(x)
    return u, residual


def burgers_operator(model, t, x, nu=0.01 / 3.141592653589793):
    """Burgers equation: u_t + u*u_x - nu*u_xx = 0  on [0,0.95]x[-1,1].

    Following Berger et al. (2025).
    """
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)

    u = model(torch.cat((t, x), dim=1))

    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    residual = u_t + u * u_x - nu * u_xx
    return u, residual


# ---------------------------------------------------------------------------
# Registry for easy lookup from config strings
# ---------------------------------------------------------------------------
OPERATOR_REGISTRY = {
    "helmholtz": helmholtz_operator,
    "wave": wave_operator,
    "klein_gordon": klein_gordon_operator,
    "heat_1d": heat_1d_operator,
    "diffusion": diffusion_operator,
    "poisson": poisson_operator,
    "burgers": burgers_operator,
}
