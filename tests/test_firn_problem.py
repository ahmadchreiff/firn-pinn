from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pinns.firn import FirnProblem  # noqa: E402


@pytest.fixture
def problem() -> FirnProblem:
    return FirnProblem()


def test_coefficients_shapes_and_finiteness(problem: FirnProblem):
    z = torch.linspace(problem.z_min, problem.z_max, 5).reshape(-1, 1)
    coeffs = [
        problem.f(z),
        problem.v(z),
        problem.w_air(z),
        problem.tau(z),
        problem.lambda_(z),
        problem.D_tilde(z),
    ]
    for c in coeffs:
        assert c.shape == z.shape
        assert torch.isfinite(c).all()


def test_pde_residual_shape_and_finiteness(problem: FirnProblem):
    n = 8
    x_int = problem.sample_interior(n)
    rho = torch.randn(n, 1)
    rho_t = torch.randn(n, 1)
    rho_z = torch.randn(n, 1)
    rho_zz = torch.randn(n, 1)

    r = problem.pde_residual(x_int, rho, rho_t, rho_z, rho_zz)
    assert r.shape == (n, 1)
    assert torch.isfinite(r).all()


def test_initial_residual_shape_and_finiteness(problem: FirnProblem):
    n = 6
    x_ic = problem.sample_initial(n)
    rho = torch.randn(n, 1)
    r = problem.initial_residual(x_ic, rho)
    assert r.shape == (n, 1)
    assert torch.isfinite(r).all()


def test_boundary_residual_shape_and_finiteness(problem: FirnProblem):
    n = 10
    x_b = problem.sample_boundary(n)
    rho = torch.randn(n, 1)
    rho_z = torch.randn(n, 1)
    r = problem.boundary_residual(x_b, rho, rho_z)
    assert r.shape == (n, 1)
    assert torch.isfinite(r).all()
