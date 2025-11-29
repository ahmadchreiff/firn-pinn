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
    D = problem.diffusivity(z)
    assert D.shape == z.shape
    assert torch.isfinite(D).all()
    assert problem.f_const > 0 and problem.adv_velocity > 0 and problem.sink_rate > 0
    t = torch.linspace(problem.t_min, problem.t_max, 3).reshape(-1, 1)
    rho_top = problem.rho_atm(t)
    assert rho_top.shape == t.shape
    assert torch.isfinite(rho_top).all()


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
