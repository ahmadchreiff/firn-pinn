from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pinns.problems.firn import FirnProblem  # noqa: E402


def test_sample_interior_shape_and_bounds():
    problem = FirnProblem()
    n = 32
    pts = problem.sample_interior(n)
    assert pts.shape == (n, 2)
    t_vals, z_vals = pts[:, 0], pts[:, 1]
    assert torch.all(t_vals >= problem.t_min) and torch.all(t_vals <= problem.t_max)
    assert torch.all(z_vals >= problem.z_min) and torch.all(z_vals <= problem.z_max)


def test_sample_initial_time_is_fixed():
    problem = FirnProblem()
    n = 16
    pts = problem.sample_initial(n)
    assert pts.shape == (n, 2)
    t_vals = pts[:, 0]
    assert torch.allclose(t_vals, torch.full_like(t_vals, problem.t_min), atol=1e-6)


def test_sample_boundary_z_is_on_bounds():
    problem = FirnProblem()
    n = 20
    pts = problem.sample_boundary(n)
    assert pts.shape == (n, 2)
    z_vals = pts[:, 1]
    z_min = torch.as_tensor(problem.z_min, dtype=pts.dtype)
    z_max = torch.as_tensor(problem.z_max, dtype=pts.dtype)
    is_min = torch.isclose(z_vals, z_min, atol=1e-6)
    is_max = torch.isclose(z_vals, z_max, atol=1e-6)
    assert torch.all(is_min | is_max)
    # ensure both boundaries present for reasonable n
    assert is_min.any() and is_max.any()
