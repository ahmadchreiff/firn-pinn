from __future__ import annotations

import sys
from pathlib import Path

import torch
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from pinns.config import LossConfig, ModelConfig, TrainingConfig  # noqa: E402
from pinns.core.base_pinn import BasePINN  # noqa: E402
from pinns.models.neural_net import MLP  # noqa: E402
from pinns.problems.firn import FirnProblem  # noqa: E402


@pytest.fixture
def small_pinn() -> BasePINN:
    problem = FirnProblem()
    model = MLP(in_dim=problem.in_dim, out_dim=problem.out_dim, hidden_layers=[8, 8], activation="tanh")
    training = TrainingConfig(epochs=1, learning_rate=1e-3, deterministic=True)
    # Override sampling counts for a tiny test
    training.n_interior = 8  # type: ignore[attr-defined]
    training.n_boundary = 8  # type: ignore[attr-defined]
    training.n_initial = 8  # type: ignore[attr-defined]
    loss_cfg = LossConfig(w_pde=1.0, w_ic=1.0, w_bc=1.0)
    pinn = BasePINN(model=model, problem=problem, training=training, loss_cfg=loss_cfg, device=torch.device("cpu"))
    return pinn


def test_compute_loss_shapes_and_finiteness(small_pinn: BasePINN):
    losses = small_pinn.compute_loss()
    for key in ("loss_total", "loss_pde", "loss_ic", "loss_bc"):
        assert key in losses
        val = losses[key]
        assert torch.is_tensor(val)
        assert val.ndim == 0  # scalar
        assert torch.isfinite(val).item()


def test_single_training_step(small_pinn: BasePINN):
    small_pinn.train(epochs=1, log_every=1)
