from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import logging

import torch
from torch import Tensor, nn

from pinns.config import Config, LossConfig, TrainingConfig
from pinns.problems.firn import FirnProblem
from pinns.utils.logging import get_logger, log_metrics

__all__ = ["BasePINN"]


class BasePINN:
    """
    Minimal PINN trainer for the rescaled firn problem.
    Builds PDE/IC/BC losses, runs a simple training loop, and keeps model/problem config.
    """

    def __init__(
        self,
        model: nn.Module,
        problem: FirnProblem,
        training: TrainingConfig,
        loss_cfg: LossConfig,
        device: Optional[torch.device | str] = None,
        logger: Optional[logging.Logger] = None,
        run_dir: Optional[Path | str] = None,
    ) -> None:
        self.model = model
        self.problem = problem
        self.training = training
        self.loss_cfg = loss_cfg
        self.run_dir = Path(run_dir) if run_dir is not None else None

        if device is not None:
            self.device = torch.device(device)
        else:
            training_device = getattr(training, "device", None)
            if training_device is not None:
                self.device = torch.device(training_device)
            else:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        lr = getattr(training, "learning_rate", getattr(training, "lr", 1e-3))
        opt_name = str(getattr(training, "optimizer", "adam")).lower()
        if opt_name == "lbfgs":
            self._is_lbfgs = True
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), lr=lr, max_iter=20, line_search_fn="strong_wolfe")
        else:
            self._is_lbfgs = False
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = None  # placeholder for optional schedulers

        self.logger = logger if logger is not None else get_logger("pinn")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _prepare_inputs(self, x: Tensor) -> Tensor:
        """Move inputs to device and enable gradient tracking."""
        x = x.to(device=self.device)
        x.requires_grad_(True)
        return x

    def _compute_derivatives(self, x: Tensor, u: Tensor) -> Dict[str, Tensor]:
        """
        Compute first and second derivatives wrt t_hat and z_hat.
        """
        grads = torch.autograd.grad(
            outputs=u.sum(),
            inputs=x,
            create_graph=True,
        )[0]
        u_t = grads[:, 0:1]
        u_z = grads[:, 1:2]

        grads2 = torch.autograd.grad(
            outputs=u_z.sum(),
            inputs=x,
            create_graph=True,
        )[0]
        u_zz = grads2[:, 1:2]

        return {"u_t": u_t, "u_z": u_z, "u_zz": u_zz}

    # ------------------------------------------------------------------
    # Loss components
    # ------------------------------------------------------------------
    def _pde_loss(self) -> Tensor:
        n_int = self.training.n_interior
        x_int = self.problem.sample_interior(n_int, device=self.device)
        x_int = self._prepare_inputs(x_int)

        u_int = self.model(x_int)
        derivs = self._compute_derivatives(x_int, u_int)

        r_pde = self.problem.pde_residual(
            inputs=x_int,
            rho=u_int,
            rho_t=derivs["u_t"],
            rho_z=derivs["u_z"],
            rho_zz=derivs["u_zz"],
        )
        return torch.mean(r_pde**2)

    def _ic_loss(self) -> Tensor:
        n_init = self.training.n_initial
        x_ic = self.problem.sample_initial(n_init, device=self.device)
        x_ic = self._prepare_inputs(x_ic)

        u_ic = self.model(x_ic)
        r_ic = self.problem.initial_residual(x_ic, u_ic)
        return torch.mean(r_ic**2)

    def _bc_loss(self) -> Tensor:
        n_bnd = self.training.n_boundary
        x_b = self.problem.sample_boundary(n_bnd, device=self.device)
        x_b = self._prepare_inputs(x_b)

        u_b = self.model(x_b)
        derivs_b = self._compute_derivatives(x_b, u_b)
        u_z_b = derivs_b["u_z"]

        r_bc = self.problem.boundary_residual(x_b, u_b, u_z_b)
        return torch.mean(r_bc**2)

    def compute_loss(self) -> Dict[str, Tensor]:
        """Compute weighted total loss and individual components."""
        loss_pde = self._pde_loss()
        loss_ic = self._ic_loss()
        loss_bc = self._bc_loss()

        w_pde = getattr(self.loss_cfg, "w_pde", 1.0)
        w_ic = getattr(self.loss_cfg, "w_ic", 1.0)
        w_bc = getattr(self.loss_cfg, "w_bc", 1.0)

        loss_total = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc

        return {
            "loss_total": loss_total,
            "loss_pde": loss_pde,
            "loss_ic": loss_ic,
            "loss_bc": loss_bc,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, epochs: Optional[int] = None, log_every: int = 100) -> None:
        num_epochs = epochs if epochs is not None else self.training.epochs

        for epoch in range(1, num_epochs + 1):
            self.model.train()

            if self._is_lbfgs:
                last_losses: Dict[str, Tensor] = {}

                def closure() -> Tensor:
                    self.optimizer.zero_grad()
                    losses_inner = self.compute_loss()
                    last_losses.update({k: v for k, v in losses_inner.items()})
                    losses_inner["loss_total"].backward()
                    return losses_inner["loss_total"]

                self.optimizer.step(closure)
                losses = last_losses if last_losses else self.compute_loss()
            else:
                self.optimizer.zero_grad()
                losses = self.compute_loss()
                loss_total = losses["loss_total"]
                loss_total.backward()
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            if epoch % log_every == 0 or epoch == 1 or epoch == num_epochs:
                loss_vals = {k: v.detach().item() for k, v in losses.items()}
                self.logger.info(
                    "Epoch %d | loss=%.4e | pde=%.4e | ic=%.4e | bc=%.4e",
                    epoch,
                    loss_vals["loss_total"],
                    loss_vals["loss_pde"],
                    loss_vals["loss_ic"],
                    loss_vals["loss_bc"],
                )
                if self.run_dir is not None:
                    log_metrics(loss_vals, run_dir=self.run_dir, filename="metrics.csv")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        self.model.eval()
        x = x.to(device=self.device)
        return self.model(x)
