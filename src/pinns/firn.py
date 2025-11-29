from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor

from pinns.config import ProblemConfig
from pinns.utils.sampling import (
    sample_boundary as sample_boundary_points,
    sample_interior as sample_interior_points,
    sample_initial as sample_initial_points,
)

__all__ = ["FirnProblem"]


class FirnProblem:
    """
    Rescaled firn gas-diffusion problem

    Inputs are (t_hat, z_hat) in [t_min, t_max] x [z_min, z_max].
    Outputs are rho_hat (dimensionless gas density).
    The class provides domain info, sampling utilities, coefficient profiles,
    and residuals for PDE, IC, and BCs.
    """

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
        *,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        Te: Optional[float] = None,
        zF: Optional[float] = None,
        f_const: Optional[float] = None,
        M_alpha: Optional[float] = None,
        g: Optional[float] = None,
        R: Optional[float] = None,
        T: Optional[float] = None,
        v_const: Optional[float] = None,
        w_air_const: Optional[float] = None,
        tau_const: Optional[float] = None,
        lambda_const: Optional[float] = None,
        D_a0: Optional[float] = None,
        D_a1: Optional[float] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        params: Dict[str, Any] = getattr(config, "parameters", {}) if config is not None else {}

        def _resolve_bound(value: Optional[float], default_cfg: float, fallback: float) -> float:
            if value is not None:
                return float(value)
            if config is not None:
                return float(default_cfg)
            return float(fallback)

        def _resolve_param(name: str, value: Optional[float], fallback: float) -> float:
            if value is not None:
                return float(value)
            if name in params:
                return float(params[name])
            return float(fallback)

        def _resolve_dtype(dt: Optional[Any]) -> torch.dtype:
            if dt is None:
                return torch.float32
            if isinstance(dt, torch.dtype):
                return dt
            key = str(dt).lower()
            if key in ("float32", "float", "fp32"):
                return torch.float32
            if key in ("float64", "double", "fp64"):
                return torch.float64
            if key in ("float16", "half", "fp16"):
                return torch.float16
            if key in ("bfloat16", "bf16"):
                return torch.bfloat16
            raise ValueError(f"Unsupported dtype: {dt}")

        # Bounds and sampling counts
        self.t_min = _resolve_bound(t_min, getattr(config, "t_min", 0.0), 0.0)
        self.t_max = _resolve_bound(t_max, getattr(config, "t_max", 1.0), 1.0)
        self.z_min = _resolve_bound(z_min, getattr(config, "z_min", 0.0), 0.0)
        self.z_max = _resolve_bound(z_max, getattr(config, "z_max", 1.0), 1.0)

        # Physical / model parameters (Table 2 defaults)
        self.Te = _resolve_param("Te", Te, 1.0)
        self.zF = _resolve_param("zF", zF, 1.0)
        self.f_const = _resolve_param("f_const", f_const, 0.2)
        self.M_alpha = _resolve_param("M_alpha", M_alpha, 0.04)
        self.g = _resolve_param("g", g, 9.8)
        self.R = _resolve_param("R", R, 8.314)
        self.T = _resolve_param("T", T, 260.0)
        self.v_const = _resolve_param("v_const", v_const, 200.0)
        self.w_air_const = _resolve_param("w_air_const", w_air_const, 485.0)
        self.tau_const = _resolve_param("tau_const", tau_const, 10.0)
        self.lambda_const = _resolve_param("lambda_const", lambda_const, 0.03)
        self.D_a0 = _resolve_param("D_a0", D_a0, 200.0)
        self.D_a1 = _resolve_param("D_a1", D_a1, 199.98)

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = _resolve_dtype(dtype)

        self.in_dim = 2  # (t_hat, z_hat)
        self.out_dim = 1  # rho_hat

        self.c_grav = self.M_alpha * self.g / (self.R * self.T)
        self.config = config

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------
    def bounds(self) -> list[list[float]]:
        """Return domain bounds as [[t_min, t_max], [z_min, z_max]]."""
        return [[self.t_min, self.t_max], [self.z_min, self.z_max]]

    @staticmethod
    def split_inputs(x: Tensor) -> tuple[Tensor, Tensor]:
        """Split concatenated inputs into (t_hat, z_hat)."""
        return x[:, 0:1], x[:, 1:2]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_interior(self, num_points: int, strategy: str = "sobol", device: Optional[torch.device | str] = None) -> Tensor:
        """Sample interior collocation points."""
        dev = device or self.device
        pts = sample_interior_points(self.bounds(), num_points, strategy=strategy, device=dev)
        return pts.to(dtype=self.dtype)

    def sample_initial(self, num_points: int, strategy: str = "sobol", device: Optional[torch.device | str] = None) -> Tensor:
        """Sample points on the initial surface t = t_min."""
        dev = device or self.device
        pts = sample_initial_points(
            bounds=self.bounds(),
            num_points=num_points,
            time_dim=0,
            time_value=self.t_min,
            strategy=strategy,
            device=dev,
        )
        return pts.to(dtype=self.dtype)

    def sample_boundary(self, num_points: int, strategy: str = "sobol", device: Optional[torch.device | str] = None) -> Tensor:
        """Sample points on z-boundaries (top and bottom)."""
        dev = device or self.device
        num_top = num_points // 2
        num_bottom = num_points - num_top

        top = sample_boundary_points(
            bounds=self.bounds(),
            num_points=num_top,
            fixed_coords=[1],
            fixed_values=[self.z_min],
            strategy=strategy,
            device=dev,
        )
        bottom = sample_boundary_points(
            bounds=self.bounds(),
            num_points=num_bottom,
            fixed_coords=[1],
            fixed_values=[self.z_max],
            strategy=strategy,
            device=dev,
        )
        pts = torch.cat([top, bottom], dim=0)
        return pts.to(dtype=self.dtype)

    # ------------------------------------------------------------------
    # Coefficient profiles (defaults from Table 2; replace with data-driven as needed)
    # ------------------------------------------------------------------
    def f(self, z_hat: Tensor) -> Tensor:
        """Open porosity factor f(z_hat)."""
        return torch.full_like(z_hat, self.f_const)

    def v(self, z_hat: Tensor) -> Tensor:
        """Vertical firn velocity v(z_hat)."""
        return torch.full_like(z_hat, self.v_const)

    def w_air(self, z_hat: Tensor) -> Tensor:
        """Air velocity w_air(z_hat)."""
        return torch.full_like(z_hat, self.w_air_const)

    def tau(self, z_hat: Tensor) -> Tensor:
        """Sink term tau(z_hat)."""
        return torch.full_like(z_hat, self.tau_const)

    def lambda_(self, z_hat: Tensor) -> Tensor:
        """Sink term lambda(z_hat)."""
        return torch.full_like(z_hat, self.lambda_const)

    def D_tilde(self, z_hat: Tensor) -> Tensor:
        """Rescaled diffusivity D_tilde(z_hat)."""
        return self.D_a0 - self.D_a1 * z_hat

    # ------------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------------
    def pde_residual(
        self,
        inputs: Tensor,
        rho: Tensor,
        rho_t: Tensor,
        rho_z: Tensor,
        rho_zz: Tensor,
    ) -> Tensor:
        """
        Compute PDE residual at interior points (Eq. 33).
        """
        _, z_hat = self.split_inputs(inputs)

        f_val = self.f(z_hat)
        v_val = self.v(z_hat)
        w_val = self.w_air(z_hat)
        tau_val = self.tau(z_hat)
        lambda_val = self.lambda_(z_hat)
        D_val = self.D_tilde(z_hat)

        U_val = v_val + w_val
        f_z_val = 0.0
        U_z_val = 0.0
        D_z_val = -self.D_a1

        term_time = (1.0 / self.Te) * f_val * rho_t
        term_adv = (1.0 / self.zF) * (rho_z * f_val * U_val + rho * (f_z_val * U_val + f_val * U_z_val))
        term_react = rho * (tau_val + lambda_val)

        phi = (1.0 / self.zF) * rho_z - rho * self.c_grav
        phi_z = (1.0 / self.zF) * rho_zz - rho_z * self.c_grav
        term_diff_rhs = (1.0 / self.zF) * (D_z_val * phi + D_val * phi_z)

        return term_time + term_adv + term_react - term_diff_rhs

    def initial_residual(self, inputs: Tensor, rho: Tensor) -> Tensor:
        """Residual for the initial condition rho_hat(z_hat, 0) = 0."""
        return rho

    def rho_atm(self, t_hat: Tensor) -> Tensor:
        """
        Atmospheric concentration at the top boundary.
        Default: 2 * (Te * t_hat) ** 0.25 for t in [0, 1] (Table 2).
        """
        return 2.0 * (self.Te * t_hat).pow(0.25)

    def boundary_residual(self, inputs: Tensor, rho: Tensor, rho_z: Tensor) -> Tensor:
        """
        Residual for boundary conditions:
        - Top (z = z_min): Dirichlet rho = rho_atm(t_hat)
        - Bottom (z = z_max): Flux D_tilde(1) * (1/zF * rho_z - c_grav * rho) = 0
        """
        t_hat, z_hat = self.split_inputs(inputs)
        residual = torch.zeros_like(rho)

        z_min_t = torch.as_tensor(self.z_min, device=z_hat.device, dtype=z_hat.dtype)
        z_max_t = torch.as_tensor(self.z_max, device=z_hat.device, dtype=z_hat.dtype)

        is_top = torch.isclose(z_hat, z_min_t, atol=1e-6)
        is_bottom = torch.isclose(z_hat, z_max_t, atol=1e-6)

        if is_top.any():
            rho_top = rho[is_top]
            t_top = t_hat[is_top]
            target_top = self.rho_atm(t_top)
            residual[is_top] = rho_top - target_top

        if is_bottom.any():
            rho_bottom = rho[is_bottom]
            rho_z_bottom = rho_z[is_bottom]
            z_bottom = z_hat[is_bottom]
            D_bottom = self.D_tilde(z_bottom)
            flux_bottom = D_bottom * ((1.0 / self.zF) * rho_z_bottom - self.c_grav * rho_bottom)
            residual[is_bottom] = flux_bottom

        return residual
