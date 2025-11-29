from __future__ import annotations

from typing import Any, Dict, Optional

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
    Rescaled firn gas-diffusion problem (Eq. 33) on (t_hat, z_hat) ∈ [0, 1]².
    Encodes the Table 2 constants from the classical solver for the PDE, IC, and BCs.
    """

    def __init__(
        self,
        config: Optional[ProblemConfig] = None,
        *,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        z_min: Optional[float] = None,
        z_max: Optional[float] = None,
        f_const: Optional[float] = None,
        adv_velocity: Optional[float] = None,
        sink_rate: Optional[float] = None,
        D_a0: Optional[float] = None,
        D_a1: Optional[float] = None,
        c_grav: Optional[float] = None,
        Te: Optional[float] = None,
        zF: Optional[float] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        params: Dict[str, Any] = getattr(config, "parameters", {}) if config is not None else {}

        def _pick(name: str, override: Optional[float], default: float) -> float:
            if override is not None:
                return float(override)
            if name in params:
                return float(params[name])
            return float(default)

        self.t_min = float(t_min if t_min is not None else getattr(config, "t_min", 0.0))
        self.t_max = float(t_max if t_max is not None else getattr(config, "t_max", 1.0))
        self.z_min = float(z_min if z_min is not None else getattr(config, "z_min", 0.0))
        self.z_max = float(z_max if z_max is not None else getattr(config, "z_max", 1.0))

        # Table 2 constants (with optional overrides)
        self.Te = _pick("Te", Te, 1.0)
        self.zF = _pick("zF", zF, 1.0)
        self.f_const = _pick("f_const", f_const, 0.2)  # open porosity f
        self.adv_velocity = _pick("adv_velocity", adv_velocity, 685.0)  # v + w_air
        self.sink_rate = _pick("sink_rate", sink_rate, 10.03)  # tau + lambda
        self.D_a0 = _pick("D_a0", D_a0, 200.0)  # D_alpha(0)
        self.D_a1 = _pick("D_a1", D_a1, 199.98)  # slope in D_alpha(z) = D_a0 - D_a1*z
        # M_alpha * g / (R * T) with M_alpha=0.04, g=9.8, R=8.314, T=260 => ~1.813e-4
        self.c_grav = _pick("c_grav", c_grav, 1.8134e-4)

        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = self._resolve_dtype(dtype)

        self.in_dim = 2  # (t_hat, z_hat)
        self.out_dim = 1  # rho_hat

    # ------------------------------------------------------------------
    # Domain helpers
    # ------------------------------------------------------------------
    def bounds(self) -> list[list[float]]:
        return [[self.t_min, self.t_max], [self.z_min, self.z_max]]

    @staticmethod
    def split_inputs(x: Tensor) -> tuple[Tensor, Tensor]:
        return x[:, 0:1], x[:, 1:2]

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_interior(self, num_points: int, strategy: str = "sobol", device: Optional[torch.device | str] = None) -> Tensor:
        pts = sample_interior_points(self.bounds(), num_points, strategy=strategy, device=device or self.device)
        return pts.to(dtype=self.dtype)

    def sample_initial(self, num_points: int, strategy: str = "sobol", device: Optional[torch.device | str] = None) -> Tensor:
        pts = sample_initial_points(
            bounds=self.bounds(),
            num_points=num_points,
            time_dim=0,
            time_value=self.t_min,
            strategy=strategy,
            device=device or self.device,
        )
        return pts.to(dtype=self.dtype)

    def sample_boundary(self, num_points: int, strategy: str = "sobol", device: Optional[torch.device | str] = None) -> Tensor:
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
        return torch.cat([top, bottom], dim=0).to(dtype=self.dtype)

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def diffusivity(self, z_hat: Tensor) -> Tensor:
        return self.D_a0 - self.D_a1 * z_hat

    def rho_atm(self, t_hat: Tensor) -> Tensor:
        return 2.0 * (self.Te * t_hat).pow(0.25)

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
        PDE residual for Eq. (33):
        (1/Te) * ∂(rho*f)/∂t + (1/zF) * ∂/∂z [rho*f*(v+w)] + rho*(tau+lambda)
        = (1/zF) * ∂/∂z [ D(z) * ( (1/zF) ∂rho/∂z - rho * c_grav ) ].
        """
        _, z_hat = self.split_inputs(inputs)
        f_val = torch.as_tensor(self.f_const, device=z_hat.device, dtype=z_hat.dtype)

        time_term = (1.0 / self.Te) * f_val * rho_t
        adv_term = (f_val * self.adv_velocity / self.zF) * rho_z
        react_term = self.sink_rate * rho

        D_val = self.diffusivity(z_hat)
        phi = (1.0 / self.zF) * rho_z - rho * self.c_grav
        phi_z = (1.0 / self.zF) * rho_zz - rho_z * self.c_grav
        diff_term = (1.0 / self.zF) * ((-self.D_a1) * phi + D_val * phi_z)

        return time_term + adv_term + react_term - diff_term

    def initial_residual(self, inputs: Tensor, rho: Tensor) -> Tensor:
        return rho  # rho(t_min, z_hat) = 0

    def boundary_residual(self, inputs: Tensor, rho: Tensor, rho_z: Tensor) -> Tensor:
        """
        Dirichlet at top (z=z_min): rho = rho_atm(t); Neumann/flux at bottom (z=z_max):
        D(1) * ((1/zF) * rho_z - c_grav * rho) = 0.
        """
        t_hat, z_hat = self.split_inputs(inputs)
        residual = torch.zeros_like(rho)

        z_min_t = torch.as_tensor(self.z_min, device=z_hat.device, dtype=z_hat.dtype)
        z_max_t = torch.as_tensor(self.z_max, device=z_hat.device, dtype=z_hat.dtype)

        is_top = torch.isclose(z_hat, z_min_t, atol=1e-6)
        is_bottom = torch.isclose(z_hat, z_max_t, atol=1e-6)

        if is_top.any():
            residual[is_top] = rho[is_top] - self.rho_atm(t_hat[is_top])

        if is_bottom.any():
            D_bottom = self.diffusivity(z_hat[is_bottom])
            flux_bottom = D_bottom * ((1.0 / self.zF) * rho_z[is_bottom] - self.c_grav * rho[is_bottom])
            residual[is_bottom] = flux_bottom

        return residual

    @staticmethod
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
