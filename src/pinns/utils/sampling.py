import torch

from typing import Iterable, Sequence


__all__ = [
    "sample_uniform",
    "sample_sobol",
    "sample_points",
    "sample_interior",
    "sample_boundary",
    "sample_initial",
]


def sample_uniform(bounds: Sequence[Sequence[float]], num_points: int, device: str | torch.device | None = None) -> torch.Tensor:
    """
    Sample uniformly within a box defined by bounds.

    Args:
        bounds: Sequence of [min, max] for each dimension.
        num_points: Number of points to sample.
        device: Optional torch device.
    """
    bounds_tensor = _as_tensor(bounds, device=device)
    low, high = bounds_tensor[:, 0], bounds_tensor[:, 1]
    dim = low.shape[0]
    samples = torch.rand((num_points, dim), device=device)
    return low + (high - low) * samples


def sample_sobol(bounds: Sequence[Sequence[float]], num_points: int, device: str | torch.device | None = None, scramble: bool = True) -> torch.Tensor:
    """
    Sample using a Sobol sequence and rescale to the given bounds.

    Args:
        bounds: Sequence of [min, max] for each dimension.
        num_points: Number of points to sample.
        device: Optional torch device.
        scramble: Whether to scramble the Sobol sequence.
    """
    bounds_tensor = _as_tensor(bounds, device=device)
    dim = bounds_tensor.shape[0]
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=scramble)
    base = engine.draw(num_points).to(device=device)
    low, high = bounds_tensor[:, 0], bounds_tensor[:, 1]
    return low + (high - low) * base


def sample_points(
    bounds: Sequence[Sequence[float]],
    num_points: int,
    strategy: str = "uniform",
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Sample points in a box using a chosen strategy.

    Args:
        bounds: Sequence of [min, max] for each dimension.
        num_points: Number of points to sample.
        strategy: One of {"uniform", "sobol"}.
        device: Optional torch device.
    """
    strategy = strategy.lower()
    if strategy == "uniform":
        return sample_uniform(bounds, num_points, device=device)
    if strategy == "sobol":
        return sample_sobol(bounds, num_points, device=device)
    raise ValueError(f"Unknown sampling strategy: {strategy}")


def sample_interior(
    bounds: Sequence[Sequence[float]],
    num_points: int,
    strategy: str = "uniform",
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Sample interior collocation points."""
    return sample_points(bounds, num_points, strategy=strategy, device=device)


def sample_boundary(
    bounds: Sequence[Sequence[float]],
    num_points: int,
    fixed_coords: Iterable[int],
    fixed_values: Iterable[float],
    strategy: str = "uniform",
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Sample boundary points by fixing specified coordinates to given values
    and sampling the rest within their bounds.

    Args:
        bounds: Sequence of [min, max] for each dimension.
        num_points: Number of points to generate.
        fixed_coords: Iterable of dimension indices to fix (e.g., [0] for t=const).
        fixed_values: Values to assign to those fixed coords (same length as fixed_coords).
        strategy: Sampling strategy for the free coordinates.
        device: Optional torch device.
    """
    bounds_tensor = _as_tensor(bounds, device=device)
    dim = bounds_tensor.shape[0]

    fixed_indices = list(fixed_coords)
    fixed_vals = list(fixed_values)
    if len(fixed_indices) != len(fixed_vals):
        raise ValueError("fixed_coords and fixed_values must have the same length")

    free_indices = [i for i in range(dim) if i not in fixed_indices]

    points = torch.empty((num_points, dim), device=device)

    # Sample free coordinates
    if free_indices:
        free_bounds = bounds_tensor[free_indices]
        free_samples = sample_points(free_bounds, num_points, strategy=strategy, device=device)
        for j, idx in enumerate(free_indices):
            points[:, idx] = free_samples[:, j]

    # Set fixed coordinates
    for idx, val in zip(fixed_indices, fixed_vals):
        points[:, idx] = torch.as_tensor(val, device=device).expand(num_points)

    return points


def sample_initial(
    bounds: Sequence[Sequence[float]],
    num_points: int,
    time_dim: int = 0,
    time_value: float | None = None,
    strategy: str = "uniform",
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Sample initial-condition points: fix the time coordinate and sample spatial dims.

    Args:
        bounds: Sequence of [min, max] for each dimension (time first by default).
        num_points: Number of points to generate.
        time_dim: Index of the time coordinate in bounds.
        time_value: Value to fix time at; defaults to lower bound for time_dim.
        strategy: Sampling strategy for spatial coordinates.
        device: Optional torch device.
    """
    bounds_tensor = _as_tensor(bounds, device=device)
    if time_value is None:
        time_value = float(bounds_tensor[time_dim, 0].item())
    return sample_boundary(
        bounds=bounds_tensor,
        num_points=num_points,
        fixed_coords=[time_dim],
        fixed_values=[time_value],
        strategy=strategy,
        device=device,
    )


def _as_tensor(bounds: Sequence[Sequence[float]], device: str | torch.device | None = None) -> torch.Tensor:
    if isinstance(bounds, torch.Tensor):
        tensor = bounds
    else:
        tensor = torch.tensor(bounds, dtype=torch.float32)
    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise ValueError("bounds must be shape [dim, 2] or iterable of [min, max] per dimension")
    if device is not None:
        tensor = tensor.to(device)
    return tensor
