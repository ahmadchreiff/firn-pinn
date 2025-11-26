from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

from pinns.problems.firn import FirnProblem

__all__ = [
    "plot_losses",
    "plot_losses_from_csv",
    "plot_solution_heatmap",
    "plot_depth_slice",
    "plot_time_slice",
    "plot_error_heatmap",
]


def _finalize(fig: plt.Figure, save_path: Optional[Path | str]) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_losses(history: Mapping[str, Sequence[float]], save_path: Optional[Path | str] = None) -> None:
    """
    Plot loss curves over epochs.

    Args:
        history: Mapping from loss name to list/sequence of values per epoch.
        save_path: Optional path to save the plot. If None, displays interactively.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = range(1, len(next(iter(history.values()))) + 1) if history else []

    for name, values in history.items():
        ax.plot(epochs, values, label=name)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _finalize(fig, save_path)


def plot_losses_from_csv(csv_path: Path | str, save_path: Optional[Path | str] = None) -> None:
    """
    Read a CSV metrics file (e.g., metrics.csv) and plot loss curves.
    Assumes header row with loss names; plots all columns.
    """
    path = Path(csv_path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in metrics file: {csv_path}")

    history: Dict[str, list[float]] = {key: [] for key in rows[0].keys()}
    for row in rows:
        for key, val in row.items():
            history[key].append(float(val))

    plot_losses(history, save_path=save_path)


@torch.no_grad()
def plot_solution_heatmap(
    model: nn.Module,
    problem: FirnProblem,
    device: torch.device,
    nt: int = 100,
    nz: int = 100,
    save_path: Optional[Path | str] = None,
) -> None:
    """
    Plot a 2D heatmap of the solution rho_hat(t_hat, z_hat).
    """
    model.eval()
    bounds = problem.bounds()
    t_min, t_max = bounds[0]
    z_min, z_max = bounds[1]

    t_lin = torch.linspace(t_min, t_max, nt, device=device)
    z_lin = torch.linspace(z_min, z_max, nz, device=device)
    tt, zz = torch.meshgrid(t_lin, z_lin, indexing="ij")
    grid = torch.stack([tt.reshape(-1), zz.reshape(-1)], dim=1)

    rho = model(grid).detach().cpu().reshape(nt, nz)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        rho,
        origin="lower",
        extent=[z_min, z_max, t_min, t_max],
        aspect="auto",
        cmap="viridis",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\hat{\\rho}$")
    ax.set_xlabel(r"$\hat{z}$")
    ax.set_ylabel(r"$\hat{t}$")
    ax.set_title("PINN solution heatmap")

    _finalize(fig, save_path)


@torch.no_grad()
def plot_depth_slice(
    model: nn.Module,
    problem: FirnProblem,
    device: torch.device,
    t_hat: float,
    nz: int = 100,
    save_path: Optional[Path | str] = None,
) -> None:
    """
    Plot rho_hat(z_hat) at fixed time t_hat.
    """
    model.eval()
    z_min, z_max = problem.bounds()[1]
    z_lin = torch.linspace(z_min, z_max, nz, device=device)
    t_vals = torch.full_like(z_lin, float(t_hat))
    grid = torch.stack([t_vals, z_lin], dim=1)

    rho = model(grid).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(z_lin.cpu().numpy(), rho, label=fr"$\hat{{t}}={t_hat}$")
    ax.set_xlabel(r"$\hat{z}$")
    ax.set_ylabel(r"$\hat{\rho}$")
    ax.set_title("Depth slice")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _finalize(fig, save_path)


@torch.no_grad()
def plot_time_slice(
    model: nn.Module,
    problem: FirnProblem,
    device: torch.device,
    z_hat: float,
    nt: int = 100,
    save_path: Optional[Path | str] = None,
) -> None:
    """
    Plot rho_hat(t_hat) at fixed depth z_hat.
    """
    model.eval()
    t_min, t_max = problem.bounds()[0]
    t_lin = torch.linspace(t_min, t_max, nt, device=device)
    z_vals = torch.full_like(t_lin, float(z_hat))
    grid = torch.stack([t_lin, z_vals], dim=1)

    rho = model(grid).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_lin.cpu().numpy(), rho, label=fr"$\hat{{z}}={z_hat}$")
    ax.set_xlabel(r"$\hat{t}$")
    ax.set_ylabel(r"$\hat{\rho}$")
    ax.set_title("Time slice")
    ax.grid(True, alpha=0.3)
    ax.legend()

    _finalize(fig, save_path)


def plot_error_heatmap(
    pred: Tensor | torch.Tensor,
    ref: Tensor | torch.Tensor,
    t_lin: Sequence[float],
    z_lin: Sequence[float],
    save_path: Optional[Path | str] = None,
) -> None:
    """
    Plot absolute error heatmap between predicted and reference fields on the same grid.

    Args:
        pred: Predicted values shaped [nt, nz] or flattenable to that grid.
        ref: Reference values shaped [nt, nz].
        t_lin: 1D sequence of time coordinates (length nt).
        z_lin: 1D sequence of depth coordinates (length nz).
    """
    pred_np = torch.as_tensor(pred).cpu().numpy()
    ref_np = torch.as_tensor(ref).cpu().numpy()
    if pred_np.shape != ref_np.shape:
        raise ValueError(f"pred and ref shapes must match, got {pred_np.shape} vs {ref_np.shape}")

    nt, nz = pred_np.shape
    if len(t_lin) != nt or len(z_lin) != nz:
        raise ValueError("t_lin and z_lin lengths must match pred/ref grid dimensions.")

    err = (pred_np - ref_np).__abs__()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        err,
        origin="lower",
        extent=[z_lin[0], z_lin[-1], t_lin[0], t_lin[-1]],
        aspect="auto",
        cmap="magma",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Absolute error")
    ax.set_xlabel(r"$\hat{z}$")
    ax.set_ylabel(r"$\hat{t}$")
    ax.set_title("Error heatmap")

    _finalize(fig, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plotting utilities demo placeholder.")
    parser.add_argument("--metrics", type=Path, default=None, help="Path to metrics CSV to plot losses.")
    args = parser.parse_args()
    if args.metrics is not None:
        plot_losses_from_csv(args.metrics)
