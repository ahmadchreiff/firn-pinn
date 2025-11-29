from __future__ import annotations

"""
Compare the classical finite-element solver output against a trained PINN.

This script will:
  1) Generate or load the classical solution via the MATLAB/Octave code in classical_solver/GenerateFirnData.m
  2) Load a trained PINN checkpoint and evaluate it on the same (t, z) grid
  3) Plot side-by-side heatmaps for visual comparison
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Allow running without installing the package
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinns.config import load_config
from pinns.models import build_model
from pinns.firn import FirnProblem


def _require_scipy_loadmat():
    try:
        from scipy.io import loadmat  # type: ignore
    except ImportError as exc:  # pragma: no cover - convenience for users without scipy
        raise SystemExit(
            "scipy is required to load the classical solver output (.mat). "
            "Install it with `pip install scipy`."
        ) from exc
    return loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare classical solver output to PINN predictions.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "defaults.yaml",
        help="YAML config used to build the PINN (default: configs/defaults.yaml).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a trained PINN checkpoint (state_dict).",
    )
    parser.add_argument(
        "--classical-mat",
        type=Path,
        default=ROOT / "data" / "raw" / "firn_forward.mat",
        help="Path to the classical solver output (.mat).",
    )
    parser.add_argument(
        "--run-classical",
        action="store_true",
        help="Run the classical solver (Octave/MATLAB) to generate the .mat file when missing.",
    )
    parser.add_argument(
        "--run-octave",
        action="store_true",
        help=argparse.SUPPRESS,  # backwards compatibility shim
    )
    parser.add_argument(
        "--solver-kind",
        choices=["octave", "matlab"],
        default="octave",
        help="Which engine to use for the classical solver (default: octave).",
    )
    parser.add_argument(
        "--solver-bin",
        type=str,
        default=None,
        help="Executable path/name for the solver. Defaults to 'octave' or 'matlab' based on solver-kind.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for the PINN (e.g., cpu, cuda). Defaults to config or CPU.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=ROOT / "comparison.png",
        help="Where to save the comparison figure.",
    )
    return parser.parse_args()


def run_classical_solver(mat_path: Path, solver_kind: str, solver_bin: str | None) -> None:
    """
    Run the classical solver via Octave or MATLAB to generate firn_forward.mat.
    """
    raw_dir = mat_path.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    executable = solver_bin or ("octave" if solver_kind == "octave" else "matlab")
    if shutil.which(executable) is None:
        raise SystemExit(
            f"Executable '{executable}' not found. Install {solver_kind} or point --solver-bin to the binary."
        )

    if solver_kind == "octave":
        cmd = [
            executable,
            "--quiet",
            "--eval",
            "cd classical_solver; GenerateFirnData; exit",
        ]
    else:
        # MATLAB batch mode (headless). Tested with R2019a+; adjust flags if needed.
        cmd = [
            executable,
            "-batch",
            "cd('classical_solver'); GenerateFirnData; exit;",
        ]

    print(f"Running classical solver via: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=ROOT)

    if not mat_path.exists():
        raise SystemExit(f"{solver_kind} ran, but {mat_path} was not created.")


def load_classical_solution(mat_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load z, t, V (solution) from the MATLAB .mat file.
    Returns t (length m), z (length n), and V with shape (m, n) for plotting.
    """
    loadmat = _require_scipy_loadmat()
    mat = loadmat(mat_path)
    z = np.asarray(mat["z"]).reshape(-1)
    t = np.asarray(mat["t"]).reshape(-1)
    V = np.asarray(mat["V"])
    if V.shape != (len(z), len(t)):
        raise ValueError(f"Unexpected V shape {V.shape}; expected ({len(z)}, {len(t)})")
    V_plot = V.T  # shape (m, n): time-major for plotting with extent
    return t, z, V_plot


def load_pinn(config_path: Path, checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, FirnProblem]:
    cfg = load_config(config_path)

    dtype_arg = getattr(cfg.model, "dtype", None)
    dtype_resolved = dtype_arg if isinstance(dtype_arg, torch.dtype) else getattr(torch, str(dtype_arg), None)

    problem = FirnProblem(
        config=cfg.problem,
        device=device,
        dtype=dtype_resolved,
    )
    model = build_model(cfg.model, in_dim=problem.in_dim, out_dim=problem.out_dim)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device=device)
    model.eval()
    return model, problem


@torch.no_grad()
def evaluate_pinn_on_grid(
    model: torch.nn.Module,
    problem: FirnProblem,
    t_vals: np.ndarray,
    z_vals: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    t_tensor = torch.tensor(t_vals, device=device, dtype=problem.dtype)
    z_tensor = torch.tensor(z_vals, device=device, dtype=problem.dtype)
    tt, zz = torch.meshgrid(t_tensor, z_tensor, indexing="ij")
    grid = torch.stack([tt.reshape(-1), zz.reshape(-1)], dim=1)
    rho = model(grid).reshape(tt.shape)
    return rho.detach().cpu().numpy()


def plot_heatmaps(t: np.ndarray, z: np.ndarray, classical: np.ndarray, pinn: np.ndarray, save_path: Path) -> None:
    vmin = min(classical.min(), pinn.min())
    vmax = max(classical.max(), pinn.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    extent = [z.min(), z.max(), t.min(), t.max()]

    im0 = axes[0].imshow(classical, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax, cmap="turbo")
    axes[0].set_title("Classical Solver")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(pinn, origin="lower", extent=extent, aspect="auto", vmin=vmin, vmax=vmax, cmap="turbo")
    axes[1].set_title("PINN Approximation")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("t")
    fig.colorbar(im1, ax=axes[1])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {save_path}")


def main() -> None:
    args = parse_args()

    device = torch.device(args.device) if args.device is not None else torch.device("cpu")

    mat_path = args.classical_mat
    run_solver = args.run_classical or args.run_octave
    if not mat_path.exists():
        if run_solver:
            run_classical_solver(mat_path, args.solver_kind, args.solver_bin)
        else:
            raise SystemExit(
                f"Classical solver output not found at {mat_path}. "
                "Pass --run-classical to generate it or provide an existing file via --classical-mat."
            )

    t_vals, z_vals, V_classical = load_classical_solution(mat_path)
    model, problem = load_pinn(args.config, args.checkpoint, device)
    V_pinn = evaluate_pinn_on_grid(model, problem, t_vals, z_vals, device)

    plot_heatmaps(t_vals, z_vals, V_classical, V_pinn, args.save)


if __name__ == "__main__":
    main()
