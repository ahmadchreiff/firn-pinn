from __future__ import annotations

"""
Evaluate PINN accuracy against the classical solver output on the same (t, z) grid.

Usage example:
python evaluate_pinn_error.py \
  --config runs/<run_name>/<config_used_to_train>.yaml \
  --checkpoint runs/<run_name>/model.pt \
  --classical-mat data/raw/firn_forward.mat \
  --num-samples 2000
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

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


def _require_loadmat():
    try:
        from scipy.io import loadmat  # type: ignore
    except ImportError as exc:
        raise SystemExit("scipy is required to read the classical .mat file. Install with `pip install scipy`.") from exc
    return loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PINN error against classical solver output.")
    parser.add_argument("--config", type=Path, required=True, help="YAML config used for training the PINN.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to trained model state_dict (model.pt).")
    parser.add_argument("--classical-mat", type=Path, required=True, help="Path to classical solver .mat file.")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference (cpu or cuda).")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of random points to sample for reporting.")
    return parser.parse_args()


def load_classical(mat_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loadmat = _require_loadmat()
    mat = loadmat(mat_path)
    t = np.asarray(mat["t"]).reshape(-1)
    z = np.asarray(mat["z"]).reshape(-1)
    V = np.asarray(mat["V"])
    if V.shape != (len(z), len(t)):
        raise ValueError(f"Unexpected V shape {V.shape}; expected ({len(z)}, {len(t)})")
    V_plot = V.T  # shape (m, n): time-major
    return t, z, V_plot


def load_pinn(config_path: Path, checkpoint: Path, device: torch.device) -> Tuple[torch.nn.Module, FirnProblem]:
    cfg = load_config(config_path)
    dtype_arg = getattr(cfg.model, "dtype", None)
    dtype_resolved = dtype_arg if isinstance(dtype_arg, torch.dtype) else getattr(torch, str(dtype_arg), None)

    problem = FirnProblem(config=cfg.problem, device=device, dtype=dtype_resolved)
    model = build_model(cfg.model, in_dim=problem.in_dim, out_dim=problem.out_dim)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device=device).eval()
    return model, problem


@torch.no_grad()
def evaluate_full_grid(
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


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device is not None else torch.device("cpu")

    t_vals, z_vals, ref = load_classical(args.classical_mat)
    model, problem = load_pinn(args.config, args.checkpoint, device)
    pred = evaluate_full_grid(model, problem, t_vals, z_vals, device)

    err = pred - ref
    l2 = float(np.sqrt((err ** 2).mean()))
    linf = float(np.abs(err).max())
    mae = float(np.abs(err).mean())
    print(f"Full-grid errors: L2={l2:.6e}, MAE={mae:.6e}, Linf={linf:.6e}")

    if args.num_samples > 0:
        m, n = ref.shape
        rng = np.random.default_rng()
        idx_flat = rng.choice(m * n, size=min(args.num_samples, m * n), replace=False)
        t_idx = idx_flat // n
        z_idx = idx_flat % n
        sampled_err = err[t_idx, z_idx]
        l2_s = float(np.sqrt((sampled_err ** 2).mean()))
        linf_s = float(np.abs(sampled_err).max())
        mae_s = float(np.abs(sampled_err).mean())
        print(
            f"Sampled ({len(sampled_err)} pts) errors: L2={l2_s:.6e}, "
            f"MAE={mae_s:.6e}, Linf={linf_s:.6e}"
        )


if __name__ == "__main__":
    main()
