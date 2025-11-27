from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from pinns.config import Config, load_config
from pinns.models.factory import build_model
from pinns.problems.firn import FirnProblem
from pinns.core.base_pinn import BasePINN
from pinns.utils.logging import create_run_dir, configure_logger
from pinns.utils.seed import seed_from_env

__all__ = ["main", "parse_args", "select_device"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PINN on the firn gas-diffusion problem.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Override experiment name from config.")
    parser.add_argument("--runs-dir", type=Path, default=None, help="Override runs directory (default 'runs').")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--device", type=str, default=None, help="Device override (e.g., 'cpu', 'cuda').")
    parser.add_argument("--log-every", type=int, default=100, help="Logging frequency in epochs.")
    return parser.parse_args()


def select_device(cli_device: str | None, cfg_device: str | None) -> torch.device:
    if cli_device is not None:
        return torch.device(cli_device)
    if cfg_device is not None:
        return torch.device(cfg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()

    # 1) Load config
    cfg: Config = load_config(args.config)

    # 2) Apply CLI overrides
    if args.experiment_name is not None:
        cfg.experiment_name = args.experiment_name
    if args.runs_dir is not None:
        cfg.runs_dir = str(args.runs_dir)
    if args.epochs is not None:
        cfg.training.epochs = args.epochs

    model_device = getattr(cfg.model, "device", None)
    device = select_device(args.device, model_device)

    # 3) Run directory, logger, seed
    run_dir = create_run_dir(
        experiment=cfg.experiment_name,
        base_dir=cfg.runs_dir,
        config_path=args.config,
    )
    logger = configure_logger(run_dir)
    logger.info("Starting firn PINN training | device=%s", device)

    seed = seed_from_env(
        default=cfg.training.seed,
        deterministic=cfg.training.deterministic,
    )
    logger.info("Using seed=%s", seed)

    # 4) Problem, model, trainer
    dtype_arg = getattr(cfg.model, "dtype", None)
    dtype_resolved = dtype_arg if isinstance(dtype_arg, torch.dtype) else getattr(torch, dtype_arg, None)

    problem = FirnProblem(
        config=cfg.problem,
        device=device,
        dtype=dtype_resolved,
    )

    model: nn.Module = build_model(
        cfg.model,
        in_dim=problem.in_dim,
        out_dim=problem.out_dim,
    )

    pinn = BasePINN(
        model=model,
        problem=problem,
        training=cfg.training,
        loss_cfg=cfg.loss,
        device=device,
        logger=logger,
        run_dir=run_dir,
    )

    # 5) Train
    pinn.train(epochs=args.epochs, log_every=args.log_every)

    # 6) Save model
    model_path = run_dir / "model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model to %s", model_path)


if __name__ == "__main__":
    main()
