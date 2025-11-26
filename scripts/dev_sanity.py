"""
Run with your actual YAML config via --config path/to/config.yaml.
Tweak --num-points/--strategy to mirror real sampling settings.
"""
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pinns.config import (  # noqa: E402
    Config,
    LossConfig,
    ModelConfig,
    ProblemConfig,
    TrainingConfig,
    load_config,
)
from pinns.models.factory import build_model  # noqa: E402
from pinns.utils.logging import configure_logger, create_run_dir  # noqa: E402
from pinns.utils.sampling import sample_interior  # noqa: E402
from pinns.utils.seed import seed_from_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick integration sanity check.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config path. When omitted, a minimal default config is used.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=8,
        help="Number of dummy input points to sample.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="uniform",
        choices=["uniform", "sobol"],
        help="Sampling strategy for dummy points.",
    )
    return parser.parse_args()


def load_or_default_config(path: Optional[Path]) -> Config:
    if path:
        return load_config(path)

    return Config(
        training=TrainingConfig(epochs=1, seed=42, deterministic=False),
        model=ModelConfig(type="mlp", in_dim=2, out_dim=1, hidden_layers=[32, 32], activation="tanh"),
        problem=ProblemConfig(
            name="dev_sanity",
            t_min=0.0,
            t_max=1.0,
            z_min=0.0,
            z_max=1.0,
            n_interior=64,
            n_boundary=16,
            n_initial=16,
        ),
        loss=LossConfig(),
        runs_dir="runs",
        experiment_name="dev_sanity",
    )


def select_device(preferred: str | torch.device | None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    cfg = load_or_default_config(args.config)

    seed = seed_from_env(default=cfg.training.seed, deterministic=cfg.training.deterministic)
    device = select_device(cfg.model.device)

    bounds = [
        [cfg.problem.t_min, cfg.problem.t_max],
        [cfg.problem.z_min, cfg.problem.z_max],
    ]
    in_dim = len(bounds)
    num_points = max(1, min(args.num_points, cfg.problem.n_interior))

    run_dir = create_run_dir(
        experiment=cfg.experiment_name,
        base_dir=cfg.runs_dir,
        config_path=args.config,
    )
    logger = configure_logger(run_dir)
    logger.info("Starting dev sanity check | seed=%s | device=%s | cfg.dtype=%s", seed, device, cfg.model.dtype)

    model = build_model(cfg.model, in_dim=in_dim, out_dim=cfg.model.out_dim)
    model = model.to(device=device)
    model.eval()

    inputs = sample_interior(bounds, num_points=num_points, strategy=args.strategy, device=device)
    try:
        model_dtype = next(model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    if inputs.dtype != model_dtype:
        inputs = inputs.to(dtype=model_dtype)

    with torch.no_grad():
        outputs = model(inputs)

    logger.info(
        "Input shape: %s | Output shape: %s | dtype: %s",
        tuple(inputs.shape),
        tuple(outputs.shape),
        model_dtype,
    )
    logger.info("First input rows:\n%s", inputs[: min(3, inputs.shape[0])])
    logger.info("First output rows:\n%s", outputs[: min(3, outputs.shape[0])])
    logger.info("Sanity check complete. Run directory: %s", run_dir)


if __name__ == "__main__":
    main()
