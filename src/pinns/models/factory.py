from __future__ import annotations

from dataclasses import fields
from typing import Any, Mapping

import torch
from torch import nn

from pinns.config import ModelConfig
from pinns.models.neural_net import MLP

__all__ = ["build_model"]


def build_model(
    model_config: ModelConfig | Mapping[str, Any],
    in_dim: int,
    out_dim: int,
) -> nn.Module:
    """
    Construct a model instance from a config object or mapping.

    Args:
        model_config: Model configuration dataclass or dictionary.
        in_dim: Input dimension for the model.
        out_dim: Output dimension for the model.

    Returns:
        Initialized torch.nn.Module.

    Raises:
        ValueError: If the requested model type is unsupported.
        TypeError: If ``model_config`` is not a mapping or ModelConfig.
    """
    cfg = _coerce_model_config(model_config)
    model_type = cfg.type.lower()

    if model_type == "mlp":
        model: nn.Module = MLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_layers=cfg.hidden_layers,
            activation=cfg.activation,
        )
    else:
        raise ValueError(f"Unsupported model type: {cfg.type}")

    dtype = _resolve_dtype(getattr(cfg, "dtype", None))
    device = getattr(cfg, "device", None)
    if dtype is not None or device is not None:
        model = model.to(device=device, dtype=dtype)

    return model


def _coerce_model_config(config: ModelConfig | Mapping[str, Any]) -> ModelConfig:
    """
    Normalize user-provided config into a ModelConfig instance.
    """
    if isinstance(config, ModelConfig):
        return config
    if isinstance(config, Mapping):
        allowed = {f.name for f in fields(ModelConfig)}
        kwargs = {k: v for k, v in config.items() if k in allowed}
        return ModelConfig(**kwargs)
    raise TypeError("model_config must be a ModelConfig or a mapping.")


def _resolve_dtype(dtype: Any) -> torch.dtype | None:
    """
    Convert a dtype specification into a torch.dtype, when possible.
    """
    if dtype is None:
        return None
    if isinstance(dtype, torch.dtype):
        return dtype

    key = str(dtype).lower()
    if key in ("float32", "float", "fp32"):
        return torch.float32
    if key in ("float64", "double", "fp64"):
        return torch.float64
    if key in ("float16", "half", "fp16"):
        return torch.float16
    if key in ("bfloat16", "bf16"):
        return torch.bfloat16

    raise ValueError(f"Unsupported dtype: {dtype}")
