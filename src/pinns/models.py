from __future__ import annotations

from dataclasses import fields
from typing import Any, Callable, Iterable, List, Mapping, Sequence

import torch
from torch import nn

from pinns.config import ModelConfig

__all__ = ["MLP", "get_activation", "build_model"]


def get_activation(name: str) -> nn.Module:
    """
    Map a string name to a torch activation module.

    Supported: relu, leaky_relu, tanh, sigmoid, gelu, silu (swish).
    """
    key = name.lower()
    if key == "relu":
        return nn.ReLU()
    if key in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "gelu":
        return nn.GELU()
    if key in ("silu", "swish"):
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    """
    Fully connected MLP: Linear + activation blocks ending with a linear head.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: Sequence[int] | Iterable[int],
        activation: str = "tanh",
        init_fn: Callable[[nn.Module], None] | None = None,
    ) -> None:
        super().__init__()

        hidden_sizes: List[int] = list(hidden_layers)
        if len(hidden_sizes) == 0:
            layers: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
        else:
            act = get_activation(activation)
            layers = []
            prev = in_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(prev, h))
                layers.append(act if isinstance(act, nn.Module) else get_activation(activation))
                prev = h
            layers.append(nn.Linear(prev, out_dim))

        self.net = nn.Sequential(*layers)

        # Apply initialization if provided; default to Xavier on Linear layers.
        if init_fn is None:
            self.apply(_xavier_init)
        else:
            self.apply(init_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


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


def _xavier_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
