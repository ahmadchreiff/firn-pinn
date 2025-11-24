from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

import torch
from torch import nn

__all__ = ["MLP", "get_activation"]


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


def _xavier_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
