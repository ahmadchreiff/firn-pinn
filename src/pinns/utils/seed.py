import os
import random
import numpy as np
import torch
from typing import Optional

__all__ = ["set_seed", "seed_from_env"] # only those two functions are part of the module's public API


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set the global RNG seed for Python, NumPy, and PyTorch.

    Args:
        seed: Seed value to apply across libraries.
        deterministic: When True, enables deterministic behavior in PyTorch/cuDNN
            where supported (may reduce performance).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed) 
    np.random.seed(seed) 

    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        _enable_torch_determinism()


def seed_from_env(env_var: str = "PINN_SEED", default: int = 42, deterministic: bool = False) -> int:
    """
    Read a seed from an environment variable or fall back to a default, then apply it.

    Args:
        env_var: Name of the environment variable to read.
        default: Seed to use when the env var is unset or invalid.
        deterministic: Forwarded to ``set_seed`` to enable deterministic PyTorch behavior.

    Returns:
        The seed value that was applied.
    """
    raw_seed: Optional[str] = os.getenv(env_var)
    seed = _parse_seed(raw_seed, default)
    set_seed(seed, deterministic=deterministic)
    return seed


def _parse_seed(value: Optional[str], fallback: int) -> int:
    if value is None:
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _enable_torch_determinism() -> None:
    # cuDNN settings for repeatability
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

    # Force deterministic algorithm selection when supported by the runtime
    use_det = getattr(torch, "use_deterministic_algorithms", None)
    if callable(use_det):
        use_det(True)  # type: ignore[call-arg]
