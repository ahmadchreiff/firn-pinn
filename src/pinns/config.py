from dataclasses import dataclass, fields, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

__all__ = [
    "TrainingConfig",
    "ModelConfig",
    "ProblemConfig",
    "LossConfig",
    "Config",
    "load_config",
]

T = TypeVar("T")


@dataclass
class TrainingConfig:
    epochs: int = 1000
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    seed: int = 42
    checkpoint_every: int = 100
    batch_size: Optional[int] = None
    deterministic: bool = False


@dataclass
class ModelConfig:
    type: str = "mlp"
    in_dim: int = 1
    out_dim: int = 1
    hidden_layers: List[int] = field(default_factory=lambda: [64, 64, 64])
    activation: str = "tanh"
    dtype: str = "float32"
    device: Optional[str] = None


@dataclass
class ProblemConfig:
    name: str = "firn"
    t_min: float = 0.0
    t_max: float = 1.0
    z_min: float = 0.0
    z_max: float = 1.0
    n_interior: int = 1000
    n_boundary: int = 200
    n_initial: int = 200
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    w_pde: float = 1.0
    w_bc: float = 1.0
    w_ic: float = 1.0
    w_data: float = 1.0


@dataclass
class Config:
    training: TrainingConfig
    model: ModelConfig
    problem: ProblemConfig
    loss: LossConfig = field(default_factory=LossConfig)
    runs_dir: str = "runs"
    experiment_name: str = "default"


def load_config(path: str | Path) -> Config:
    """
    Load configuration from a YAML file into structured dataclasses.
    """
    cfg_path = Path(path)
    data = yaml.safe_load(cfg_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a mapping at the top level.")

    training = _to_dataclass(TrainingConfig, data.get("training"))
    model = _to_dataclass(ModelConfig, data.get("model"))
    problem = _to_dataclass(ProblemConfig, data.get("problem"))

    loss_data = data.get("loss")
    loss = _to_dataclass(LossConfig, loss_data) if loss_data is not None else LossConfig()

    runs_dir = data.get("runs_dir", "runs")
    experiment_name = data.get("experiment_name", "default")

    return Config(
        training=training,
        model=model,
        problem=problem,
        loss=loss,
        runs_dir=runs_dir,
        experiment_name=experiment_name,
    )


def _to_dataclass(cls: Type[T], data: Optional[Dict[str, Any]]) -> T:
    """
    Map a dictionary to a dataclass, ignoring unknown keys and using defaults.
    """
    if data is None:
        return cls()  # type: ignore[call-arg]

    field_names = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in field_names}
    return cls(**kwargs)  # type: ignore[arg-type]
