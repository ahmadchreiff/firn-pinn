# Firn PINN

Physics-informed neural network for the rescaled firn gas-diffusion problem. This repository defines the PDE, sampling, losses, training loop, and plotting utilities to solve for the dimensionless gas density `rho_hat(t_hat, z_hat)` on `[0,1] x [0,1]`.

## Quickstart

1) Install in editable mode (prefer a virtualenv):
```bash
pip install -e .
# or include test deps:
# pip install -e .[dev]
```

2) Train with default config:
```bash
python -m pinns.utils.training --config configs/defaults.yaml
# or: python scripts/train.py --config configs/defaults.yaml
```
This creates a timestamped run directory under `runs/`, saves logs and `model.pt`.

3) Train with the LBFGS variant (optimizer support included):
```bash
python -m pinns.utils.training --config configs/firn_lbfgs.yaml
```

4) Plot metrics or solutions using the utilities in `src/pinns/utils/plotting.py` (e.g., import and call `plot_losses_from_csv` on your `metrics.csv` if you log metrics, or `plot_solution_heatmap` on a trained model). You can also run `python scripts/plot_results.py --metrics runs/<run>/metrics.csv` to quickly render loss curves.

## Configuration

See `configs/defaults.yaml` for the standard experiment setup (MLP, sampling counts, loss weights, problem bounds). The `configs/firn_lbfgs.yaml` is a variant that requests LBFGS (supported in `BasePINN`), with adjusted epochs/lr.

## Structure (high level)

- `scripts/train.py` or `python -m pinns.utils.training`: thin entrypoints; load config, build model/problem, run training, save checkpoint.
- `configs/`: YAML configs for experiments.
- `src/pinns/firn.py`: FirnProblem with domain, sampling, coefficient profiles, and PDE/BC/IC residuals.
- `src/pinns/base_pinn.py`: PINN wrapper to assemble losses, compute autograd derivatives, and train (Adam or LBFGS).
- `src/pinns/models.py`: MLP and model builder.
- `src/pinns/utils/`: sampling, logging, plotting, training helper, seeding.
- `tests/`: pytest sanity checks for sampling, firn problem, and loss/training plumbing.

## Usage notes

- Device/dtype: configured via YAML (`model.device`, `model.dtype`) or CLI `--device`.
- Optimizer: defaults to Adam; set `training.optimizer: lbfgs` to use LBFGS (closure-based) with `training.learning_rate`.
- Problem coefficients are the Table 2 defaults; replace with data-driven profiles as needed.

## Dependencies

See `requirements.txt`. Core: torch, numpy, matplotlib, pytest (for tests).
