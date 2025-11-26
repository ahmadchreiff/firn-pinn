# Firn PINN

Physics-informed neural network for the rescaled firn gas-diffusion problem (Eq. 33). This repository defines the PDE, sampling, losses, training loop, and plotting utilities to solve for the dimensionless gas density `rho_hat(t_hat, z_hat)` on `[0,1] x [0,1]`.

## Quickstart

1) Install dependencies (prefer a virtualenv):
```bash
pip install -r requirements.txt
```

2) Train with default config:
```bash
python training.py --config configs/defaults.yaml
```
This creates a timestamped run directory under `runs/`, saves logs and `model.pt`.

3) Train with the LBFGS variant (optimizer support included):
```bash
python training.py --config configs/firn_lbfgs.yaml
```

4) Plot metrics or solutions using the utilities in `src/pinns/utils/plotting.py` (e.g., import and call `plot_losses_from_csv` on your `metrics.csv` if you log metrics, or `plot_solution_heatmap` on a trained model).

## Configuration

See `configs/defaults.yaml` for the standard experiment setup (MLP, sampling counts, loss weights, problem bounds). The `configs/firn_lbfgs.yaml` is a variant that requests LBFGS (supported in `BasePINN`), with adjusted epochs/lr.

## Structure (high level)

- `training.py` / `scripts/train.py`: thin entrypoints; load config, build model/problem, run training, save checkpoint.
- `configs/`: YAML configs for experiments.
- `src/pinns/problems/firn.py`: FirnProblem with domain, sampling, coefficient profiles, and PDE/BC/IC residuals.
- `src/pinns/core/base_pinn.py`: PINN wrapper to assemble losses, compute autograd derivatives, and train (Adam or LBFGS).
- `src/pinns/models/`: MLP and factory.
- `src/pinns/utils/`: sampling, logging, plotting, training helper, seeding.
- `tests/`: pytest sanity checks for sampling, firn problem, and loss/training plumbing.

## Usage notes

- Device/dtype: configured via YAML (`model.device`, `model.dtype`) or CLI `--device`.
- Optimizer: defaults to Adam; set `training.optimizer: lbfgs` to use LBFGS (closure-based) with `training.learning_rate`.
- Problem coefficients are the Table 2 defaults; replace with data-driven profiles as needed.

## Dependencies

See `requirements.txt`. Core: torch, numpy, matplotlib, pytest (for tests).
