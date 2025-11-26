# How to Understand This Repo

Recommended reading order with brief descriptions.

1. `README.md` — high-level overview, quickstart commands, repo layout.
2. `configs/defaults.yaml` (and `configs/firn_lbfgs.yaml`) — experiment settings: model, training, loss weights, problem bounds, optimizer choice.
3. `training.py` / `scripts/train.py` — thin entrypoints; see how configs are loaded and training is invoked.
4. `src/pinns/utils/training.py` — main training helper: device selection, run dir/logger, seed, model/problem/BasePINN construction, training and checkpoint save.
5. `src/pinns/problems/firn.py` — firn PDE definition: domain bounds, sampling methods, coefficients, PDE/IC/BC residuals.
6. `src/pinns/models/factory.py` and `src/pinns/models/neural_net.py` — model construction (MLP) and activation/dtype handling.
7. `src/pinns/core/base_pinn.py` — PINN wrapper: autograd derivatives, PDE/IC/BC losses, optimizer (Adam/LBFGS), training loop, predict.
8. `src/pinns/utils/plotting.py` — plotting utilities: loss curves, solution heatmaps, slices, error maps.
9. `src/pinns/utils/` (sampling, logging, seed) — support utilities used across training and plotting.
10. `tests/` — pytest sanity checks for sampling, firn problem residuals, and loss/training plumbing.
