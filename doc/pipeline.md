# PINN vs Classical Solver Pipeline

This project compares a physics‐informed neural network (PINN) against the classical MATLAB forward solver for the rescaled firn gas‑diffusion equation. The pipeline has three stages: train the PINN, generate the classical solution, then compare the two on a shared grid.

## Key components
- PINN training:
  - Configs: `configs/defaults.yaml` (baseline), `configs/firn_tight.yaml` (heavier training/capacity).
  - Trainer: `pinns.utils.training` (entry via `python -m pinns.utils.training` or `scripts/train.py`).
  - Core classes: `src/pinns/base_pinn.py`, `src/pinns/firn.py`, `src/pinns/models.py` (MLP).
  - Outputs: timestamped run dir under `runs/` with `model.pt` (checkpoint) and logs.
- Classical solver:
  - MATLAB code in `classical_solver/` (esp. `GenerateFirnData.m`).
  - Output: `data/raw/firn_forward.mat` containing `t`, `z`, and `V` (classical solution on a 129×129 grid).
- Comparison:
  - Script: `compare_firn_solutions.py` evaluates the PINN on the same (t, z) grid and plots side‑by‑side heatmaps.
  - Produces: `comparison.png`.

## Typical workflow
1) Train the PINN
   - Choose a config (e.g., `configs/firn_tight.yaml` for stronger training).
   - Run:
     ```
     python -m pinns.utils.training --config configs/firn_tight.yaml
     ```
   - Note the created folder under `runs/` (e.g., `runs/firn_tight_YYYYMMDD-HHMMSS/model.pt`).

2) Generate the classical solution
   - In MATLAB GUI: `cd` to the repo (or `classical_solver/`) and run `GenerateFirnData`. It writes `data/raw/firn_forward.mat` (creates folders if missing). It is better to run this in the MATLAB GUI.

3) Compare PINN vs classical
   - With both `model.pt` and `data/raw/firn_forward.mat` present:
     ```
     python compare_firn_solutions.py \
       --checkpoint runs/<your_run>/model.pt \
       --classical-mat data/raw/firn_forward.mat \
       --save comparison.png
     ```
   - The script loads the `.mat`, builds the PINN model matching the config, evaluates on the same grid, and saves the side‑by‑side heatmap.

## File map (what matters)
- Training: `src/pinns/base_pinn.py`, `src/pinns/firn.py`, `src/pinns/models.py`, `src/pinns/utils/training.py`, `configs/*.yaml`.
- Classical solver: `classical_solver/GenerateFirnData.m` (+ its helper .m files).
- Comparison: `compare_firn_solutions.py` (uses the PINN checkpoint and the `.mat`).
- Outputs: `runs/<name>/model.pt`, `data/raw/firn_forward.mat`, `comparison.png`.

## Notes and tips
- Use consistent parameters: the PINN and MATLAB code both assume the paper’s defaults (Te=1, zF=1, f=0.2, D_tilde=200−199.98·z, tau=10, lambda=0.03, v=200, w_air=485).
- Improving PINN fit: more epochs, stronger loss weights on PDE/BC, more collocation points, deeper/wider MLP, and LBFGS (or Adam warmup + LBFGS). `configs/firn_tight.yaml` is a starting point.
- GPU: training benefits from GPU (Colab or local). The classical solver remains MATLAB and is typically run locally. Download `model.pt` from Colab, generate `firn_forward.mat` locally, then run the comparison.
