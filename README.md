# FIRN–PINN

Physics-Informed Neural Network (PINN) implementation for the firn gas-diffusion PDE, together with the classical MATLAB forward solver used to generate reference data.

This repository provides:
- MATLAB code for the classical solution of the firn PDE.
- A modular PyTorch-based PINN framework with clean separation between core PINN logic and firn-specific physics.
- Reproducible training and evaluation pipelines.

---

## Directory Structure

### `classical_solver/`
MATLAB implementation of the forward and inverse firn solvers.

- **`COEFFc.m`** — Diffusion coefficient profile.  
- **`COEFFv.m`** — Advection velocity profile.  
- **`DirectPbResc.m`** — Rescaled forward PDE solver.  
- **`GenerateFirnData.m`** — Generates `firn_forward.mat`.  
- **`InversePbRescV.m`** — Inverse problem solver (not used in the PINN MVP).  
- **`LinearSpline.m`** — Linear interpolation routine.  
- **`Test.m`**, **`Testnoise.m`** — Forward-solver tests.  
- **`testinv3nonDecDifMeshfmincon.m`** — Inverse reconstruction test using optimization.

---

## `data/`
Datasets used by the PINN.

- **`raw/firn_forward.mat`**  
  Contains: depth grid, time grid, classical solution matrix, and physical coefficients, generated via the MATLAB solver.

---

## `experiments/`
Entry points for running experiments.

- **`run_firn_pinn.py`**  
  Loads data, instantiates the model, performs training, and runs evaluation.

---

## `src/`
Main PyTorch PINN implementation.

### `src/core/`
Framework-level PINN components (PDE-agnostic).

- **`pinn_base.py`** — Training engine: sampling, optimization loop, and abstract residual hooks.  
- **`model.py`** — MLP mapping `(z, t) → V(z, t)`.  
- **`losses.py`** — Physics, boundary/initial condition, and supervised loss terms.  
- **`utils.py`** — Helper utilities (sampling, scheduling, early stopping, etc.).

### `src/firn/`
Firn-specific physics and PINN assembly.

- **`firn_problem.py`** — Firn PDE definition: coefficients, grids, IC/BC, and discrete finite-difference residual operator.  
- **`firn_data.py`** — Loads `firn_forward.mat` and exposes data structures for training and evaluation.  
- **`firn_pinn.py`** — Connects:  
  MLP model + firn physics + base PINN engine → complete firn PINN model.

### `src/evaluation/`
Evaluation and visualization tools.

- **`metrics.py`** — Error metrics comparing PINN predictions to the classical solver.  
- **`plotting.py`** — Plot generation for predictions, losses, and residuals.

---

## Repository Root

- **`plot_classical.py`** — Quick visualization of the MATLAB forward solution.  
- **`requirements.txt`** — Python dependencies.  
- **`.gitignore`** — Git ignore rules.  
- **`README.md`** — Project documentation.  
- **`venv/`** — Local virtual environment (not tracked).

---

## Summary

This architecture cleanly separates:
- **Physics (firn PDE)**  
- **PINN framework (core)**  
- **Experiments and reproducibility**  
- **Classical solver (MATLAB baseline)**

It supports a robust MVP now while remaining extensible for inverse PINNs, advanced losses, and alternative architectures.
