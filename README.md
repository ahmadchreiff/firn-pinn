# Repository Overview

This repository contains the workflow for generating firn forward-model data using MATLAB and implementing a Physics-Informed Neural Network (PINN) in PyTorch.

---

## Directory Structure

### `data/`
Contains datasets used by the PINN.

- **`raw/firn_forward.mat`**  
  MATLAB-generated forward solution containing the depth grid, time grid, solution matrix, and PDE coefficients.

---

### `experiments/`
Experiment entry points.

- **`run_firn_pinn.py`**  
  Main script that loads data, constructs the model, trains the PINN, and performs evaluation.

---

### `matlab/`
MATLAB implementation of the classical firn solver and supporting utilities.

- **`COEFFc.m`** – Computes the diffusion coefficient profile.  
- **`COEFFv.m`** – Computes the advection velocity profile.  
- **`DirectPbResc.m`** – Rescaled forward PDE solver.  
- **`GenerateFirnData.m`** – Script that generates `firn_forward.mat`.  
- **`InversePbRescV.m`** – Inverse problem solver (not required for the PINN MVP).  
- **`LinearSpline.m`** – Linear interpolation helper.  
- **`Test.m`**, **`Testnoise.m`** – MATLAB test scripts.  
- **`testinv3nonDecDifMeshfmincon.m`** – Inverse solver testing with optimization.

---

### `src/`
Core PyTorch implementation of the PINN.

- **`__init__.py`**  
  Marks the directory as a Python package.

- **`direct_pinn.py`**  
  PDE-agnostic PINN engine containing the training loop, sampling routines, and abstract residual/loss interfaces.

- **`evaluation.py`**  
  Evaluation utilities for computing errors against the classical solver and generating plots.

- **`firn_data.py`**  
  Loads `firn_forward.mat` and constructs grids, classical solution matrices, and optional supervised samples.

- **`firn_discrete_pinn.py`**  
  Firn-specific PINN implementation that connects the neural network with the firn PDE and computes discrete residuals and loss terms.

- **`firn_problem.py`**  
  Defines the firn PDE, coefficients, grids, initial and boundary conditions, and finite-difference residual operator. Pure physics and discretization.

- **`losses.py`**  
  Implements physics loss, IC/BC losses, supervised loss, and weighted combination of all loss terms.

- **`model.py`**  
  PyTorch MLP defining the mapping from `(z, t)` to `V(z, t)`.

- **`utils.py`**  
  Utility functions including sampling, early stopping, and general helpers.

---

## Repository Root

- **`plot_classical.py`**  
  Standalone script for visualizing the MATLAB classical forward solution.

- **`.gitignore`**  
  Git ignore rules.

- **`requirements.txt`**  
  Python dependencies for the PINN codebase.

- **`README.md`**  
  Project documentation.

- **`venv/`**  
  Local virtual environment (untracked).
