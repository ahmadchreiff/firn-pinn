## Proposed Structure

```
.                         # repo root
├── README.md             # quickstart, examples, environment setup
├── requirements.txt      # minimal runtime deps (torch, numpy, matplotlib, hydra-core/omegaconf optional)
├── pyproject.toml        # (optional) package metadata if you want `pip install -e .`
├── configs/
│   ├── defaults.yaml     # base training + paths; hydra/omegaconf style
│   └── firn_lbfgs.yaml   # example experiment config (points, optimizer, patience, logging)
├── scripts/
│   ├── train.py          # CLI entrypoint: loads config, builds problem, runs training, saves artifacts
│   └── plot_results.py   # optional: load checkpoint + metrics and render plots
├── src/
│   └── pinns/
│       ├── __init__.py
│       ├── config.py             # dataclasses/schemas for training + optimizer configs
│       ├── models/
│       │   ├── neural_net.py     # MLP with init, dtype/device handling, activation selection
│       │   └── factory.py        # helper to build networks from config
│       ├── core/
│       │   └── base_pinn.py      # generalized DirectPINN: sampling, loss aggregation, training loop
│       ├── problems/
│       │   └── firn.py           # FirnProblem implementing PDE residual + BCs + constants
│       └── utils/
│           ├── sampling.py       # Sobol/LHS point generation, domain mapping
│           ├── plotting.py       # training curves, solution scatter/heatmaps (headless-safe)
│           ├── logging.py        # console + file logging, timestamped run dirs
│           ├── training.py       # CLI helper: load config, build problem/model, run training, save artifacts
│           └── seed.py           # reproducibility helpers
├── tests/
│   ├── test_sampling.py          # shapes/ranges for generated points
│   ├── test_losses.py            # loss components finite-difference sanity checks
│   └── test_firn_problem.py      # residual/BC shape tests, dtype/device coverage
├── classical_solver/             # existing MATLAB solver; keep for baseline/data gen (run via MATLAB/Octave)
└── outputs/                      # auto-created; run-specific dirs for checkpoints/plots/metrics
```
