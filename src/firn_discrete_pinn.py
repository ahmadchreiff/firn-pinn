"""
PINN implementation specific to the firn PDE.
Connects the MLP + FirnProblem + discrete PDE residual.
Uses loss functions from losses.py.
"""

import torch
from .direct_pinn import DirectPINN

class FirnDiscretePINN(DirectPINN):
    def __init__(self, model, problem, data=None):
        super().__init__(model)
        self.problem = problem
        self.data = data

    def sample_collocation(self):
        """Sample (z, t) for physics loss."""
        pass

    def compute_residual(self, pts):
        """Compute firn PDE residual using FirnProblem."""
        pass

    def compute_loss(self, batch):
        """Combine physics, BC/IC, and optional data losses."""
        pass
