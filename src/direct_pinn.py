"""
Generic PINN engine: sampling, training loop, and abstract methods.
Completely PDE-agnostic.
"""

import torch

class DirectPINN:
    def __init__(self, model):
        self.model = model

    def sample_collocation(self):
        """Return collocation points."""
        raise NotImplementedError

    def compute_residual(self, pts):
        """PDE residual, implemented by subclass."""
        raise NotImplementedError

    def compute_loss(self, batch):
        """Total loss, implemented by subclass."""
        raise NotImplementedError

    def train(self, steps):
        """Main training loop."""
        pass
